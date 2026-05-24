#!/bin/bash
# Submit the downstream-eval pipeline for a single backbone (misalignment_detection only).
#
# If CPT sub-stage 2 + HCL sub-stage 2 adapters already exist locally under
#   outputs/runs/<safe_model>/{cpt_substage2,hcl_substage2}/final_model/
# the jobs run as soon as the queue lets them. Otherwise, pass --dep-on <JOBID>
# so the whole batch waits on the HCL sub-stage 2 job that's still producing
# those adapters.
#
# Submits 5 inference jobs + 1 SFT training + 1 SFT inference:
#
#   A  zero-shot, raw backbone                         (1xH100, ~10 min)
#   B  zero-shot, CPT-merged + HCL attached            (1xH100, ~10 min)
#   C  few-shot k=5, raw backbone                      (1xH100, ~15 min)
#   D  few-shot k=5, CPT-merged + HCL attached         (1xH100, ~15 min)
#   E  SFT training on sft_align, chained on CPT+HCL   (1xH100, ~hours)
#   F  SFT inference (afterok:E)                       (1xH100, ~10 min)
#
# Usage:
#   bash scripts/submit_downstream_quartz.sh [model_name] [n_shot] [--dep-on JOBID]
# Defaults: Foundation-Sec-8B-Reasoning, k=5, no dep

set -euo pipefail

MODEL="${1:-Foundation-Sec-8B-Reasoning}"
N_SHOT="${2:-5}"
DEP_JOBID=""
if [[ "${3:-}" == "--dep-on" ]]; then
  DEP_JOBID="${4:?--dep-on requires a job id}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INFER_LAUNCHER="${SCRIPT_DIR}/run_infer_quartz.sh"
SFT_LAUNCHER="${SCRIPT_DIR}/run_sft_quartz.sh"

cd "${REPO_ROOT}"

SAFE_MODEL="$(printf '%s' "${MODEL}" | sed 's|/|__|g; s| |_|g')"
OUT_ROOT="${REPO_ROOT}/outputs/runs"
CPT_ADAPTER="${OUT_ROOT}/${SAFE_MODEL}/cpt_substage2/final_model"
HCL_ADAPTER="${OUT_ROOT}/${SAFE_MODEL}/hcl_substage2/final_model"

if [[ -z "${DEP_JOBID}" ]]; then
  # No dependency -> adapters must already be on disk.
  for p in "${CPT_ADAPTER}" "${HCL_ADAPTER}"; do
    if [[ ! -d "$p" ]]; then
      echo "[downstream] FATAL: adapter missing: $p" >&2
      echo "             (pass --dep-on <JOBID> to queue jobs that wait on a producer job)" >&2
      exit 64
    fi
  done
else
  echo "[downstream] dependency mode: all jobs will wait on afterok:${DEP_JOBID}"
fi

# Data layout (sft_align / sft_align_challenge — eval on the gold set).
DATA_ROOT=/N/project/AdversarialModeling/datasets/agent_skills/misalignment
SFT_TRAIN_DIR="${DATA_ROOT}/sft/sft_align"
SFT_EVAL_DIR="${DATA_ROOT}/sft/sft_align_challenge"
SFT_EVAL_FILE="${SFT_EVAL_DIR}/align_challenge_test.parquet"
EXEMPLAR_FILE="${SFT_TRAIN_DIR}/align_train.parquet"

# Auto-detect cluster -> partition + QOS.
HOST="$(hostname -s 2>/dev/null || hostname)"
case "${HOST}" in
  h[0-9]*|quartz*|*.quartz.*)   PARTITION="hopper" ; QOS="hopper" ; CLUSTER="Quartz"    ;;
  bigred*|*.bigred*)            PARTITION="gpu"    ; QOS=""       ; CLUSTER="BigRed200" ;;
  *)
    if sinfo -p hopper >/dev/null 2>&1 && sinfo -p hopper -h 2>/dev/null | grep -q .; then
      PARTITION="hopper" ; QOS="hopper" ; CLUSTER="Quartz(detected)"
    else
      PARTITION="gpu" ; QOS="" ; CLUSTER="unknown(fallback gpu)"
    fi
    ;;
esac
QOS_FLAG=()
[[ -n "${QOS}" ]] && QOS_FLAG=(--qos="${QOS}")

DEP_FLAG=()
[[ -n "${DEP_JOBID}" ]] && DEP_FLAG=(--dependency="afterok:${DEP_JOBID}")

INFER_CFG=stages/downstream/configs/inference_misalignment_challenge.yaml
SFT_CFG=stages/downstream/configs/sft_misalignment_example.yaml

echo "[downstream] cluster=${CLUSTER}  partition=${PARTITION}  qos=${QOS:-default}"
echo "[downstream] model=${MODEL}"
echo "[downstream] CPT adapter=${CPT_ADAPTER}"
echo "[downstream] HCL adapter=${HCL_ADAPTER}"
echo "[downstream] eval file  =${SFT_EVAL_FILE}"
echo "[downstream] exemplar   =${EXEMPLAR_FILE}  (n_shot=${N_SHOT})"

submit_infer() {
  local jobname="$1"; shift
  local run_name="$1"; shift
  local extra=("$@")
  sbatch --parsable \
      "${DEP_FLAG[@]}" \
      -p "${PARTITION}" "${QOS_FLAG[@]}" \
      -J "${jobname}_${MODEL}" \
      "${INFER_LAUNCHER}" "${INFER_CFG}" \
          "model.name=${MODEL}" \
          "run.output_root=${OUT_ROOT}" \
          "run.run_name=${run_name}" \
          "data.test_file=${SFT_EVAL_FILE}" \
          "data.task=misalignment_detection" \
          "${extra[@]}"
}

# ----- A: zero-shot, raw backbone -----
JOB_A="$(submit_infer "zs_raw_misalign"        "zs_misalign_raw" \
            "inference.n_shot=0")"
echo "[downstream] A (zero-shot raw)            : ${JOB_A}"

# ----- B: zero-shot, CPT-merged + HCL attached -----
JOB_B="$(submit_infer "zs_cptHcl_misalign"     "zs_misalign_cpt_hcl" \
            "inference.n_shot=0" \
            "model.phase1_adapter_path=${CPT_ADAPTER}" \
            "model.adapter_path=${HCL_ADAPTER}")"
echo "[downstream] B (zero-shot CPT+HCL)        : ${JOB_B}"

# ----- C: few-shot, raw backbone -----
JOB_C="$(submit_infer "fs${N_SHOT}_raw_misalign" "fs${N_SHOT}_misalign_raw" \
            "inference.n_shot=${N_SHOT}" \
            "data.exemplar_file=${EXEMPLAR_FILE}")"
echo "[downstream] C (few-shot k=${N_SHOT} raw)        : ${JOB_C}"

# ----- D: few-shot, CPT-merged + HCL attached -----
JOB_D="$(submit_infer "fs${N_SHOT}_cptHcl_misalign" "fs${N_SHOT}_misalign_cpt_hcl" \
            "inference.n_shot=${N_SHOT}" \
            "data.exemplar_file=${EXEMPLAR_FILE}" \
            "model.phase1_adapter_path=${CPT_ADAPTER}" \
            "model.adapter_path=${HCL_ADAPTER}")"
echo "[downstream] D (few-shot k=${N_SHOT} CPT+HCL)    : ${JOB_D}"

# ----- E: SFT training, chained on CPT+HCL -----
SFT_RUN_NAME=sft_align_real
SFT_OUT_DIR="${OUT_ROOT}/${SAFE_MODEL}/sft_misalignment_detection__${SFT_RUN_NAME}"
SFT_ADAPTER="${SFT_OUT_DIR}/final_model"
JOB_E="$(sbatch --parsable \
    "${DEP_FLAG[@]}" \
    -p "${PARTITION}" "${QOS_FLAG[@]}" \
    -J "sft_align_${MODEL}" \
    "${SFT_LAUNCHER}" "${SFT_CFG}" \
        "model.name=${MODEL}" \
        "model.phase1_adapter_path=${CPT_ADAPTER}" \
        "model.phase2_adapter_path=${HCL_ADAPTER}" \
        "run.output_root=${OUT_ROOT}" \
        "run.run_name=${SFT_RUN_NAME}" \
        "data.train_file=${SFT_TRAIN_DIR}/align_train.parquet" \
        "data.validation_file=${SFT_TRAIN_DIR}/align_val.parquet" \
        "data.test_file=${SFT_TRAIN_DIR}/align_test.parquet")"
echo "[downstream] E (SFT training, real)       : ${JOB_E}"

# ----- F: SFT inference (afterok:E) -----
JOB_F="$(sbatch --parsable \
    --dependency="afterok:${JOB_E}" \
    -p "${PARTITION}" "${QOS_FLAG[@]}" \
    -J "sft_infer_misalign_${MODEL}" \
    "${INFER_LAUNCHER}" "${INFER_CFG}" \
        "model.name=${MODEL}" \
        "model.phase1_adapter_path=${CPT_ADAPTER}" \
        "model.phase2_adapter_path=${HCL_ADAPTER}" \
        "model.adapter_path=${SFT_ADAPTER}" \
        "run.output_root=${OUT_ROOT}" \
        "run.run_name=sft_infer_misalign" \
        "data.test_file=${SFT_EVAL_FILE}" \
        "data.task=misalignment_detection" \
        "inference.n_shot=0")"
echo "[downstream] F (SFT inference, afterok:E) : ${JOB_F}"

cat <<EOF

Submitted downstream eval for ${MODEL}:
  A  zs raw            ${JOB_A}
  B  zs CPT+HCL        ${JOB_B}
  C  fs${N_SHOT} raw            ${JOB_C}
  D  fs${N_SHOT} CPT+HCL        ${JOB_D}
  E  SFT train         ${JOB_E}
  F  SFT infer         ${JOB_F}  (afterok:${JOB_E})

Monitor:
  squeue -u \$USER
  ls ${OUT_ROOT}/${SAFE_MODEL}/
EOF
