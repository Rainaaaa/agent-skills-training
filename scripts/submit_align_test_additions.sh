#!/bin/bash
# Submit the 5 align_test inference variants for ONE backbone:
#   A2  zs raw                                        align_test, cap=5000
#   B2  zs CPT-merged + HCL attached                  align_test, cap=5000
#   C2  fs5 raw                                       align_test, cap=5000
#   D2  fs5 CPT-merged + HCL attached                 align_test, cap=5000
#   F2  SFT inference (afterok:SFT-train job)         align_test, cap=5000
#
# Uses run_infer_quartz.sh (1xH100, 2h). Wall time on 5000 rows:
#   zs  ≈ 25-30 min
#   fs5 ≈ ~3.5h per run
# Cap is set with data.max_rows=5000 + a fresh dataset shuffle (the inference
# loader doesn't shuffle by default — but max_rows is applied AFTER load, so
# we get the first 5000 rows of the parquet, which is fine for these IID
# scanner-labeled splits).
#
# Usage:
#   bash scripts/submit_align_test_additions.sh <MODEL> [--pretrain-dep JOBID] [--sft-train-jobid JOBID]
#
#   --pretrain-dep   : if set, A2-D2 wait on afterok:<JOBID> (the HCL substage-2
#                      job for backbones whose adapters aren't on disk yet).
#   --sft-train-jobid: required for F2 to be queued (afterok). Without it,
#                      F2 is skipped (use when SFT training adapter already exists).

set -euo pipefail

MODEL="${1:?Usage: $0 <model> [--pretrain-dep JOBID] [--sft-train-jobid JOBID]}"
shift

PRETRAIN_DEP=""
SFT_TRAIN_JOB=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pretrain-dep)    PRETRAIN_DEP="$2"; shift 2 ;;
    --sft-train-jobid) SFT_TRAIN_JOB="$2"; shift 2 ;;
    *) echo "[align-test] unknown arg: $1" >&2; exit 64 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INFER_LAUNCHER="${SCRIPT_DIR}/run_infer_quartz.sh"
INFER_CFG=stages/downstream/configs/inference_misalignment_challenge.yaml

cd "${REPO_ROOT}"

SAFE_MODEL="$(printf '%s' "${MODEL}" | sed 's|/|__|g; s| |_|g')"
OUT_ROOT="${REPO_ROOT}/outputs/runs"
CPT_ADAPTER="${OUT_ROOT}/${SAFE_MODEL}/cpt_substage2/final_model"
HCL_ADAPTER="${OUT_ROOT}/${SAFE_MODEL}/hcl_substage2/final_model"
SFT_ADAPTER="${OUT_ROOT}/${SAFE_MODEL}/sft_misalignment_detection__sft_align_real/final_model"

# Align_test = the broad scanner-labeled test split (26K rows), capped to 5000.
DATA_ROOT=/N/project/AdversarialModeling/datasets/agent_skills/misalignment
ALIGN_TEST_FILE="${DATA_ROOT}/sft/sft_align/align_test.parquet"
EXEMPLAR_FILE="${DATA_ROOT}/sft/sft_align/align_train.parquet"
MAX_ROWS=5000

# Cluster auto-detect.
HOST="$(hostname -s 2>/dev/null || hostname)"
case "${HOST}" in
  h[0-9]*|quartz*|*.quartz.*)   PARTITION="hopper" ; QOS="hopper" ;;
  bigred*|*.bigred*)            PARTITION="gpu"    ; QOS=""       ;;
  *)
    if sinfo -p hopper >/dev/null 2>&1; then PARTITION="hopper" ; QOS="hopper"
    else PARTITION="gpu" ; QOS=""; fi
    ;;
esac
QOS_FLAG=()
[[ -n "${QOS}" ]] && QOS_FLAG=(--qos="${QOS}")

PRETRAIN_DEP_FLAG=()
[[ -n "${PRETRAIN_DEP}" ]] && PRETRAIN_DEP_FLAG=(--dependency="afterok:${PRETRAIN_DEP}")

# Bigger walltime for fs5 on 5000 rows (~3.5h).
WALL_FS="--time=05:00:00"

echo "[align-test] model=${MODEL}  partition=${PARTITION}  qos=${QOS:-default}"
echo "[align-test] eval file=${ALIGN_TEST_FILE}  max_rows=${MAX_ROWS}"
echo "[align-test] pretrain dep=${PRETRAIN_DEP:-none}  sft train job=${SFT_TRAIN_JOB:-none}"

submit_infer() {
  local jobname="$1"; shift
  local run_name="$1"; shift
  local extra_walltime="$1"; shift
  local extra=("$@")
  sbatch --parsable \
      "${PRETRAIN_DEP_FLAG[@]}" \
      ${extra_walltime} \
      -p "${PARTITION}" "${QOS_FLAG[@]}" \
      -J "${jobname}_${MODEL}" \
      "${INFER_LAUNCHER}" "${INFER_CFG}" \
          "model.name=${MODEL}" \
          "run.output_root=${OUT_ROOT}" \
          "run.run_name=${run_name}" \
          "data.test_file=${ALIGN_TEST_FILE}" \
          "data.task=misalignment_detection" \
          "data.max_rows=${MAX_ROWS}" \
          "${extra[@]}"
}

JOB_A2="$(submit_infer "zs_raw_aligntest"    "zs_misalign_raw_aligntest" "" \
            "inference.n_shot=0")"
echo "[align-test] A2 (zs raw on align_test 5k)        : ${JOB_A2}"

JOB_B2="$(submit_infer "zs_cptHcl_aligntest" "zs_misalign_cpt_hcl_aligntest" "" \
            "inference.n_shot=0" \
            "model.phase1_adapter_path=${CPT_ADAPTER}" \
            "model.adapter_path=${HCL_ADAPTER}")"
echo "[align-test] B2 (zs CPT+HCL on align_test 5k)    : ${JOB_B2}"

JOB_C2="$(submit_infer "fs5_raw_aligntest"   "fs5_misalign_raw_aligntest" "${WALL_FS}" \
            "inference.n_shot=5" \
            "data.exemplar_file=${EXEMPLAR_FILE}")"
echo "[align-test] C2 (fs5 raw on align_test 5k)       : ${JOB_C2}"

JOB_D2="$(submit_infer "fs5_cptHcl_aligntest" "fs5_misalign_cpt_hcl_aligntest" "${WALL_FS}" \
            "inference.n_shot=5" \
            "data.exemplar_file=${EXEMPLAR_FILE}" \
            "model.phase1_adapter_path=${CPT_ADAPTER}" \
            "model.adapter_path=${HCL_ADAPTER}")"
echo "[align-test] D2 (fs5 CPT+HCL on align_test 5k)   : ${JOB_D2}"

# F2 only if we know the SFT-train job ID (otherwise the SFT adapter may not
# exist yet and we can't depend on a job ID we don't have).
JOB_F2=""
if [[ -n "${SFT_TRAIN_JOB}" ]]; then
  JOB_F2="$(sbatch --parsable \
      --dependency="afterok:${SFT_TRAIN_JOB}" \
      -p "${PARTITION}" "${QOS_FLAG[@]}" \
      -J "sft_infer_aligntest_${MODEL}" \
      "${INFER_LAUNCHER}" "${INFER_CFG}" \
          "model.name=${MODEL}" \
          "model.phase1_adapter_path=${CPT_ADAPTER}" \
          "model.phase2_adapter_path=${HCL_ADAPTER}" \
          "model.adapter_path=${SFT_ADAPTER}" \
          "run.output_root=${OUT_ROOT}" \
          "run.run_name=sft_infer_misalign_aligntest" \
          "data.test_file=${ALIGN_TEST_FILE}" \
          "data.task=misalignment_detection" \
          "data.max_rows=${MAX_ROWS}" \
          "inference.n_shot=0")"
  echo "[align-test] F2 (sft inference on align_test 5k) : ${JOB_F2}  (afterok:${SFT_TRAIN_JOB})"
else
  echo "[align-test] F2 SKIPPED (no --sft-train-jobid provided)"
fi
