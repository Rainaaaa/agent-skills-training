#!/bin/bash
# CPT-only ablation: same matrix as the main downstream eval but WITHOUT
# the HCL Phase-2 adapter. Lets us isolate HCL's contribution by comparing
# `cpt_only` results to the existing `cpt_hcl` results.
#
# 7 jobs per backbone:
#   AC  zs_cpt_only                 challenge (1444)
#   BC  zs_cpt_only_aligntest       align_test cap 5000
#   CC  fs5_cpt_only                challenge (1444), k=5
#   DC  fs5_cpt_only_aligntest      align_test cap 5000, k=5
#   EC  sft_on_cpt_only             SFT training, chained ONLY on phase1=CPT (no HCL)
#   FC  sft_infer_cpt_only          challenge (1444), afterok:EC
#   GC  sft_infer_cpt_only_aligntest align_test cap 5000, afterok:EC
#
# Usage:
#   bash scripts/submit_cpt_only_ablation_quartz.sh <MODEL> [n_shot]
# Defaults: n_shot=5
#
# Assumes the CPT sub-stage 2 adapter exists locally:
#   outputs/runs/<safe_model>/cpt_substage2/final_model/

set -euo pipefail

MODEL="${1:?Usage: $0 <model> [n_shot]}"
N_SHOT="${2:-5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INFER_LAUNCHER="${SCRIPT_DIR}/run_infer_quartz.sh"
SFT_LAUNCHER="${SCRIPT_DIR}/run_sft_quartz.sh"
INFER_CFG=stages/downstream/configs/inference_misalignment_challenge.yaml
SFT_CFG=stages/downstream/configs/sft_misalignment_example.yaml

cd "${REPO_ROOT}"

SAFE_MODEL="$(printf '%s' "${MODEL}" | sed 's|/|__|g; s| |_|g')"
OUT_ROOT="${REPO_ROOT}/outputs/runs"
CPT_ADAPTER="${OUT_ROOT}/${SAFE_MODEL}/cpt_substage2/final_model"

if [[ ! -d "${CPT_ADAPTER}" ]]; then
  echo "[cpt-abl] FATAL: CPT adapter missing: ${CPT_ADAPTER}" >&2
  exit 64
fi

# Data layout.
DATA_ROOT=/N/project/AdversarialModeling/datasets/agent_skills/misalignment
SFT_TRAIN_DIR="${DATA_ROOT}/sft/sft_align"
CHALLENGE_FILE="${DATA_ROOT}/sft/sft_align_challenge/align_challenge_test.parquet"
ALIGN_TEST_FILE="${SFT_TRAIN_DIR}/align_test.parquet"
EXEMPLAR_FILE="${SFT_TRAIN_DIR}/align_train.parquet"
MAX_ROWS_ALIGN=5000

# Cluster auto-detect.
HOST="$(hostname -s 2>/dev/null || hostname)"
case "${HOST}" in
  h[0-9]*|quartz*|*.quartz.*)   PARTITION="hopper" ; QOS="hopper" ;;
  bigred*|*.bigred*)            PARTITION="gpu"    ; QOS=""       ;;
  *) PARTITION="hopper" ; QOS="hopper" ;;
esac
QOS_FLAG=()
[[ -n "${QOS}" ]] && QOS_FLAG=(--qos="${QOS}")

WALL_FS="--time=05:00:00"   # fs5 on 5000 rows can take ~3.5h

echo "[cpt-abl] model=${MODEL}  partition=${PARTITION}  qos=${QOS:-default}"
echo "[cpt-abl] CPT adapter=${CPT_ADAPTER}"
echo "[cpt-abl] (no HCL adapter — this is the CPT-only ablation)"

submit_infer() {
  local jobname="$1"; shift
  local run_name="$1"; shift
  local extra_walltime="$1"; shift
  local extra=("$@")
  sbatch --parsable \
      ${extra_walltime} \
      -p "${PARTITION}" "${QOS_FLAG[@]}" \
      -J "${jobname}_${MODEL}" \
      "${INFER_LAUNCHER}" "${INFER_CFG}" \
          "model.name=${MODEL}" \
          "model.phase1_adapter_path=${CPT_ADAPTER}" \
          "run.output_root=${OUT_ROOT}" \
          "run.run_name=${run_name}" \
          "data.task=misalignment_detection" \
          "${extra[@]}"
}

# --- AC: zs CPT-only on challenge ---
JOB_AC="$(submit_infer "zs_cptOnly_chal"     "zs_misalign_cpt_only" "" \
            "inference.n_shot=0" \
            "data.test_file=${CHALLENGE_FILE}")"
echo "[cpt-abl] AC zs_cpt_only / challenge      : ${JOB_AC}"

# --- BC: zs CPT-only on align_test 5000 ---
JOB_BC="$(submit_infer "zs_cptOnly_aligntest" "zs_misalign_cpt_only_aligntest" "" \
            "inference.n_shot=0" \
            "data.test_file=${ALIGN_TEST_FILE}" \
            "data.max_rows=${MAX_ROWS_ALIGN}")"
echo "[cpt-abl] BC zs_cpt_only / align_test 5k  : ${JOB_BC}"

# --- CC: fs5 CPT-only on challenge ---
JOB_CC="$(submit_infer "fs${N_SHOT}_cptOnly_chal" "fs${N_SHOT}_misalign_cpt_only" "${WALL_FS}" \
            "inference.n_shot=${N_SHOT}" \
            "data.exemplar_file=${EXEMPLAR_FILE}" \
            "data.test_file=${CHALLENGE_FILE}")"
echo "[cpt-abl] CC fs${N_SHOT}_cpt_only / challenge : ${JOB_CC}"

# --- DC: fs5 CPT-only on align_test 5000 ---
JOB_DC="$(submit_infer "fs${N_SHOT}_cptOnly_aligntest" "fs${N_SHOT}_misalign_cpt_only_aligntest" "${WALL_FS}" \
            "inference.n_shot=${N_SHOT}" \
            "data.exemplar_file=${EXEMPLAR_FILE}" \
            "data.test_file=${ALIGN_TEST_FILE}" \
            "data.max_rows=${MAX_ROWS_ALIGN}")"
echo "[cpt-abl] DC fs${N_SHOT}_cpt_only / align_test 5k : ${JOB_DC}"

# --- EC: SFT trained chained ONLY on CPT (no phase2_adapter_path) ---
SFT_RUN_NAME=sft_align_cpt_only
SFT_OUT_DIR="${OUT_ROOT}/${SAFE_MODEL}/sft_misalignment_detection__${SFT_RUN_NAME}"
SFT_ADAPTER="${SFT_OUT_DIR}/final_model"
JOB_EC="$(sbatch --parsable \
    -p "${PARTITION}" "${QOS_FLAG[@]}" \
    -J "sft_cptOnly_${MODEL}" \
    "${SFT_LAUNCHER}" "${SFT_CFG}" \
        "model.name=${MODEL}" \
        "model.phase1_adapter_path=${CPT_ADAPTER}" \
        "run.output_root=${OUT_ROOT}" \
        "run.run_name=${SFT_RUN_NAME}" \
        "data.train_file=${SFT_TRAIN_DIR}/align_train.parquet" \
        "data.validation_file=${SFT_TRAIN_DIR}/align_val.parquet" \
        "data.test_file=${SFT_TRAIN_DIR}/align_test.parquet")"
echo "[cpt-abl] EC sft training on CPT-only       : ${JOB_EC}"

# --- FC: SFT-on-CPT inference on challenge (afterok:EC) ---
JOB_FC="$(sbatch --parsable \
    --dependency="afterok:${JOB_EC}" \
    -p "${PARTITION}" "${QOS_FLAG[@]}" \
    -J "sft_infer_cptOnly_chal_${MODEL}" \
    "${INFER_LAUNCHER}" "${INFER_CFG}" \
        "model.name=${MODEL}" \
        "model.phase1_adapter_path=${CPT_ADAPTER}" \
        "model.adapter_path=${SFT_ADAPTER}" \
        "run.output_root=${OUT_ROOT}" \
        "run.run_name=sft_infer_misalign_cpt_only" \
        "data.test_file=${CHALLENGE_FILE}" \
        "data.task=misalignment_detection" \
        "inference.n_shot=0")"
echo "[cpt-abl] FC sft_infer_cpt_only / challenge : ${JOB_FC}  (afterok:${JOB_EC})"

# --- GC: SFT-on-CPT inference on align_test 5000 (afterok:EC) ---
JOB_GC="$(sbatch --parsable \
    --dependency="afterok:${JOB_EC}" \
    -p "${PARTITION}" "${QOS_FLAG[@]}" \
    -J "sft_infer_cptOnly_aligntest_${MODEL}" \
    "${INFER_LAUNCHER}" "${INFER_CFG}" \
        "model.name=${MODEL}" \
        "model.phase1_adapter_path=${CPT_ADAPTER}" \
        "model.adapter_path=${SFT_ADAPTER}" \
        "run.output_root=${OUT_ROOT}" \
        "run.run_name=sft_infer_misalign_cpt_only_aligntest" \
        "data.test_file=${ALIGN_TEST_FILE}" \
        "data.max_rows=${MAX_ROWS_ALIGN}" \
        "data.task=misalignment_detection" \
        "inference.n_shot=0")"
echo "[cpt-abl] GC sft_infer_cpt_only / align_test: ${JOB_GC}  (afterok:${JOB_EC})"
