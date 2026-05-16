#!/bin/bash
# Submit a continual two-sub-stage CPT chain (stage1 @ 4K, then stage2 @ 10K)
# for a single backbone on Quartz hopper (or BR200 gpu).
#
# Sub-stage 2 uses --dependency=afterok on sub-stage 1 and chains via
# RESUME_PARENT_DIR (run_cpt_quartz.sh picks the latest checkpoint-* dir
# inside that path at job start).
#
# LR-schedule note (per KELLY_IT_HANDOFF §5):
#   The default `cosine` LR decays to ~0 by the end of sub-stage 1.
#   Sub-stage 2 here overrides `training.lr_scheduler_type=constant` so the
#   inherited optimizer/scheduler state doesn't pin LR to zero.
#
# Usage:
#   bash scripts/submit_cpt_chain_quartz.sh [model_name] [config_path]
#
# Defaults: llama3.1-8b, stages/pretraining/configs/full_cpt_v3_quarter.yaml
set -euo pipefail

MODEL="${1:-llama3.1-8b}"
CONFIG="${2:-stages/pretraining/configs/full_cpt_v3_quarter.yaml}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LAUNCHER="${SCRIPT_DIR}/run_cpt_quartz.sh"

cd "${REPO_ROOT}"

# Auto-detect cluster -> partition + QOS (same logic as submit_smoke_all.sh).
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

# Data layout (challenge-scrubbed v3, two-substage continual).
DATA_ROOT=/N/project/AdversarialModeling/datasets/agent_skills/misalignment
S1_DIR="${DATA_ROOT}/full_cpt/full_cpt_v3/stage1"
S2_DIR="${DATA_ROOT}/full_cpt/full_cpt_v3/stage2"

# Output dir for sub-stage 1 — passed to sub-stage 2 via RESUME_PARENT_DIR.
# Path must match what stages/pretraining/train.py derives:
#   <output_root>/<sanitize_for_path(model.name)>/<run.run_name>
SAFE_MODEL="$(printf '%s' "${MODEL}" | sed 's|/|__|g; s| |_|g')"
OUT_ROOT="${REPO_ROOT}/outputs/runs"
S1_RUN_NAME=cpt_substage1
S2_RUN_NAME=cpt_substage2
S1_OUT_DIR="${OUT_ROOT}/${SAFE_MODEL}/${S1_RUN_NAME}"

echo "[chain] cluster=${CLUSTER}  partition=${PARTITION}  qos=${QOS:-default}"
echo "[chain] model=${MODEL}      config=${CONFIG}"
echo "[chain] sub-stage 1 out=${S1_OUT_DIR}"

# ----- sub-stage 1 -----
#
# Effective-batch policy (keep effective_batch=8 across resource shapes):
#   per_device_train_batch_size=1 × grad_accum × world_size == 8
#
# Sub-stage 1 (seq=4096, ~492K train rows): 1 GPU is enough — grad_accum=8.
# Sub-stage 2 (seq=10240, all-files data):  2 GPUs to roughly halve wall
#                                           time on the longer epoch — grad_accum=4.
# Both keep effective batch 8, so no LR rescale is needed when chaining.
SUB1_JOBID="$(sbatch --parsable \
    -p "${PARTITION}" "${QOS_FLAG[@]}" \
    --gres=gpu:1 --mem=96G --cpus-per-task=16 \
    -J "cpt_substage1_${MODEL}" \
    "${LAUNCHER}" "${CONFIG}" \
        "model.name=${MODEL}" \
        "run.output_root=${OUT_ROOT}" \
        "run.run_name=${S1_RUN_NAME}" \
        "data.train_file=${S1_DIR}/train.parquet" \
        "data.validation_file=${S1_DIR}/val.parquet" \
        "data.test_file=${S1_DIR}/test.parquet" \
        "data.max_seq_length=4096" \
        "training.gradient_accumulation_steps=8")"
echo "[chain] submitted sub-stage 1: jobid=${SUB1_JOBID}"

# ----- sub-stage 2 (afterok dependency on sub-stage 1) -----
# Override lr_scheduler_type=constant so the resumed scheduler doesn't keep
# LR pinned at ~0 (cosine decays to 0 by end of epoch 1 in sub-stage 1).
SUB2_JOBID="$(sbatch --parsable \
    --dependency="afterok:${SUB1_JOBID}" \
    --export="ALL,RESUME_PARENT_DIR=${S1_OUT_DIR}" \
    -p "${PARTITION}" "${QOS_FLAG[@]}" \
    --gres=gpu:2 --mem=192G --cpus-per-task=32 \
    -J "cpt_substage2_${MODEL}" \
    "${LAUNCHER}" "${CONFIG}" \
        "model.name=${MODEL}" \
        "run.output_root=${OUT_ROOT}" \
        "run.run_name=${S2_RUN_NAME}" \
        "data.train_file=${S2_DIR}/train.parquet" \
        "data.validation_file=${S2_DIR}/val.parquet" \
        "data.test_file=${S2_DIR}/test.parquet" \
        "data.max_seq_length=10240" \
        "training.gradient_accumulation_steps=4" \
        "training.lr_scheduler_type=constant")"
echo "[chain] submitted sub-stage 2: jobid=${SUB2_JOBID}  (afterok:${SUB1_JOBID})"

cat <<EOF

Submitted CPT chain:
  Model       : ${MODEL}
  Config      : ${CONFIG}
  Partition   : ${PARTITION}  qos=${QOS:-default}

  Sub-stage 1 : ${SUB1_JOBID}   1xH100  stage1 data, max_seq_length=4096
                  grad_accum=8, effective_batch=8
  Sub-stage 2 : ${SUB2_JOBID}   2xH100  stage2 data, max_seq_length=10240
                  grad_accum=4, effective_batch=8, lr_scheduler=constant
                  Resumes from sub-stage 1's latest checkpoint.
                  Starts only if sub-stage 1 exits 0.

Monitor:
  squeue -u \$USER
  tail -f ${REPO_ROOT}/log/cpt_substage1_${MODEL}_${SUB1_JOBID}.log

Outputs:
  ${S1_OUT_DIR}/   (sub-stage 1)
  ${OUT_ROOT}/${SAFE_MODEL}/${S2_RUN_NAME}/   (sub-stage 2)
EOF
