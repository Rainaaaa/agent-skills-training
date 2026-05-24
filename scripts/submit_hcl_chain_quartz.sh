#!/bin/bash
# Submit a continual two-sub-stage HCL chain (stage1 pairs @ 4K, then
# stage2 pairs @ 10K) for a single backbone on Quartz hopper / BR200 gpu.
#
# Sub-stage 1 chains on the CPT sub-stage 2 final_model adapter
# (model.phase1_adapter_path). Sub-stage 2 uses --dependency=afterok on
# sub-stage 1 and resumes from its latest checkpoint via RESUME_PARENT_DIR
# (run_hcl_quartz.sh resolves the dir to the highest-numbered checkpoint-*).
#
# Resource shape mirrors the CPT chain decision:
#   - Sub-stage 1 (seq=4096) on 1xH100, grad_accum=8  -> effective_batch 8
#   - Sub-stage 2 (seq=10240) on 2xH100, grad_accum=4 -> effective_batch 8
# per_device_train_batch_size is pinned to 1 on both sides for memory headroom
# (HCL forwards the encoder TWICE per step — once for anchor, once for pair).
#
# LR-schedule note (per KELLY_IT_HANDOFF §5):
#   `cosine` decays to ~0 by end of sub-stage 1; sub-stage 2 here overrides
#   `training.lr_scheduler_type=constant` so the resumed scheduler doesn't
#   pin LR at zero.
#
# Usage:
#   bash scripts/submit_hcl_chain_quartz.sh [model_name] [config_path] [cpt_adapter_path]
#
# Defaults:
#   model_name        : llama3.1-8b
#   config_path       : stages/hcl/configs/lp_hcl_v2_quarter.yaml
#   cpt_adapter_path  : outputs/runs/<safe_model>/cpt_substage2/final_model

set -euo pipefail

MODEL="${1:-llama3.1-8b}"
CONFIG="${2:-stages/hcl/configs/lp_hcl_v2_quarter.yaml}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LAUNCHER="${SCRIPT_DIR}/run_hcl_quartz.sh"

cd "${REPO_ROOT}"

SAFE_MODEL="$(printf '%s' "${MODEL}" | sed 's|/|__|g; s| |_|g')"
OUT_ROOT="${REPO_ROOT}/outputs/runs"
CPT_ADAPTER="${3:-${OUT_ROOT}/${SAFE_MODEL}/cpt_substage2/final_model}"

if [[ ! -d "${CPT_ADAPTER}" ]]; then
  echo "[chain] FATAL: CPT sub-stage 2 adapter not found at ${CPT_ADAPTER}" >&2
  echo "        Pass the path explicitly as the 3rd arg, or run the CPT chain first." >&2
  exit 64
fi

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

DATA_ROOT=/N/project/AdversarialModeling/datasets/agent_skills/misalignment
S1_DIR="${DATA_ROOT}/pl_hcl/pl_hcl_v2/stage1"
S2_DIR="${DATA_ROOT}/pl_hcl/pl_hcl_v2/stage2"

S1_RUN_NAME=hcl_substage1
S2_RUN_NAME=hcl_substage2
S1_OUT_DIR="${OUT_ROOT}/${SAFE_MODEL}/${S1_RUN_NAME}"

echo "[chain] cluster=${CLUSTER}  partition=${PARTITION}  qos=${QOS:-default}"
echo "[chain] model=${MODEL}      config=${CONFIG}"
echo "[chain] cpt_adapter=${CPT_ADAPTER}"
echo "[chain] sub-stage 1 out=${S1_OUT_DIR}"

# ----- sub-stage 1 (1 GPU, seq=4096, chained on CPT sub-stage 2) -----
# Skip-on-rerun: if sub-stage 1's final_model is already on disk, treat it
# as DONE and submit only sub-stage 2 (with no Slurm dependency).
DEP_FLAG=()
if [[ -d "${S1_OUT_DIR}/final_model" ]]; then
  echo "[chain] sub-stage 1 already complete (found ${S1_OUT_DIR}/final_model) — skipping submission"
  SUB1_JOBID="(done)"
else
  SUB1_JOBID="$(sbatch --parsable \
      -p "${PARTITION}" "${QOS_FLAG[@]}" \
      --gres=gpu:1 --mem=96G --cpus-per-task=16 \
      -J "hcl_substage1_${MODEL}" \
      "${LAUNCHER}" "${CONFIG}" \
          "model.name=${MODEL}" \
          "model.phase1_adapter_path=${CPT_ADAPTER}" \
          "run.output_root=${OUT_ROOT}" \
          "run.run_name=${S1_RUN_NAME}" \
          "data.train_file=${S1_DIR}/pairs_train.parquet" \
          "data.validation_file=${S1_DIR}/pairs_val.parquet" \
          "data.test_file=${S1_DIR}/pairs_test.parquet" \
          "data.max_seq_length=4096" \
          "training.per_device_train_batch_size=1" \
          "training.gradient_accumulation_steps=8")"
  DEP_FLAG=(--dependency="afterok:${SUB1_JOBID}")
  echo "[chain] submitted sub-stage 1: jobid=${SUB1_JOBID}"
fi

# ----- sub-stage 2 (2 GPUs, seq=10240, resumes from sub-stage 1) -----
# IMPORTANT: phase1_adapter_path MUST be passed again here.
# HclTrainer._save writes only trainable params (Phase 2 LoRA + hcl_head.pt)
# — NOT the merged base. On resume the model is reconstructed via:
#   base -> load+merge phase1 adapter -> attach fresh Phase 2 LoRA
# and only then the checkpoint's pytorch_model.bin (trainable-only) is
# loaded. Skip phase1_adapter_path and the YAML's
# ${AGENTSKILLS_PHASE1_ADAPTER:-./outputs/phase1/final_model} default
# kicks in -> FileNotFoundError at model build.
SUB2_JOBID="$(sbatch --parsable \
    "${DEP_FLAG[@]}" \
    --export="ALL,RESUME_PARENT_DIR=${S1_OUT_DIR}" \
    -p "${PARTITION}" "${QOS_FLAG[@]}" \
    --gres=gpu:2 --mem=192G --cpus-per-task=32 \
    -J "hcl_substage2_${MODEL}" \
    "${LAUNCHER}" "${CONFIG}" \
        "model.name=${MODEL}" \
        "model.phase1_adapter_path=${CPT_ADAPTER}" \
        "run.output_root=${OUT_ROOT}" \
        "run.run_name=${S2_RUN_NAME}" \
        "data.train_file=${S2_DIR}/pairs_train.parquet" \
        "data.validation_file=${S2_DIR}/pairs_val.parquet" \
        "data.test_file=${S2_DIR}/pairs_test.parquet" \
        "data.max_seq_length=10240" \
        "training.per_device_train_batch_size=1" \
        "training.gradient_accumulation_steps=4" \
        "training.lr_scheduler_type=constant")"
echo "[chain] submitted sub-stage 2: jobid=${SUB2_JOBID}  ${DEP_FLAG[*]:-(no dependency)}"

cat <<EOF

Submitted HCL chain:
  Model            : ${MODEL}
  Config           : ${CONFIG}
  Phase 1 adapter  : ${CPT_ADAPTER}
  Partition        : ${PARTITION}  qos=${QOS:-default}

  Sub-stage 1 : ${SUB1_JOBID}   1xH100  stage1 pairs, max_seq_length=4096
                  chains on CPT sub-stage 2 (phase1_adapter_path)
                  per_device=1, grad_accum=8, effective_batch=8
  Sub-stage 2 : ${SUB2_JOBID}   2xH100  stage2 pairs, max_seq_length=10240
                  per_device=1, grad_accum=4, effective_batch=8, lr_scheduler=constant
                  Resumes from sub-stage 1's latest checkpoint.
                  Starts only if sub-stage 1 exits 0.

Monitor:
  squeue -u \$USER
  tail -f ${REPO_ROOT}/log/hcl_substage1_${MODEL}_${SUB1_JOBID}.log

Outputs:
  ${S1_OUT_DIR}/   (sub-stage 1)
  ${OUT_ROOT}/${SAFE_MODEL}/${S2_RUN_NAME}/   (sub-stage 2)
EOF
