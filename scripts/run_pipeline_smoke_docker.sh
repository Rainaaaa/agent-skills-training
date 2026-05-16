#!/usr/bin/env bash
# End-to-end smoke for the WHOLE pipeline on a single backbone, using the
# docker-based workflow on Jetstream (or anywhere with Docker + nvidia-docker).
# Mirrors scripts/run_pipeline_smoke.sh (BR200/SLURM/conda variant) but runs
# everything via `docker compose run --rm <service>`.
#
# Five steps, tiny smoke caps, ~15-30 min on one A100:
#   1. Stage 1 — CPT pretraining  (full_cpt/full_cpt_v3/stage1)
#   2. Stage 2 — HCL contrastive  (pl_hcl/pl_hcl_v2/stage1), chained on Stage 1 LoRA
#   3. Stage 3 — SFT misalignment (sft/sft_align/align_*), chained on Stage 1 LoRA
#   4. Stage 3 — Inference on the sft_align test split, using the Stage 3 adapter
#   5. Stage 3 — eval_baseline    (CLM perplexity on full_cpt_v3/stage1/test)
#
# Per-step failures do NOT abort the script — the summary at the end shows
# which steps passed/failed. Steps after CPT auto-SKIP if the CPT adapter is
# missing (so a CPT failure short-circuits the chain cleanly).
#
# Usage:
#   ./scripts/run_pipeline_smoke_docker.sh [model_name]
# Default model = Foundation-Sec-8B-Reasoning.
# To run on another model:  ./scripts/run_pipeline_smoke_docker.sh llama3.1-8b
#
# Required host-side env (with sensible Jetstream defaults if unset):
#   AGENTSKILLS_DATA_HOST    = /media/volume/skills/misalignment  (compose default)
#   AGENTSKILLS_OUTPUTS_HOST = ./outputs                           (compose default)
#   HF_TOKEN                 = required for gated models (Llama, Gemma, ...)

set -uo pipefail

MODEL="${1:-Foundation-Sec-8B-Reasoning}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Inside-container data dirs (must match the bind-mount in docker-compose.yml).
CPT_DIR=/data/full_cpt/full_cpt_v3/stage1
HCL_DIR=/data/pl_hcl/pl_hcl_v2/stage1
SFT_ALIGN_DIR=/data/sft/sft_align

# Host-side output root — must match the AGENTSKILLS_OUTPUTS_HOST mount.
OUT_HOST="${AGENTSKILLS_OUTPUTS_HOST:-${REPO_ROOT}/outputs}"
OUT_RUNS="${OUT_HOST}/runs"
mkdir -p "${OUT_RUNS}"

# Run-name suffixes — bump if the smoke recipe changes so existing markers
# don't silently skip the new exercise.
RUN_BASE=pipe_smoke_docker
CPT_RUN="${RUN_BASE}_cpt"
HCL_RUN="${RUN_BASE}_hcl"
SFT_RUN="${RUN_BASE}_sft_align"
INF_RUN="${RUN_BASE}_infer_align"
BASE_RUN="${RUN_BASE}_eval_baseline"

# common/registry.py:sanitize_for_path replaces '/' with '__' and ' ' with '_'.
SAFE_MODEL="$(printf '%s' "${MODEL}" | sed 's|/|__|g; s| |_|g')"

# Adapter paths (inside container; OUT_RUNS is mounted at /app/outputs/runs).
CPT_ADAPTER="/app/outputs/runs/${SAFE_MODEL}/${CPT_RUN}/final_model"
SFT_ALIGN_ADAPTER="/app/outputs/runs/${SAFE_MODEL}/sft_misalignment_detection__${SFT_RUN}/final_model"

# Host-side marker paths (for the already-done short-circuit + per-step OK
# check).
host_marker() { echo "${OUT_RUNS}/${SAFE_MODEL}/$1"; }

declare -A RESULTS
run_step() {
  local name="$1"; shift
  local marker="$1"; shift
  echo ""
  echo "============================================================"
  echo "  [pipe] STEP: $name"
  echo "============================================================"
  if [[ -e "$marker" ]]; then
    echo "  [skip] marker already exists: $marker"
    RESULTS[$name]="SKIP"
    return 0
  fi
  if "$@"; then
    if [[ -e "$marker" ]]; then
      RESULTS[$name]="PASS"
    else
      RESULTS[$name]="PASS(no-marker:$marker)"
    fi
  else
    RESULTS[$name]="FAIL(exit=$?)"
  fi
}

echo "[pipe] model=${MODEL}"
echo "[pipe] data host=${AGENTSKILLS_DATA_HOST:-<compose-default>}"
echo "[pipe] outputs host=${OUT_HOST}"
command -v nvidia-smi >/dev/null && nvidia-smi -L || true

# ----- 1) Stage 1 CPT -----
run_step "1_cpt" "$(host_marker "${CPT_RUN}/final_model")" \
  docker compose run --rm pretraining \
    stages/pretraining/train.py \
    --config stages/pretraining/configs/example.yaml \
    model.name="${MODEL}" \
    run.output_root=/app/outputs/runs \
    run.run_name="${CPT_RUN}" \
    "data.train_file=${CPT_DIR}/train.parquet" \
    "data.validation_file=${CPT_DIR}/val.parquet" \
    "data.test_file=${CPT_DIR}/test.parquet" \
    training.max_steps=4 \
    data.max_train_rows=64 data.max_validation_rows=8 data.max_test_rows=8 \
    data.max_seq_length=1024 \
    model.attn_implementation=sdpa

# ----- 2) Stage 2 HCL (chained on CPT) -----
if [[ -d "$(host_marker "${CPT_RUN}/final_model")" ]]; then
  run_step "2_hcl" "$(host_marker "${HCL_RUN}/final_model")" \
    docker compose run --rm hcl \
      stages/hcl/train.py \
      --config stages/hcl/configs/example.yaml \
      model.name="${MODEL}" \
      run.output_root=/app/outputs/runs \
      run.run_name="${HCL_RUN}" \
      "model.phase1_adapter_path=${CPT_ADAPTER}" \
      "data.train_file=${HCL_DIR}/pairs_train.parquet" \
      "data.validation_file=${HCL_DIR}/pairs_val.parquet" \
      "data.test_file=${HCL_DIR}/pairs_test.parquet" \
      training.max_steps=20 \
      data.max_train_rows=256 data.max_validation_rows=64 data.max_test_rows=99 \
      data.stratify_by=pair_kind \
      data.max_seq_length=512 \
      model.attn_implementation=sdpa \
      model.hcl.ce_loss_weight=0.5
else
  echo "[pipe] SKIP step 2 — Stage 1 adapter missing"
  RESULTS[2_hcl]="SKIP(no-cpt-adapter)"
fi

# ----- 3) Stage 3 SFT misalignment (chained on CPT) -----
if [[ -d "$(host_marker "${CPT_RUN}/final_model")" ]]; then
  run_step "3_sft_align" "$(host_marker "sft_misalignment_detection__${SFT_RUN}/final_model")" \
    docker compose run --rm sft-align \
      stages/downstream/sft/train.py \
      --config stages/downstream/configs/sft_misalignment_example.yaml \
      model.name="${MODEL}" \
      run.output_root=/app/outputs/runs \
      run.run_name="${SFT_RUN}" \
      "model.phase1_adapter_path=${CPT_ADAPTER}" \
      "data.train_file=${SFT_ALIGN_DIR}/align_train.parquet" \
      "data.validation_file=${SFT_ALIGN_DIR}/align_val.parquet" \
      "data.test_file=${SFT_ALIGN_DIR}/align_test.parquet" \
      training.max_steps=4 \
      data.max_seq_length=512
else
  echo "[pipe] SKIP step 3 — Stage 1 adapter missing"
  RESULTS[3_sft_align]="SKIP(no-cpt-adapter)"
fi

# ----- 4) Inference using the SFT-misalignment adapter -----
if [[ -d "$(host_marker "sft_misalignment_detection__${SFT_RUN}/final_model")" ]]; then
  INF_MARKER="$(host_marker "infer_misalignment_detection__${INF_RUN}/metrics_misalignment_detection.json")"
  run_step "4_inference" "${INF_MARKER}" \
    docker compose run --rm inference \
      stages/downstream/inference.py \
      --config stages/downstream/configs/inference_example.yaml \
      model.name="${MODEL}" \
      run.output_root=/app/outputs/runs \
      run.run_name="${INF_RUN}" \
      "model.adapter_path=${SFT_ALIGN_ADAPTER}" \
      data.task=misalignment_detection \
      "data.test_file=${SFT_ALIGN_DIR}/align_test.parquet" \
      data.max_rows=8
else
  echo "[pipe] SKIP step 4 — SFT-align adapter missing"
  RESULTS[4_inference]="SKIP(no-sft-adapter)"
fi

# ----- 5) eval_baseline — CLM perplexity on the CPT test split -----
BASE_MARKER="$(host_marker "${BASE_RUN}/baseline_metrics.json")"
run_step "5_eval_baseline" "${BASE_MARKER}" \
  docker compose run --rm baseline \
    stages/downstream/eval_baseline.py \
    --config stages/downstream/configs/eval_baseline_example.yaml \
    model.name="${MODEL}" \
    run.output_root=/app/outputs/runs \
    run.run_name="${BASE_RUN}" \
    "data.test_file=${CPT_DIR}/test.parquet" \
    data.max_test_rows=8 \
    data.max_seq_length=512

# ----- summary -----
echo ""
echo "============================================================"
echo "  [pipe] SUMMARY  model=${MODEL}"
echo "============================================================"
fail_count=0
for step in 1_cpt 2_hcl 3_sft_align 4_inference 5_eval_baseline; do
  status="${RESULTS[$step]:-NOT_RUN}"
  printf "  %-18s %s\n" "$step" "$status"
  [[ "$status" == PASS* || "$status" == SKIP* ]] || fail_count=$((fail_count+1))
done
echo "  Output root: ${OUT_RUNS}/${SAFE_MODEL}/"
echo "============================================================"

exit "$fail_count"
