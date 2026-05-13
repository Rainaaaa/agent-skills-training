#!/usr/bin/env bash
# scripts/run_full_eval.sh — full evaluation matrix across N backbones.
#
# Per model, runs the curated-dataset eval matrix end-to-end:
#
#   1. baseline ppl (intrinsic, pre-CPT)
#   2. zero-shot inference, misalignment_detection, pre-CPT
#   3. zero-shot inference, malicious_detection,    pre-CPT
#   4. CPT sub-stage 1 training (stage1 data, max_seq_length=4096)
#   5. baseline ppl (intrinsic, post-CPT — adapter-aware)
#   6. zero-shot inference, misalignment_detection, post-CPT
#   7. zero-shot inference, malicious_detection,    post-CPT
#   8. SFT misalignment training (chained from CPT)
#   9. inference with SFT-misalignment adapter
#  10. SFT malicious training (chained from CPT)
#  11. inference with SFT-malicious adapter
#
# Outputs land under:
#   $AGENTSKILLS_OUTPUTS_HOST/runs/<model>/<run_name>/
#
# Idempotent: each step checks for its expected output artifact and skips
# if it already exists. Failures don't abort the loop — failing models
# are reported at the end.
#
# Usage:
#   bash scripts/run_full_eval.sh                          # all 6 default models
#   bash scripts/run_full_eval.sh Foundation-Sec-8B-Reasoning Qwen3-8B
#
# Requires:
#   AGENTSKILLS_DATA_HOST    — bind-mount source for /data
#   AGENTSKILLS_OUTPUTS_HOST — bind-mount source for /app/outputs
#   HF_TOKEN                 — for gated HF model downloads (Llama, Gemma, etc.)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODELS=(
  Foundation-Sec-8B-Reasoning
  RedSage-Qwen3-8B-DPO
  WhiteRabbitNeo-2-8B
  llama3.1-8b
  Qwen3-8B
  gemma-4-E4B
)
if [ "$#" -gt 0 ]; then
  MODELS=("$@")
fi

: "${AGENTSKILLS_OUTPUTS_HOST:?Set AGENTSKILLS_OUTPUTS_HOST}"
: "${AGENTSKILLS_DATA_HOST:?Set AGENTSKILLS_DATA_HOST}"
mkdir -p "$AGENTSKILLS_OUTPUTS_HOST/runs"

# Run-name constants — must match `scripts/aggregate_eval.py`.
NAME_BASELINE_PRE="eval_baseline_pre_cpt"
NAME_BASELINE_POST="eval_baseline_post_cpt"
NAME_ZS_ALIGN_PRE="eval_zs_misalignment_pre_cpt"
NAME_ZS_ALIGN_POST="eval_zs_misalignment_post_cpt"
NAME_ZS_MAL_PRE="eval_zs_malicious_pre_cpt"
NAME_ZS_MAL_POST="eval_zs_malicious_post_cpt"
NAME_CPT="cpt_substage1"
NAME_SFT_ALIGN="sft_misalignment"
NAME_SFT_MAL="sft_malicious"
NAME_EVAL_SFT_ALIGN="eval_sft_misalignment"
NAME_EVAL_SFT_MAL="eval_sft_malicious"

# Data paths inside the container.
CPT_DATA_DIR=/data/full_cpt_v2/stage1
ALIGN_TEST=/data/adapted/sft_align/test.parquet
MAL_TEST=/data/adapted/sft_malicious/test.parquet

host_dir() { echo "$AGENTSKILLS_OUTPUTS_HOST/runs/$1/$2"; }
in_dir()   { echo "/app/outputs/runs/$1/$2"; }

already_done() {
  local marker=$1 step=$2
  if [ -e "$marker" ]; then
    echo "    [skip] $step (found $(basename "$marker"))"
    return 0
  fi
  return 1
}

# --- step wrappers ---

step_baseline() {
  local model=$1 run_name=$2 adapter=${3:-}
  local marker="$(host_dir "$model" "$run_name")/baseline_metrics.json"
  already_done "$marker" "baseline $run_name" && return 0
  echo "    [run]  baseline $run_name"
  local overrides=(
    model.name="$model"
    data.test_file="$CPT_DATA_DIR/test.parquet"
    data.max_test_rows=0
    run.run_name="$run_name"
  )
  [ -n "$adapter" ] && overrides+=("model.adapter_path=$adapter")
  docker compose run --rm baseline \
      --config stages/downstream/configs/eval_baseline_example.yaml \
      "${overrides[@]}"
}

step_zs_inference() {
  local model=$1 task=$2 run_name=$3
  local test_file
  case "$task" in
    misalignment_detection) test_file="$ALIGN_TEST" ;;
    malicious_detection)    test_file="$MAL_TEST" ;;
    *) echo "    [BUG] unknown task: $task" >&2; return 1 ;;
  esac
  local marker="$(host_dir "$model" "$run_name")/metrics_${task}.json"
  already_done "$marker" "zero-shot $task" && return 0
  echo "    [run]  zero-shot inference $task — $run_name"
  docker compose run --rm inference \
      --config stages/downstream/configs/inference_example.yaml \
      model.name="$model" \
      "data.task=$task" \
      "data.test_file=$test_file" \
      data.max_rows=0 \
      run.run_name="$run_name"
}

step_sft_inference() {
  local model=$1 task=$2 adapter=$3 run_name=$4
  local test_file
  case "$task" in
    misalignment_detection) test_file="$ALIGN_TEST" ;;
    malicious_detection)    test_file="$MAL_TEST" ;;
  esac
  local marker="$(host_dir "$model" "$run_name")/metrics_${task}.json"
  already_done "$marker" "sft inference $task" && return 0
  echo "    [run]  sft-adapter inference $task — $run_name"
  docker compose run --rm inference \
      --config stages/downstream/configs/inference_example.yaml \
      model.name="$model" \
      "model.adapter_path=$adapter" \
      "data.task=$task" \
      "data.test_file=$test_file" \
      data.max_rows=0 \
      run.run_name="$run_name"
}

step_cpt() {
  local model=$1
  local marker="$(host_dir "$model" "$NAME_CPT")/final_model"
  already_done "$marker" "CPT $NAME_CPT" && return 0
  echo "    [run]  CPT sub-stage 1"
  docker compose run --rm pretraining \
      stages/pretraining/train.py \
      --config stages/pretraining/configs/full_cpt_v1_quarter.yaml \
      model.name="$model" \
      run.run_name="$NAME_CPT" \
      "data.train_file=$CPT_DATA_DIR/train.parquet" \
      "data.validation_file=$CPT_DATA_DIR/val.parquet" \
      "data.test_file=$CPT_DATA_DIR/test.parquet" \
      data.max_seq_length=4096
}

step_sft() {
  local model=$1 service=$2 config=$3 cpt_adapter=$4 run_name=$5
  local marker="$(host_dir "$model" "$run_name")/final_model"
  already_done "$marker" "SFT $run_name" && return 0
  echo "    [run]  SFT $run_name"
  docker compose run --rm "$service" \
      stages/downstream/sft/train.py \
      --config "$config" \
      model.name="$model" \
      "model.phase1_adapter_path=$cpt_adapter" \
      run.run_name="$run_name"
}

# --- main loop ---

FAILED_MODELS=()
SUCCEEDED_MODELS=()

for MODEL in "${MODELS[@]}"; do
  echo ""
  echo "============================================================"
  echo "  Model: $MODEL"
  echo "============================================================"

  CPT_ADAPTER=$(in_dir "$MODEL" "$NAME_CPT")/final_model
  SFT_ALIGN_ADAPTER=$(in_dir "$MODEL" "$NAME_SFT_ALIGN")/final_model
  SFT_MAL_ADAPTER=$(in_dir "$MODEL" "$NAME_SFT_MAL")/final_model

  any_failed=0
  step_baseline     "$MODEL" "$NAME_BASELINE_PRE"                                      || any_failed=1
  step_zs_inference "$MODEL" misalignment_detection "$NAME_ZS_ALIGN_PRE"               || any_failed=1
  step_zs_inference "$MODEL" malicious_detection    "$NAME_ZS_MAL_PRE"                 || any_failed=1
  step_cpt          "$MODEL"                                                            || { any_failed=1; echo "    [skip downstream — CPT failed]"; FAILED_MODELS+=("$MODEL"); continue; }
  step_baseline     "$MODEL" "$NAME_BASELINE_POST" "$CPT_ADAPTER"                      || any_failed=1
  step_zs_inference "$MODEL" misalignment_detection "$NAME_ZS_ALIGN_POST"              || any_failed=1
  step_zs_inference "$MODEL" malicious_detection    "$NAME_ZS_MAL_POST"                || any_failed=1
  step_sft           "$MODEL" sft-align \
                     stages/downstream/configs/sft_misalignment_example.yaml \
                     "$CPT_ADAPTER" "$NAME_SFT_ALIGN"                                   || any_failed=1
  step_sft_inference "$MODEL" misalignment_detection "$SFT_ALIGN_ADAPTER" "$NAME_EVAL_SFT_ALIGN" || any_failed=1
  step_sft           "$MODEL" sft-mal \
                     stages/downstream/configs/sft_malicious_example.yaml \
                     "$CPT_ADAPTER" "$NAME_SFT_MAL"                                     || any_failed=1
  step_sft_inference "$MODEL" malicious_detection "$SFT_MAL_ADAPTER" "$NAME_EVAL_SFT_MAL" || any_failed=1

  if [ "$any_failed" -ne 0 ]; then
    FAILED_MODELS+=("$MODEL")
  else
    SUCCEEDED_MODELS+=("$MODEL")
  fi
done

echo ""
echo "============================================================"
echo "  Summary"
echo "============================================================"
echo "  Succeeded (${#SUCCEEDED_MODELS[@]}): ${SUCCEEDED_MODELS[*]:-(none)}"
echo "  Failed    (${#FAILED_MODELS[@]}): ${FAILED_MODELS[*]:-(none)}"
echo ""
echo "  Aggregate the results:"
echo "    docker compose run --rm --no-deps --entrypoint python pretraining \\"
echo "        scripts/aggregate_eval.py /app/outputs/runs"
echo ""

[ "${#FAILED_MODELS[@]}" -eq 0 ]
