#!/bin/bash
#SBATCH -J pipe_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH -A r00954
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=cz1@iu.edu
#SBATCH --output=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-training/log/%x_%j.log
#SBATCH --error=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-training/log/%x_%j.err
#
# End-to-end smoke test for the WHOLE pipeline on a single backbone, against
# the new (challenge-scrubbed) datasets — full_cpt_v3 / pl_hcl_v2 / sft_challenge:
#
#   1. Stage 1 — CPT pretraining (full_cpt_v3/stage1)
#   2. Stage 2 — HCL contrastive  (pl_hcl_v2/stage1/pairs_*),  chained on Stage 1 LoRA
#   3. Stage 3 — SFT misalignment (sft/sft_align/align_*),     chained on Stage 1 LoRA
#                — sft_align_challenge/ is eval-only, never an SFT training source.
#   4. Stage 3 — Inference on the gold-label challenge test set, using the
#                SFT-misalignment adapter from step 3.
#   5. Stage 3 — eval_baseline (CLM perplexity on full_cpt_v3/stage1/test.parquet)
#
# Each step uses tiny smoke caps (max_steps=4, ~8-64 rows). Wall ≈ 15-30 min
# on a single H100. Per-step failures do NOT abort the script — the summary
# at the end shows which steps passed/failed.
#
# Usage:
#   sbatch -p hopper --qos=hopper scripts/run_pipeline_smoke.sh [model_name]
#   sbatch -p gpu               scripts/run_pipeline_smoke.sh [model_name]   # BR200
#
# Default model = Foundation-Sec-8B-Reasoning (validated on Jetstream).

set -uo pipefail

MODEL="${1:-Foundation-Sec-8B-Reasoning}"

REPO_ROOT=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-training
CONDA_BASE=/N/slate/cz1/miniconda3
CONDA_ENV_PATH=/N/slate/cz1/conda/envs/AgentSkillsOSS
DATA_ROOT=/N/project/AdversarialModeling/datasets/agent_skills/misalignment

CPT_DIR="${DATA_ROOT}/full_cpt/full_cpt_v3/stage1"
HCL_DIR="${DATA_ROOT}/pl_hcl/pl_hcl_v2/stage1"
SFT_TRAIN_DIR="${DATA_ROOT}/sft/sft_align"           # broad scanner-labeled set, for SFT training
SFT_EVAL_DIR="${DATA_ROOT}/sft/sft_align_challenge"   # gold-label challenge set, eval-only

mkdir -p "${REPO_ROOT}/log" "${REPO_ROOT}/outputs/runs"

# ----- env -----
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
else
  echo "[pipe] FATAL: ${CONDA_BASE}/etc/profile.d/conda.sh not found" >&2; exit 64
fi
PY="$(command -v python)"
[[ "${PY}" == "${CONDA_ENV_PATH}/bin/python" ]] || { echo "[pipe] FATAL: wrong python=${PY}" >&2; exit 64; }

if [[ -z "${CUDA_HOME:-}" ]]; then
  for c in /N/soft/sles15sp6/cuda/gnu/12.6 /N/soft/sles15sp6/cuda/gnu/12.2 /usr/local/cuda; do
    [[ -x "${c}/bin/nvcc" ]] && export CUDA_HOME="${c}" && break
  done
fi
[[ -n "${CUDA_HOME:-}" ]] && export PATH="${CUDA_HOME}/bin:${PATH}" && export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HOME="${REPO_ROOT}/outputs/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

cd "${REPO_ROOT}"

echo "[pipe] host=$(hostname) partition=${SLURM_JOB_PARTITION:-?} job=${SLURM_JOB_ID:-?}"
echo "[pipe] python=${PY}"
echo "[pipe] model=${MODEL}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true

# ----- shared run-name + adapter paths -----
# Bump the suffix when the smoke recipe changes so already_done() doesn't
# silently re-use last-run's artifacts and skip the new exercise.
RUN_BASE=pipe_smoke
CPT_RUN="${RUN_BASE}_cpt"
HCL_RUN="${RUN_BASE}_hcl_v4"        # v4 = stratify by pair_kind (3-way), breakdown emitted
SFT_RUN="${RUN_BASE}_sft_align"
INF_RUN="${RUN_BASE}_infer_align"
BASE_RUN="${RUN_BASE}_eval_baseline"

OUT_ROOT="${REPO_ROOT}/outputs/runs"
# Match common/registry.py:sanitize_for_path — replaces '/' with '__' and
# spaces with '_'; dots stay (so "llama3.1-8b" -> "llama3.1-8b").
SAFE_MODEL="$(printf '%s' "${MODEL}" | sed 's|/|__|g; s| |_|g')"
CPT_ADAPTER="${OUT_ROOT}/${SAFE_MODEL}/${CPT_RUN}/final_model"
SFT_ALIGN_ADAPTER="${OUT_ROOT}/${SAFE_MODEL}/sft_misalignment_detection__${SFT_RUN}/final_model"

# ----- step runner: name, marker, command... -----
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

# ----- 1) Stage 1 CPT -----
run_step "1_cpt" "${CPT_ADAPTER}" \
  python -u -m stages.pretraining.train \
    --config stages/pretraining/configs/example.yaml \
    model.name="${MODEL}" \
    run.output_root="${OUT_ROOT}" \
    run.run_name="${CPT_RUN}" \
    "data.train_file=${CPT_DIR}/train.parquet" \
    "data.validation_file=${CPT_DIR}/val.parquet" \
    "data.test_file=${CPT_DIR}/test.parquet" \
    training.max_steps=4 \
    data.max_train_rows=64 data.max_validation_rows=8 data.max_test_rows=8 \
    data.max_seq_length=1024 \
    model.attn_implementation=sdpa

# ----- 2) Stage 2 HCL (chained) -----
# Sized so the eval batch is statistically meaningful: the pair-test split is
# ~88% label=0 / 12% label=1, so 8 random eval rows often have ZERO positives
# (P ≈ 36%) — `pos_prob_mean` then collapses to 0 and accuracy looks like a
# bug. 64-row evals give P(no positives) ≈ 0.03% and a real signal.
# max_steps=20 also lets the BCE-on-cosine head move past its random init.
# ce_loss_weight=0.5 turns on the supplementary InfoNCE term so the
# encoder gets gradient from the positive subset too (the default config
# leaves it off for unit-test minimality).
if [[ -d "${CPT_ADAPTER}" ]]; then
  HCL_MARKER="${OUT_ROOT}/${SAFE_MODEL}/${HCL_RUN}/final_model"
  run_step "2_hcl" "${HCL_MARKER}" \
    python -u -m stages.hcl.train \
      --config stages/hcl/configs/example.yaml \
      model.name="${MODEL}" \
      run.output_root="${OUT_ROOT}" \
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
  echo "[pipe] SKIP step 2 — Stage 1 adapter missing at ${CPT_ADAPTER}"
  RESULTS[2_hcl]="SKIP(no-cpt-adapter)"
fi

# ----- 3) Stage 3 SFT misalignment (challenge, chained on CPT) -----
if [[ -d "${CPT_ADAPTER}" ]]; then
  run_step "3_sft_align" "${SFT_ALIGN_ADAPTER}" \
    python -u -m stages.downstream.sft.train \
      --config stages/downstream/configs/sft_misalignment_example.yaml \
      model.name="${MODEL}" \
      run.output_root="${OUT_ROOT}" \
      run.run_name="${SFT_RUN}" \
      "model.phase1_adapter_path=${CPT_ADAPTER}" \
      "data.train_file=${SFT_TRAIN_DIR}/align_train.parquet" \
      "data.validation_file=${SFT_TRAIN_DIR}/align_val.parquet" \
      "data.test_file=${SFT_TRAIN_DIR}/align_test.parquet" \
      training.max_steps=4 \
      data.max_seq_length=512
else
  echo "[pipe] SKIP step 3 — Stage 1 adapter missing"
  RESULTS[3_sft_align]="SKIP(no-cpt-adapter)"
fi

# ----- 4) Inference using SFT-misalignment adapter -----
if [[ -d "${SFT_ALIGN_ADAPTER}" ]]; then
  INF_DIR="${OUT_ROOT}/${SAFE_MODEL}/infer_misalignment_detection__${INF_RUN}"
  INF_MARKER="${INF_DIR}/metrics_misalignment_detection.json"
  run_step "4_inference" "${INF_MARKER}" \
    python -u -m stages.downstream.inference \
      --config stages/downstream/configs/inference_example.yaml \
      model.name="${MODEL}" \
      run.output_root="${OUT_ROOT}" \
      run.run_name="${INF_RUN}" \
      "model.adapter_path=${SFT_ALIGN_ADAPTER}" \
      data.task=misalignment_detection \
      "data.test_file=${SFT_EVAL_DIR}/align_challenge_test.parquet" \
      data.max_rows=8
else
  echo "[pipe] SKIP step 4 — SFT-align adapter missing"
  RESULTS[4_inference]="SKIP(no-sft-adapter)"
fi

# ----- 5) eval_baseline — CLM perplexity on the CPT test split -----
BASE_DIR="${OUT_ROOT}/${SAFE_MODEL}/${BASE_RUN}"
BASE_MARKER="${BASE_DIR}/baseline_metrics.json"
run_step "5_eval_baseline" "${BASE_MARKER}" \
  python -u -m stages.downstream.eval_baseline \
    --config stages/downstream/configs/eval_baseline_example.yaml \
    model.name="${MODEL}" \
    run.output_root="${OUT_ROOT}" \
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
echo "  Output root: ${OUT_ROOT}/${SAFE_MODEL}/"
echo "============================================================"

exit "$fail_count"
