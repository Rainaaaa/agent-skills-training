#!/bin/bash
#SBATCH -J smoke_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH -A r00954
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=cz1@iu.edu
#SBATCH --output=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-training/log/%x_%j.log
#SBATCH --error=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-training/log/%x_%j.err
#
# Pipeline smoke test for one backbone model. Submits to:
#   - hopper partition on Quartz (H100 80GB)
#   - gpu    partition on BigRed200 (A100 40GB)
# Pass `-p hopper` / `-p gpu` on the sbatch CLI from submit_smoke_all.sh.
#
# Runs only Stage 1 (CPT) with smoke caps from KELLY_IT_HANDOFF.md §4:
#   training.max_steps=4, max_*_rows tiny, max_seq_length=1024.
#
# Usage (called from submit_smoke_all.sh):
#   sbatch -p hopper -J smoke_<model> scripts/run_smoke.sh <model_name>

set -euo pipefail

MODEL="${1:?Usage: $0 <model_name>}"

REPO_ROOT=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-training
CONDA_BASE=/N/slate/cz1/miniconda3
CONDA_ENV_PATH=/N/slate/cz1/conda/envs/AgentSkillsOSS
DATA_ROOT=/N/project/AdversarialModeling/datasets/agent_skills/misalignment

CPT_DATA_DIR="${DATA_ROOT}/full_cpt/full_cpt_v3/stage1"   # challenge-scrubbed v3

# ----- env setup -----
mkdir -p "${REPO_ROOT}/log" "${REPO_ROOT}/outputs/runs"

# Activate by absolute path — the env lives under /N/slate/cz1/conda/envs
# but conda itself is in /N/slate/cz1/miniconda3, so name-based lookup fails.
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
else
  echo "[smoke] FATAL: ${CONDA_BASE}/etc/profile.d/conda.sh not found" >&2
  exit 64
fi

# Sanity check — we must be on the env's python, not /usr/bin/python3.6.
PY="$(command -v python)"
echo "[smoke] python=${PY}"
case "${PY}" in
  "${CONDA_ENV_PATH}/bin/python") : ;;
  *) echo "[smoke] FATAL: conda activate did not switch python to ${CONDA_ENV_PATH}/bin/python" >&2; exit 64 ;;
esac

# deepspeed import in newer accelerate needs CUDA_HOME (no-op since we don't use deepspeed).
if [[ -z "${CUDA_HOME:-}" ]]; then
  for c in /N/soft/sles15sp6/cuda/gnu/12.6 /N/soft/sles15sp6/cuda/gnu/12.2 /N/soft/rhel8/cuda/12.6 /usr/local/cuda; do
    [[ -x "${c}/bin/nvcc" ]] && export CUDA_HOME="${c}" && break
  done
fi
[[ -n "${CUDA_HOME:-}" ]] && export PATH="${CUDA_HOME}/bin:${PATH}" && export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# HF cache lands inside the repo's outputs/ so it survives container/host swaps.
export HF_HOME="${REPO_ROOT}/outputs/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

cd "${REPO_ROOT}"

# ----- machine + partition log -----
echo "[smoke] host=$(hostname) partition=${SLURM_JOB_PARTITION:-?} job=${SLURM_JOB_ID:-?}"
echo "[smoke] model=${MODEL}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || true
fi

# ----- Stage 1 (CPT) smoke -----
echo "[smoke] Stage 1 — pretraining (CPT) smoke"
python -u -m stages.pretraining.train \
    --config stages/pretraining/configs/example.yaml \
    model.name="${MODEL}" \
    run.output_root="${REPO_ROOT}/outputs/runs" \
    run.run_name=smoke_pretrain \
    "data.train_file=${CPT_DATA_DIR}/train.parquet" \
    "data.validation_file=${CPT_DATA_DIR}/val.parquet" \
    "data.test_file=${CPT_DATA_DIR}/test.parquet" \
    training.max_steps=4 \
    data.max_train_rows=64 \
    data.max_validation_rows=8 \
    data.max_test_rows=8 \
    data.max_seq_length=1024 \
    model.attn_implementation=sdpa

echo "[smoke] DONE model=${MODEL}"
