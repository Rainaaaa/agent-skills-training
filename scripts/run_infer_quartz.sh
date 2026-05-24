#!/bin/bash
#SBATCH -J infer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -A r00954
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=cz1@iu.edu
#SBATCH --output=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-training/log/%x_%j.log
#SBATCH --error=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-training/log/%x_%j.err
#
# Single downstream-inference launcher for Quartz hopper / BR200 gpu.
# Mirrors run_cpt_quartz.sh's env setup but invokes stages.downstream.inference.
#
# Usage:
#   sbatch -p hopper --qos=hopper scripts/run_infer_quartz.sh \
#       stages/downstream/configs/inference_misalignment_challenge.yaml \
#       model.name=Foundation-Sec-8B-Reasoning \
#       run.run_name=zs_misalignment_raw \
#       inference.n_shot=0

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml> [key=value ...]}"
shift

REPO_ROOT=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-training
CONDA_BASE=/N/slate/cz1/miniconda3
CONDA_ENV_PATH=/N/slate/cz1/conda/envs/AgentSkillsOSS

mkdir -p "${REPO_ROOT}/log" "${REPO_ROOT}/outputs/runs"

if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
else
  echo "[infer] FATAL: ${CONDA_BASE}/etc/profile.d/conda.sh not found" >&2; exit 64
fi
PY="$(command -v python)"
[[ "${PY}" == "${CONDA_ENV_PATH}/bin/python" ]] || {
  echo "[infer] FATAL: wrong python=${PY}" >&2; exit 64
}

if [[ -z "${CUDA_HOME:-}" ]]; then
  for c in /N/soft/sles15sp6/cuda/gnu/12.6 /N/soft/sles15sp6/cuda/gnu/12.2 /usr/local/cuda; do
    [[ -x "${c}/bin/nvcc" ]] && export CUDA_HOME="${c}" && break
  done
fi
[[ -n "${CUDA_HOME:-}" ]] && export PATH="${CUDA_HOME}/bin:${PATH}" \
  && export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HOME="${REPO_ROOT}/outputs/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

cd "${REPO_ROOT}"

echo "[infer] host=$(hostname) partition=${SLURM_JOB_PARTITION:-?} job=${SLURM_JOB_ID:-?}"
echo "[infer] config=${CONFIG}"
echo "[infer] overrides:" "$@"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true

exec python -u -m stages.downstream.inference \
    --config "${CONFIG}" "$@"
