#!/bin/bash
#SBATCH -J cpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH -A r00954
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cz1@iu.edu
#SBATCH --output=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-training/log/%x_%j.log
#SBATCH --error=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-training/log/%x_%j.err
#
# Real-training CPT launcher for Quartz hopper / BR200 gpu.
#
# Usage:
#   sbatch -p hopper --qos=hopper scripts/run_cpt_quartz.sh \
#       stages/pretraining/configs/full_cpt_v3_quarter.yaml \
#       model.name=llama3.1-8b \
#       run.run_name=cpt_substage1 \
#       data.train_file=/path/.../stage1/train.parquet  ...
#
# Sub-stage-2 chained launch (set RESUME_PARENT_DIR env to sub-stage-1's
# output dir — the script picks the highest-numbered checkpoint inside it
# at job-start time and passes it as run.resume_from_checkpoint):
#   sbatch --dependency=afterok:<sub1_jobid> \
#       --export=ALL,RESUME_PARENT_DIR=/path/.../cpt_substage1 \
#       -p hopper --qos=hopper \
#       scripts/run_cpt_quartz.sh stages/pretraining/configs/full_cpt_v3_quarter.yaml \
#       model.name=llama3.1-8b run.run_name=cpt_substage2 \
#       data.train_file=.../stage2/train.parquet ... \
#       data.max_seq_length=10240 training.lr_scheduler_type=constant

set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml> [key=value ...]}"
shift

REPO_ROOT=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-training
CONDA_BASE=/N/slate/cz1/miniconda3
CONDA_ENV_PATH=/N/slate/cz1/conda/envs/AgentSkillsOSS

mkdir -p "${REPO_ROOT}/log" "${REPO_ROOT}/outputs/runs"

# ----- env -----
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_PATH}"
else
  echo "[cpt] FATAL: ${CONDA_BASE}/etc/profile.d/conda.sh not found" >&2; exit 64
fi
PY="$(command -v python)"
[[ "${PY}" == "${CONDA_ENV_PATH}/bin/python" ]] || {
  echo "[cpt] FATAL: wrong python=${PY}" >&2; exit 64
}

# CUDA toolkit (deepspeed import touches CUDA_HOME even though we don't use it).
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export HF_HOME="${REPO_ROOT}/outputs/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

cd "${REPO_ROOT}"

echo "[cpt] host=$(hostname) partition=${SLURM_JOB_PARTITION:-?} job=${SLURM_JOB_ID:-?}"
echo "[cpt] python=${PY}"
echo "[cpt] config=${CONFIG}"
echo "[cpt] overrides:" "$@"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || true

# ----- optional chained resume from a parent run's latest checkpoint -----
EXTRA_OVERRIDES=()
if [[ -n "${RESUME_PARENT_DIR:-}" ]]; then
  if [[ ! -d "${RESUME_PARENT_DIR}" ]]; then
    echo "[cpt] FATAL: RESUME_PARENT_DIR=${RESUME_PARENT_DIR} does not exist" >&2
    exit 64
  fi
  LATEST_CKPT="$(ls -d "${RESUME_PARENT_DIR}"/checkpoint-* 2>/dev/null \
                  | sort -t- -k2 -n | tail -1)"
  if [[ -z "${LATEST_CKPT}" ]]; then
    echo "[cpt] FATAL: no checkpoint-* dirs under ${RESUME_PARENT_DIR}" >&2
    exit 64
  fi
  echo "[cpt] Resuming from latest checkpoint: ${LATEST_CKPT}"
  EXTRA_OVERRIDES+=("run.resume_from_checkpoint=${LATEST_CKPT}")
fi

# ----- launch (DDP if >1 GPU, plain python otherwise) -----
NUM_GPUS="${SLURM_GPUS_ON_NODE:-}"
if [[ -z "${NUM_GPUS}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  NUM_GPUS="$(nvidia-smi -L | wc -l)"
fi
NUM_GPUS="${NUM_GPUS:-1}"
echo "[cpt] NUM_GPUS=${NUM_GPUS}"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  exec torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
      -m stages.pretraining.train \
      --config "${CONFIG}" "$@" "${EXTRA_OVERRIDES[@]}"
else
  exec python -u -m stages.pretraining.train \
      --config "${CONFIG}" "$@" "${EXTRA_OVERRIDES[@]}"
fi
