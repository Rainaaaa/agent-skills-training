#!/bin/bash
#SBATCH -J lp_hcl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH -A r00954
#SBATCH -p gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cz1@iu.edu
#SBATCH --output=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-training/lp_hcl/log/%x_%j.log
#SBATCH --error=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-training/lp_hcl/log/%x_%j.err

# Phase 2 LP-HCL launcher (LoRA, smoke-test by default).
#
# Usage:
#   sbatch scripts/run_lp_hcl.sh configs/example_lp_hcl.yaml [key=value ...]
#
# Or run locally on an interactive GPU node:
#   bash scripts/run_lp_hcl.sh configs/example_lp_hcl.yaml training.max_steps=20

set -euo pipefail

CONFIG="${1:-configs/example_lp_hcl.yaml}"
shift || true

PHASE_ROOT=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-training/lp_hcl
SRC_ROOT=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-training/src
CONDA_BASE=/N/slate/cz1/miniconda3
CONDA_ENV=BiasDetection

mkdir -p "${PHASE_ROOT}/log" "${PHASE_ROOT}/output"

if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

# Accelerate's Trainer.unwrap_model unconditionally imports deepspeed if the
# package is installed, and deepspeed's op-compat probe needs a real
# CUDA_HOME at import time. Point at the cluster's CUDA toolkit so the
# import succeeds cleanly even though this run does NOT use deepspeed.
if [[ -z "${CUDA_HOME:-}" ]]; then
  for candidate in \
      /N/soft/sles15sp6/cuda/gnu/12.6 \
      /N/soft/sles15sp6/cuda/gnu/12.2 \
      /N/soft/sles15sp6/cuda/gnu/11.8 \
      /usr/local/cuda; do
    if [[ -x "${candidate}/bin/nvcc" ]]; then
      export CUDA_HOME="${candidate}"
      break
    fi
  done
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  echo "[launcher] CUDA_HOME=${CUDA_HOME}"
else
  echo "[launcher] WARNING: CUDA_HOME could not be resolved; deepspeed import may crash."
fi

export PYTHONPATH="${SRC_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_VERBOSITY=info
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

cd "${PHASE_ROOT}"

# Resolve run.output_dir from the YAML (same logic as full_cpt) and symlink
# the model-name parent dir into the local output/ tree so checkpoints / logs
# show up here even when the actual data lives on /N/project/.
RESOLVED="$(PYTHONPATH="${SRC_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  /N/slate/cz1/conda/envs/BiasDetection/bin/python - "${CONFIG}" <<'PY'
import sys, yaml
from pathlib import Path
from common.registry import sanitize_for_path

cfg_path = sys.argv[1]
with open(cfg_path) as f:
    cfg = yaml.safe_load(f) or {}

run = cfg.get("run", {}) or {}
model = cfg.get("model", {}) or {}

if run.get("output_dir"):
    out = Path(run["output_dir"]).expanduser().resolve()
    name = model.get("name") or ""
else:
    name = model.get("name") or ""
    if not name:
        print(""); print("")
        sys.exit(0)
    root = run.get("output_root")
    run_name = run.get("run_name") or run.get("experiment_name")
    if not root or not run_name:
        print(""); print("")
        sys.exit(0)
    out = (Path(root).expanduser() / sanitize_for_path(name) / run_name).resolve()
print(out)
print(name)
PY
)"
RUN_OUT_DIR="$(echo "${RESOLVED}" | sed -n '1p')"
MODEL_NAME="$(echo "${RESOLVED}" | sed -n '2p')"

if [[ -n "${RUN_OUT_DIR}" ]]; then
  mkdir -p "${RUN_OUT_DIR}"
  if [[ -n "${MODEL_NAME}" ]]; then
    MODEL_DIR_REMOTE="$(dirname "${RUN_OUT_DIR}")"
    SAFE_NAME="$(basename "${MODEL_DIR_REMOTE}")"
    ln -sfn "${MODEL_DIR_REMOTE}" "${PHASE_ROOT}/output/${SAFE_NAME}"
    echo "[launcher] Symlinked output/${SAFE_NAME} -> ${MODEL_DIR_REMOTE}"
  else
    RUN_NAME="$(basename "${RUN_OUT_DIR}")"
    ln -sfn "${RUN_OUT_DIR}" "${PHASE_ROOT}/output/${RUN_NAME}"
    echo "[launcher] Symlinked output/${RUN_NAME} -> ${RUN_OUT_DIR}"
  fi
fi

NUM_GPUS="${SLURM_GPUS_ON_NODE:-}"
if [[ -z "${NUM_GPUS}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS="$(nvidia-smi -L | wc -l)"
  else
    NUM_GPUS=1
  fi
fi
echo "[launcher] NUM_GPUS=${NUM_GPUS}"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  accelerate launch \
    --num_processes="${NUM_GPUS}" \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    "${SRC_ROOT}/lp_hcl/train.py" --config "${CONFIG}" "$@"
else
  python -u "${SRC_ROOT}/lp_hcl/train.py" --config "${CONFIG}" "$@"
fi
