#!/bin/bash
#SBATCH -J baseline_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH -A r00954
#SBATCH -p gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cz1@iu.edu
#SBATCH --output=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-training/baseline/log/%x_%j.log
#SBATCH --error=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-training/baseline/log/%x_%j.err

# Baseline LLM evaluation launcher (one model per job).
#
# Usage:
#   sbatch --gres=gpu:N --job-name=baseline_<model> scripts/run_baseline.sh \
#       configs/<model>.yaml [override=value ...]

set -euo pipefail

CONFIG="${1:-}"
if [[ -z "${CONFIG}" ]]; then
  echo "usage: $0 <config.yaml> [override=value ...]" >&2
  exit 2
fi
shift

PHASE_ROOT=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-training/baseline
SRC_ROOT=/N/slate/cz1/GitHub/AgentSkills-OSS/agent-skills-training/src
CONDA_BASE=/N/slate/cz1/miniconda3
CONDA_ENV=AgentSkillsOSS

mkdir -p "${PHASE_ROOT}/log" "${PHASE_ROOT}/output"

if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

# Same CUDA fix as full_cpt: ensure deepspeed's import probe finds a real toolkit.
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
fi

export PYTHONPATH="${SRC_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_VERBOSITY=info
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

cd "${PHASE_ROOT}"

# Symlink output/<MODEL_NAME>/ -> <output_root>/<MODEL_NAME>/ so all per-model
# runs are browsable in the local repo. Matches the full_cpt launcher.
RESOLVED="$(/N/slate/cz1/conda/envs/AgentSkillsOSS/bin/python - "${CONFIG}" <<'PY'
import sys, yaml
from pathlib import Path
from common.registry import sanitize_for_path

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f) or {}
run = cfg.get("run", {}) or {}
model = cfg.get("model", {}) or {}

if run.get("output_dir"):
    out = Path(run["output_dir"]).expanduser().resolve()
    name = model.get("name") or ""
else:
    name = model.get("name") or ""
    root = run.get("output_root")
    rname = run.get("run_name") or run.get("experiment_name")
    if not (name and root and rname):
        print(""); print(""); sys.exit(0)
    out = (Path(root).expanduser() / sanitize_for_path(name) / rname).resolve()
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

python -u "${SRC_ROOT}/baseline/eval_baseline.py" --config "${CONFIG}" "$@"
