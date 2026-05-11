#!/bin/bash
#SBATCH -J downstream_infer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err

# Usage:
#   sbatch run_inference.sh configs/inference_example.yaml [overrides...]

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG="${1:?usage: run_inference.sh <config.yaml> [overrides...]}"
shift

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"
python -u -m stages.downstream.inference --config "${CONFIG}" "$@"
