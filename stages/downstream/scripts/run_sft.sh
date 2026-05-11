#!/bin/bash
#SBATCH -J downstream_sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err

# Usage:
#   sbatch run_sft.sh configs/sft_malicious_example.yaml [section.key=value ...]
#   sbatch run_sft.sh configs/sft_misalignment_example.yaml

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG="${1:?usage: run_sft.sh <config.yaml> [overrides...]}"
shift

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"
python -u -m stages.downstream.sft.train --config "${CONFIG}" "$@"
