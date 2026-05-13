#!/usr/bin/env bash
# Dispatch to any stage entry point.
#
#   docker run --rm --gpus all agent-skills-training stages/pretraining/train.py --help
#   docker run --rm --gpus all agent-skills-training stages/hcl/train.py --help
#   docker run --rm --gpus all agent-skills-training stages/downstream/sft/train.py --help
#   docker run --rm --gpus all agent-skills-training stages/downstream/inference.py --help
#   docker run --rm --gpus all agent-skills-training stages/downstream/eval_baseline.py --help
#
# Trainer scripts (*/train.py) auto-launch under torchrun when >1 GPU is
# visible; everything else stays single-process. Override with
# AGENTSKILLS_NPROC=<int> in the env (e.g. AGENTSKILLS_NPROC=1 to force
# single-process even on a multi-GPU host).
set -euo pipefail

if [ "$#" -eq 0 ]; then
  set -- --help
fi

case "$1" in
  --help|-h)
    cat <<'EOF'
agent-skills-training container

Stage entry points (each accepts --help for full flags):

  Pretraining (Phase 1 — CLM continued pretraining)
      stages/pretraining/train.py        --config <yaml>

  HCL (Phase 2 — hierarchical contrastive learning post-CPT)
      stages/hcl/train.py                --config <yaml>

  Downstream
      stages/downstream/sft/train.py     --config <sft_*.yaml>
      stages/downstream/inference.py     --config <inference.yaml>
      stages/downstream/eval_baseline.py --config <eval_baseline.yaml>

Required mounts:
  --gpus all                                            # CUDA access
  -v $(pwd)/outputs:/app/outputs                        # checkpoints + logs
  -v <dataset_root>:/data                               # training data
  -v <model_path.json>:/app/model_path.json:ro          # model registry

Multi-GPU (trainer scripts only):
  Trainer scripts (*/train.py) auto-detect visible GPUs and launch under
  torchrun when count >= 2. Inference / eval scripts always stay
  single-process (they're not DDP-aware and would duplicate work).
  Override: AGENTSKILLS_NPROC=<int>   # 1 forces single-process;
                                      # any other positive int picks that
                                      # nproc_per_node explicitly.

Pass overrides on the CLI:
  ... stages/pretraining/train.py --config foo.yaml training.max_steps=20
EOF
    exit 0
    ;;
esac

script="$1"; shift

# Non-python targets are exec'd verbatim (rare, but supported for ad-hoc
# tools shipped inside the image).
if [[ "$script" != *.py ]]; then
  exec "$script" "$@"
fi

if [ ! -f "/app/$script" ]; then
  echo "[entrypoint] python script not found: $script" >&2
  exit 64
fi

cd /app
# Convert path "stages/foo/bar.py" → module "stages.foo.bar" so the package
# imports inside (e.g. `from common.config import …`) work.
module="$(echo "${script%.py}" | tr '/' '.')"

# Determine launch mode:
#   1. AGENTSKILLS_NPROC overrides everything (must be a positive int).
#   2. For trainer scripts (*/train.py), auto-detect via
#      torch.cuda.device_count() — honors CUDA_VISIBLE_DEVICES.
#   3. Everything else stays single-process.
nproc=""
if [ -n "${AGENTSKILLS_NPROC:-}" ]; then
  nproc="$AGENTSKILLS_NPROC"
elif [[ "$script" == */train.py ]]; then
  nproc="$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo 1)"
fi
: "${nproc:=1}"

if ! [[ "$nproc" =~ ^[0-9]+$ ]] || [ "$nproc" -lt 1 ]; then
  echo "[entrypoint] invalid AGENTSKILLS_NPROC=$nproc; must be a positive integer" >&2
  exit 64
fi

if [ "$nproc" -gt 1 ]; then
  echo "[entrypoint] DDP launch: torchrun --standalone --nproc_per_node=$nproc -m $module $*"
  exec torchrun --standalone --nproc_per_node="$nproc" -m "$module" "$@"
fi

echo "[entrypoint] single-process launch: python -m $module $*"
exec python -u -m "$module" "$@"
