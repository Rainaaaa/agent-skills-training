#!/usr/bin/env bash
# Dispatch to any stage entry point.
#
#   docker run --rm --gpus all agent-skills-training stages/pretraining/train.py --help
#   docker run --rm --gpus all agent-skills-training stages/hcl/train.py --help
#   docker run --rm --gpus all agent-skills-training stages/downstream/sft/train.py --help
#   docker run --rm --gpus all agent-skills-training stages/downstream/inference.py --help
#   docker run --rm --gpus all agent-skills-training stages/downstream/eval_baseline.py --help
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

Pass overrides on the CLI:
  ... stages/pretraining/train.py --config foo.yaml training.max_steps=20
EOF
    exit 0
    ;;
esac

# Path ending in .py → python -m the dotted module; everything else exec'd.
script="$1"; shift
if [[ "$script" == *.py ]]; then
  if [ -f "/app/$script" ]; then
    cd /app
    # Convert path "stages/foo/bar.py" → module "stages.foo.bar" so the
    # package imports inside (e.g. `from common.config import …`) work.
    module="$(echo "${script%.py}" | tr '/' '.')"
    exec python -u -m "$module" "$@"
  fi
  echo "[entrypoint] python script not found: $script" >&2
  exit 64
fi
exec "$script" "$@"
