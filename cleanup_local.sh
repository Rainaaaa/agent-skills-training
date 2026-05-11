#!/usr/bin/env bash
# Optional local cleanup — wipes Python caches + training-run artifacts.
#
#   bash cleanup_local.sh                # actually delete
#   DRYRUN=1 bash cleanup_local.sh       # preview only
#
# CHECKPOINTS ARE PRESERVED by default (they're expensive to rebuild). To
# also delete them, set WIPE_CHECKPOINTS=1.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run() {
  if [ "${DRYRUN:-0}" = "1" ]; then
    echo "[dry-run] $*"
  else
    echo "[run] $*"
    "$@"
  fi
}

cd "$ROOT"

# Python + IDE caches
find . -name __pycache__   -type d -prune -exec rm -rf {} + 2>/dev/null || true
find . -name '.ipynb_checkpoints' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find . -name '*.pyc' -delete 2>/dev/null || true

# SLURM cluster outputs
find . -name 'slurm-*.out' -delete 2>/dev/null || true
find . -name '*.log' -path '*/log/*' -delete 2>/dev/null || true
find . -name '*.err' -path '*/log/*' -delete 2>/dev/null || true

# Local secrets
[ -f .env ] && run rm -f .env

if [ "${WIPE_CHECKPOINTS:-0}" = "1" ]; then
  echo "[warn] WIPE_CHECKPOINTS=1 — deleting outputs/runs/*"
  find outputs/ -name 'checkpoint-*' -type d -prune -exec rm -rf {} + 2>/dev/null || true
  find outputs/ -name 'final_model'  -type d -prune -exec rm -rf {} + 2>/dev/null || true
fi

# The pre-refactor model_training/ tree has been replaced — delete it
# manually when you're sure the new pipeline works:
PARENT="$(dirname "$ROOT")"
if [ -d "$PARENT/model_training" ]; then
  cat <<EOF

NOTE: The pre-refactor model_training/ tree still exists at:
    $PARENT/model_training
This new agent-skills-training/ replaces it. Delete with:
    rm -rf $PARENT/model_training
(skipped here so you can review first; logs + final_models live there too.)
EOF
fi

echo "[done] cleanup complete."
