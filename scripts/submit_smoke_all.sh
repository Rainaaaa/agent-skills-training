#!/bin/bash
#
# Submit a Stage-1 pretraining smoke test for every model in model_path.json,
# auto-selecting the right Slurm partition for the cluster we're on:
#   - Quartz       -> -p hopper  (H100 80GB)
#   - BigRed200    -> -p gpu     (A100 40GB)
#
# Each model gets its own job; failures don't block siblings.
#
# Usage:
#   ./submit_smoke_all.sh                # all models from model_path.json
#   ./submit_smoke_all.sh Qwen3-8B ...   # subset
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LAUNCHER="${SCRIPT_DIR}/run_smoke.sh"
REGISTRY="${REPO_ROOT}/model_path.json"

# ----- auto-detect cluster -> partition + QOS -----
# Quartz hopper partition's AllowQos=hopper, so the job needs --qos=hopper.
# BR200 gpu partition uses the default (allocated) QOS — no override needed.
HOST="$(hostname -s 2>/dev/null || hostname)"
case "${HOST}" in
  h[0-9]*|quartz*|*.quartz.*)   PARTITION="hopper" ; QOS="hopper" ; CLUSTER="Quartz"    ;;
  bigred*|h2.bigred*|*.bigred*) PARTITION="gpu"    ; QOS=""       ; CLUSTER="BigRed200" ;;
  *)
    if sinfo -p hopper >/dev/null 2>&1 && sinfo -p hopper -h 2>/dev/null | grep -q .; then
      PARTITION="hopper" ; QOS="hopper" ; CLUSTER="Quartz(hopper detected)"
    else
      PARTITION="gpu" ; QOS="" ; CLUSTER="unknown(falling back to gpu)"
    fi
    ;;
esac
echo "[submit] host=${HOST} -> cluster=${CLUSTER} partition=${PARTITION} qos=${QOS:-<default>}"

# ----- model list -----
if (( $# > 0 )); then
  MODELS=("$@")
else
  # Pull every model name from model_path.json. Pure python = no jq dependency.
  mapfile -t MODELS < <(python3 - "$REGISTRY" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    reg = json.load(f)
for cat, entries in reg.items():
    if isinstance(entries, dict) and not ({"local", "hf"} & set(entries.keys())):
        for name in entries:
            print(name)
    else:
        print(cat)
PY
)
fi

if (( ${#MODELS[@]} == 0 )); then
  echo "[submit] no models to submit" >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/log"

QOS_FLAG=()
[[ -n "${QOS}" ]] && QOS_FLAG=(--qos="${QOS}")

printf "%-32s %-10s %-8s %s\n" MODEL PARTITION QOS JOBID
for m in "${MODELS[@]}"; do
  safe="$(echo "${m}" | tr '/.' '__')"
  jobid="$(sbatch --parsable \
      -p "${PARTITION}" \
      "${QOS_FLAG[@]}" \
      -J "smoke_${safe}" \
      "${LAUNCHER}" "${m}")"
  printf "%-32s %-10s %-8s %s\n" "${m}" "${PARTITION}" "${QOS:-default}" "${jobid}"
done

cat <<EOF

Watch:    squeue -u \$USER
Logs:     ${REPO_ROOT}/log/smoke_<MODEL>_<JOBID>.{log,err}
Outputs:  ${REPO_ROOT}/outputs/runs/<MODEL>/smoke_pretrain/
EOF
