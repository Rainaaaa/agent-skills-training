#!/bin/bash
#
# Run the same Phase 1 / full-CPT recipe across multiple base models in
# parallel. Each model submits one SLURM job; outputs land under
#   <run.output_root>/<model.name>/<run.run_name>/
# so jobs never collide.
#
# Models pulled from model_training/model_path.json by logical name. The
# table below pins per-model resource hints (GPUs / mem / time / attn impl);
# unsupported architectures (multimodal MoE, research kernels, broken
# snapshots, gpt-oss without flash-attn, etc.) are intentionally excluded.
#
# Usage:
#   ./submit_models.sh                                   # default config + default model list
#   ./submit_models.sh path/to/config.yaml               # custom YAML
#   ./submit_models.sh - Qwen3-8B Foundation-Sec-8B-Reasoning  # default config + selected models

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LAUNCHER="${SCRIPT_DIR}/run_full_cpt.sh"

DEFAULT_CONFIG="${PHASE_ROOT}/configs/full_cpt_v1_quarter.yaml"
CONFIG="${1:-${DEFAULT_CONFIG}}"
[[ "${CONFIG}" == "-" ]] && CONFIG="${DEFAULT_CONFIG}"
shift || true

# Per-model overrides. Only these are known to work end-to-end with the
# Phase 1 recipe (LoRA on q_proj,v_proj + flash-attn 2 + 8-bit base + DDP).
# Skipped: gpt-oss-20b (needs eager attn + 4 GPUs), DeepSeek-V4-Flash
# (research kernels), Qwen3.6-35B-A3B (multimodal MoE), gemma-4-E4B
# (broken snapshot), llama3.1-8b (still downloading).
#
# Format:  model_name | gres | mem | time | attn_impl
KNOWN=(
  "Qwen3-8B                    | gpu:2 | 192G | 2-00:00:00 | flash_attention_2"
  "Foundation-Sec-8B-Reasoning | gpu:2 | 192G | 2-00:00:00 | flash_attention_2"
  "RedSage-Qwen3-8B-DPO        | gpu:2 | 192G | 2-00:00:00 | flash_attention_2"
  "WhiteRabbitNeo-2-8B         | gpu:2 | 192G | 2-00:00:00 | flash_attention_2"
)

# If model names were given on CLI, narrow to those; otherwise submit the lot.
SELECTED=("$@")
if (( ${#SELECTED[@]} > 0 )); then
  FILTERED=()
  for row in "${KNOWN[@]}"; do
    name="$(echo "${row%%|*}" | xargs)"
    for sel in "${SELECTED[@]}"; do
      if [[ "${name}" == "${sel}" ]]; then
        FILTERED+=("${row}"); break
      fi
    done
  done
  KNOWN=("${FILTERED[@]}")
fi

if (( ${#KNOWN[@]} == 0 )); then
  echo "No models selected. Available:" >&2
  echo "  Qwen3-8B  Foundation-Sec-8B-Reasoning  RedSage-Qwen3-8B-DPO  WhiteRabbitNeo-2-8B" >&2
  exit 2
fi

echo "Config: ${CONFIG}"
printf "%-30s %-8s %-6s %-12s %-22s %s\n" MODEL_NAME GRES MEM TIME ATTN JOBID

for row in "${KNOWN[@]}"; do
  IFS='|' read -r name gres mem walltime attn <<< "${row}"
  name="$(echo "${name}" | xargs)"
  gres="$(echo "${gres}" | xargs)"
  mem="$(echo "${mem}" | xargs)"
  walltime="$(echo "${walltime}" | xargs)"
  attn="$(echo "${attn}" | xargs)"

  jobid="$(sbatch --parsable \
    --job-name="cpt_$(echo "${name}" | tr '/.' '__')" \
    --gres="${gres}" \
    --mem="${mem}" \
    --time="${walltime}" \
    "${LAUNCHER}" "${CONFIG}" \
      "model.name=${name}" \
      "model.attn_implementation=${attn}")"

  printf "%-30s %-8s %-6s %-12s %-22s %s\n" "${name}" "${gres}" "${mem}" "${walltime}" "${attn}" "${jobid}"
done

cat <<EOF

Watch:    squeue -u \$USER
Logs:     ${PHASE_ROOT}/log/
Outputs:  ${PHASE_ROOT}/output/<MODEL_NAME>/$(grep -E "^\s*run_name:" "${CONFIG}" | head -1 | awk '{print $2}')/
EOF
