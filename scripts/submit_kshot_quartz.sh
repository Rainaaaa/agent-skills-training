#!/bin/bash
# Submit zero-shot AND k-shot inference jobs for a list of backbones, on
# both the challenge set (1444 rows) and the broad align_test set (capped
# to 5000 rows). Optionally use the CPT+HCL adapter stack instead of raw.
#
# Skips any (backbone, shots, testset) combo whose metrics file already
# exists, so this is safe to re-run.
#
# Usage:
#   bash scripts/submit_kshot_quartz.sh <K> <MODEL...>                       # raw baseline
#   bash scripts/submit_kshot_quartz.sh <K> --with-cpt-hcl <MODEL...>        # CPT-merged + HCL attached

set -euo pipefail

K="${1:?Usage: $0 <K> [--with-cpt-hcl] <MODEL...>}"; shift

ADAPTER_MODE=raw
if [[ "${1:-}" == "--with-cpt-hcl" ]]; then
  ADAPTER_MODE=cpt_hcl
  shift
fi

MODELS=("$@")
(( ${#MODELS[@]} > 0 )) || { echo "no models given" >&2; exit 64; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INFER_LAUNCHER="${SCRIPT_DIR}/run_infer_quartz.sh"
INFER_CFG=stages/downstream/configs/inference_misalignment_challenge.yaml
OUT_ROOT="${REPO_ROOT}/outputs/runs"

DATA_ROOT=/N/project/AdversarialModeling/datasets/agent_skills/misalignment
CHALLENGE_FILE="${DATA_ROOT}/sft/sft_align_challenge/align_challenge_test.parquet"
ALIGN_TEST_FILE="${DATA_ROOT}/sft/sft_align/align_test.parquet"
EXEMPLAR_FILE="${DATA_ROOT}/sft/sft_align/align_train.parquet"
MAX_ROWS_ALIGN=5000

# Cluster auto-detect.
HOST="$(hostname -s 2>/dev/null || hostname)"
case "${HOST}" in
  h[0-9]*|quartz*|*.quartz.*) PARTITION="hopper"; QOS="hopper" ;;
  bigred*|*.bigred*)          PARTITION="gpu";    QOS=""       ;;
  *) PARTITION="hopper"; QOS="hopper" ;;
esac
QOS_FLAG=()
[[ -n "${QOS}" ]] && QOS_FLAG=(--qos="${QOS}")

cd "${REPO_ROOT}"

# Run-name convention:
#   raw:     {zs|fsK}_misalign_raw[_aligntest]
#   cpt_hcl: {zs|fsK}_misalign_cpt_hcl[_aligntest]
# (matches the existing dirs from submit_downstream_quartz.sh)

submit_one() {
  local model="$1" shots="$2" testset="$3"
  local safe_model run_name jobname_prefix test_file marker
  safe_model="$(printf '%s' "${model}" | sed 's|/|__|g; s| |_|g')"

  if [[ "${shots}" -eq 0 ]]; then
    local stem="zs_misalign_${ADAPTER_MODE}"
  else
    local stem="fs${shots}_misalign_${ADAPTER_MODE}"
  fi
  if [[ "${testset}" == "challenge" ]]; then
    run_name="${stem}"
    test_file="${CHALLENGE_FILE}"
    local rows_arg=()
  else
    run_name="${stem}_aligntest"
    test_file="${ALIGN_TEST_FILE}"
    local rows_arg=("data.max_rows=${MAX_ROWS_ALIGN}")
  fi
  jobname_prefix="${run_name/misalign_/}"

  marker="${OUT_ROOT}/${safe_model}/infer_misalignment_detection__${run_name}/metrics_misalignment_detection.json"
  if [[ -e "${marker}" ]]; then
    printf "  [skip] %-30s %-40s (already complete)\n" "${model}" "${run_name}"
    return 0
  fi
  # Skip if a job for this output dir is already queued (prevents racing).
  if squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -qF "${jobname_prefix}_${model}"; then
    printf "  [skip] %-30s %-40s (already in queue)\n" "${model}" "${run_name}"
    return 0
  fi

  local fs_args=()
  if [[ "${shots}" -gt 0 ]]; then
    fs_args=("inference.n_shot=${shots}" "data.exemplar_file=${EXEMPLAR_FILE}")
  fi
  local adapter_args=()
  if [[ "${ADAPTER_MODE}" == "cpt_hcl" ]]; then
    local cpt="${OUT_ROOT}/${safe_model}/cpt_substage2/final_model"
    local hcl="${OUT_ROOT}/${safe_model}/hcl_substage2/final_model"
    if [[ ! -d "${cpt}" || ! -d "${hcl}" ]]; then
      printf "  [skip] %-30s %-40s (CPT/HCL adapter missing)\n" "${model}" "${run_name}"
      return 0
    fi
    adapter_args=(
      "model.phase1_adapter_path=${cpt}"
      "model.adapter_path=${hcl}"
    )
  fi
  # fs5+ are slow; fs0-2 fit comfortably in 2h.
  local walltime
  if [[ "${shots}" -ge 5 ]]; then walltime="--time=05:00:00"; else walltime="--time=02:00:00"; fi

  jobid=$(sbatch --parsable ${walltime} \
      -p "${PARTITION}" "${QOS_FLAG[@]}" \
      -J "${jobname_prefix}_${model}" \
      "${INFER_LAUNCHER}" "${INFER_CFG}" \
          "model.name=${model}" \
          "run.output_root=${OUT_ROOT}" \
          "run.run_name=${run_name}" \
          "data.test_file=${test_file}" \
          "data.task=misalignment_detection" \
          "${rows_arg[@]}" \
          "${fs_args[@]}" \
          "${adapter_args[@]}")
  printf "  %-30s %-40s %s jobid=%s\n" "${model}" "${run_name}" "${testset}" "${jobid}"
}

echo "[kshot] adapter_mode=${ADAPTER_MODE}  K=${K}  models=${MODELS[*]}"
for m in "${MODELS[@]}"; do
  for testset in challenge align_test; do
    submit_one "$m" 0 "$testset"   # zero-shot
    submit_one "$m" "$K" "$testset"   # k-shot
  done
done
