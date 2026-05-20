#!/usr/bin/env bash
# Smoke test for run_full_pipeline_docker.sh — same chain, tiny caps.
#
# Runs the full CPT 1→2, HCL 1→2, and three SFT-misalignment variants
# (base / CPT-chained / HCL-chained) with very small step counts, row caps,
# and short sequence lengths so the whole pipeline finishes in ~25-40 min
# on a single A100, not days. The point is to validate that the pipeline
# wiring works end-to-end — checkpoint discovery between substages, the
# CPT→HCL adapter handoff, and (critically) the new SFT phase-adapter
# merge logic added in stages/downstream/sft/train.py.
#
# Output run names all carry a `_smoke` suffix so they live in separate
# dirs from production runs, e.g.:
#   outputs/runs/<model>/cpt_substage1_smoke/
#   outputs/runs/<model>/cpt_substage2_smoke/
#   outputs/runs/<model>/hcl_substage1_smoke/
#   outputs/runs/<model>/hcl_substage2_smoke/
#   outputs/runs/<model>/sft_misalignment_detection__sft_align_base_smoke/
#   outputs/runs/<model>/sft_misalignment_detection__sft_align_cpt_smoke/
#   outputs/runs/<model>/sft_misalignment_detection__sft_align_hcl_smoke/
#
# That way you can keep real-training outputs (cpt_substage1/, etc.) and
# delete the smoke ones in a single rm without touching anything important.
#
# Usage:
#   ./scripts/run_full_pipeline_smoke_docker.sh                 # default model
#   ./scripts/run_full_pipeline_smoke_docker.sh Qwen3-8B
#
# Smoke caps applied per stage (all set via CLI override, configs unchanged):
#   training.max_steps        = 4
#   data.max_train_rows       = 64
#   data.max_validation_rows  = 8   (CPT/SFT) or 16 (HCL — pairs are smaller)
#   data.max_test_rows        = 8
#   data.max_seq_length       = 512 (CPT1) / 1024 (CPT2/HCL2) / 256 (HCL1) / 512 (SFT)
#
# A failed stage aborts the script. Each stage logs a header so it's easy
# to tell which one died from the console output. Re-running picks up
# from scratch (smoke runs are not auto-resume-aware).

set -uo pipefail

MODEL="${1:-Foundation-Sec-8B-Reasoning}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Inside-container data dirs (must match the bind-mount in docker-compose.yml).
CPT_S1_DIR=/data/full_cpt/full_cpt_v3/stage1
CPT_S2_DIR=/data/full_cpt/full_cpt_v3/stage2
HCL_S1_DIR=/data/pl_hcl/pl_hcl_v2/stage1
HCL_S2_DIR=/data/pl_hcl/pl_hcl_v2/stage2
# NOTE: sft_align/, not sft_v2/. The data was refactored upstream (commit
# 8d0b471) — the new layout has task-specific dirs with renamed files:
#   sft_align/align_{train,val,test}.parquet
#   sft_mal/mal_{train,val,test}.parquet
SFT_ALIGN_DIR=/data/sft/sft_align

OUT_HOST="${AGENTSKILLS_OUTPUTS_HOST:-${REPO_ROOT}/outputs}"
SAFE_MODEL="$(printf '%s' "${MODEL}" | sed 's|/|__|g; s| |_|g')"

# Smoke run names (NOT the production names; suffixed with _smoke so they
# don't collide with real runs).
CPT_S1_RUN=cpt_substage1_smoke
CPT_S2_RUN=cpt_substage2_smoke
HCL_S1_RUN=hcl_substage1_smoke
HCL_S2_RUN=hcl_substage2_smoke
SFT_BASE_RUN=sft_align_base_smoke
SFT_CPT_RUN=sft_align_cpt_smoke
SFT_HCL_RUN=sft_align_hcl_smoke

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

step_header() {
  echo
  echo "============================================================"
  echo "[$(date '+%F %T')] SMOKE  MODEL=${MODEL}  STEP=${1}"
  echo "============================================================"
}

latest_checkpoint_in() {
  local run_name="$1"
  local host_run_dir="${OUT_HOST}/runs/${SAFE_MODEL}/${run_name}"
  local latest
  latest="$(ls -d "${host_run_dir}"/checkpoint-* 2>/dev/null | sort -V | tail -1)"
  if [[ -z "${latest}" ]]; then
    return 1
  fi
  printf '/app/outputs/runs/%s/%s/%s' "${SAFE_MODEL}" "${run_name}" "$(basename "${latest}")"
}

die() {
  echo
  echo "============================================================"
  echo "[$(date '+%F %T')] SMOKE FAILED at: $1"
  echo "============================================================"
  exit 1
}

T0=$(date +%s)

# ------------------------------------------------------------------
# 1/7 — CPT substage 1 (smoke)
# ------------------------------------------------------------------

step_header "1/7 CPT substage 1 (smoke: 4 steps, 64 train rows, seq=512)"
docker compose run --rm pretraining \
    stages/pretraining/train.py \
    --config stages/pretraining/configs/full_cpt_v3_quarter.yaml \
    model.name="${MODEL}" \
    run.run_name="${CPT_S1_RUN}" \
    training.max_steps=4 \
    data.max_train_rows=64 \
    data.max_validation_rows=8 \
    data.max_test_rows=8 \
    data.train_file="${CPT_S1_DIR}/train.parquet" \
    data.validation_file="${CPT_S1_DIR}/val.parquet" \
    data.test_file="${CPT_S1_DIR}/test.parquet" \
    data.max_seq_length=512 \
  || die "CPT substage 1"

CPT_S1_CKPT="$(latest_checkpoint_in "${CPT_S1_RUN}")" \
  || die "no checkpoint from CPT substage 1 — cannot resume into substage 2"

# ------------------------------------------------------------------
# 2/7 — CPT substage 2 (smoke, resumes from substage 1)
# ------------------------------------------------------------------

step_header "2/7 CPT substage 2 (smoke: 4 steps, seq=1024, resume ${CPT_S1_CKPT})"
docker compose run --rm pretraining \
    stages/pretraining/train.py \
    --config stages/pretraining/configs/full_cpt_v3_quarter.yaml \
    model.name="${MODEL}" \
    run.run_name="${CPT_S2_RUN}" \
    run.resume_from_checkpoint="${CPT_S1_CKPT}" \
    training.max_steps=4 \
    data.max_train_rows=64 \
    data.max_validation_rows=8 \
    data.max_test_rows=8 \
    data.train_file="${CPT_S2_DIR}/train.parquet" \
    data.validation_file="${CPT_S2_DIR}/val.parquet" \
    data.test_file="${CPT_S2_DIR}/test.parquet" \
    data.max_seq_length=1024 \
  || die "CPT substage 2"

CPT_FINAL_ADAPTER="/app/outputs/runs/${SAFE_MODEL}/${CPT_S2_RUN}/final_model"
if [[ ! -d "${OUT_HOST}/runs/${SAFE_MODEL}/${CPT_S2_RUN}/final_model" ]]; then
  die "CPT substage 2 ran but final_model/ is missing — cannot start HCL"
fi

# ------------------------------------------------------------------
# 3/7 — HCL substage 1 (smoke, chained on CPT substage 2's adapter)
# ------------------------------------------------------------------

step_header "3/7 HCL substage 1 (smoke: 4 steps, seq=256, chained on CPT adapter)"
docker compose run --rm hcl \
    stages/hcl/train.py \
    --config stages/hcl/configs/lp_hcl_v2_quarter.yaml \
    model.name="${MODEL}" \
    run.run_name="${HCL_S1_RUN}" \
    model.phase1_adapter_path="${CPT_FINAL_ADAPTER}" \
    training.max_steps=4 \
    data.max_train_rows=64 \
    data.max_validation_rows=16 \
    data.max_test_rows=16 \
    data.train_file="${HCL_S1_DIR}/pairs_train.parquet" \
    data.validation_file="${HCL_S1_DIR}/pairs_val.parquet" \
    data.test_file="${HCL_S1_DIR}/pairs_test.parquet" \
    data.max_seq_length=256 \
  || die "HCL substage 1"

HCL_S1_CKPT="$(latest_checkpoint_in "${HCL_S1_RUN}")" \
  || die "no checkpoint from HCL substage 1 — cannot resume into substage 2"

# ------------------------------------------------------------------
# 4/7 — HCL substage 2 (smoke, resumes from substage 1)
# ------------------------------------------------------------------

step_header "4/7 HCL substage 2 (smoke: 4 steps, seq=1024, resume ${HCL_S1_CKPT})"
docker compose run --rm hcl \
    stages/hcl/train.py \
    --config stages/hcl/configs/lp_hcl_v2_quarter.yaml \
    model.name="${MODEL}" \
    run.run_name="${HCL_S2_RUN}" \
    run.resume_from_checkpoint="${HCL_S1_CKPT}" \
    training.max_steps=4 \
    data.max_train_rows=64 \
    data.max_validation_rows=16 \
    data.max_test_rows=16 \
    data.train_file="${HCL_S2_DIR}/pairs_train.parquet" \
    data.validation_file="${HCL_S2_DIR}/pairs_val.parquet" \
    data.test_file="${HCL_S2_DIR}/pairs_test.parquet" \
    data.max_seq_length=1024 \
  || die "HCL substage 2"

HCL_FINAL_ADAPTER="/app/outputs/runs/${SAFE_MODEL}/${HCL_S2_RUN}/final_model"
if [[ ! -d "${OUT_HOST}/runs/${SAFE_MODEL}/${HCL_S2_RUN}/final_model" ]]; then
  die "HCL substage 2 ran but final_model/ is missing — cannot start SFT (hcl variant)"
fi

# ------------------------------------------------------------------
# 5a/7 — SFT misalignment on the RAW base (smoke)
# ------------------------------------------------------------------

step_header "5a/7 SFT misalignment — base (smoke)"
docker compose run --rm sft-align \
    stages/downstream/sft/train.py \
    --config stages/downstream/configs/sft_misalignment_example.yaml \
    model.name="${MODEL}" \
    run.run_name="${SFT_BASE_RUN}" \
    model.phase1_adapter_path= \
    model.phase2_adapter_path= \
    training.max_steps=4 \
    data.max_train_rows=64 \
    data.max_validation_rows=8 \
    data.max_test_rows=8 \
    data.max_seq_length=512 \
    data.train_file="${SFT_ALIGN_DIR}/align_train.parquet" \
    data.validation_file="${SFT_ALIGN_DIR}/align_val.parquet" \
    data.test_file="${SFT_ALIGN_DIR}/align_test.parquet" \
  || die "SFT misalignment (base)"

# ------------------------------------------------------------------
# 5b/7 — SFT misalignment chained on CPT adapter (smoke)
# ------------------------------------------------------------------

step_header "5b/7 SFT misalignment — CPT (smoke)"
docker compose run --rm sft-align \
    stages/downstream/sft/train.py \
    --config stages/downstream/configs/sft_misalignment_example.yaml \
    model.name="${MODEL}" \
    run.run_name="${SFT_CPT_RUN}" \
    model.phase1_adapter_path="${CPT_FINAL_ADAPTER}" \
    model.phase2_adapter_path= \
    training.max_steps=4 \
    data.max_train_rows=64 \
    data.max_validation_rows=8 \
    data.max_test_rows=8 \
    data.max_seq_length=512 \
    data.train_file="${SFT_ALIGN_DIR}/align_train.parquet" \
    data.validation_file="${SFT_ALIGN_DIR}/align_val.parquet" \
    data.test_file="${SFT_ALIGN_DIR}/align_test.parquet" \
  || die "SFT misalignment (cpt)"

# ------------------------------------------------------------------
# 5c/7 — SFT misalignment chained on CPT + HCL adapters (smoke)
# ------------------------------------------------------------------

step_header "5c/7 SFT misalignment — HCL (smoke)"
docker compose run --rm sft-align \
    stages/downstream/sft/train.py \
    --config stages/downstream/configs/sft_misalignment_example.yaml \
    model.name="${MODEL}" \
    run.run_name="${SFT_HCL_RUN}" \
    model.phase1_adapter_path="${CPT_FINAL_ADAPTER}" \
    model.phase2_adapter_path="${HCL_FINAL_ADAPTER}" \
    training.max_steps=4 \
    data.max_train_rows=64 \
    data.max_validation_rows=8 \
    data.max_test_rows=8 \
    data.max_seq_length=512 \
    data.train_file="${SFT_ALIGN_DIR}/align_train.parquet" \
    data.validation_file="${SFT_ALIGN_DIR}/align_val.parquet" \
    data.test_file="${SFT_ALIGN_DIR}/align_test.parquet" \
  || die "SFT misalignment (hcl)"

# ------------------------------------------------------------------
# Done — sanity check that each stage wrote a final_model/
# ------------------------------------------------------------------

T1=$(date +%s)
ELAPSED=$(( T1 - T0 ))

echo
echo "============================================================"
echo "[$(date '+%F %T')] SMOKE PIPELINE — verifying outputs"
echo "============================================================"

missing=0
for run in "${CPT_S1_RUN}" "${CPT_S2_RUN}" "${HCL_S1_RUN}" "${HCL_S2_RUN}"; do
  fm="${OUT_HOST}/runs/${SAFE_MODEL}/${run}/final_model"
  if [[ -d "${fm}" ]]; then
    echo "  OK   ${fm}"
  else
    echo "  MISS ${fm}"
    missing=$(( missing + 1 ))
  fi
done
for run in "${SFT_BASE_RUN}" "${SFT_CPT_RUN}" "${SFT_HCL_RUN}"; do
  fm="${OUT_HOST}/runs/${SAFE_MODEL}/sft_misalignment_detection__${run}/final_model"
  if [[ -d "${fm}" ]]; then
    echo "  OK   ${fm}"
  else
    echo "  MISS ${fm}"
    missing=$(( missing + 1 ))
  fi
done

echo "============================================================"
if [[ "${missing}" -eq 0 ]]; then
  echo "[$(date '+%F %T')] SMOKE PASSED — all 7 stages wrote final_model/  (${ELAPSED}s)"
  exit 0
else
  echo "[$(date '+%F %T')] SMOKE FAILED — ${missing} stage(s) missing final_model/  (${ELAPSED}s)"
  exit 1
fi
