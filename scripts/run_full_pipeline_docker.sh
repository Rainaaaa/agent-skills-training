#!/usr/bin/env bash
# Full training pipeline for ONE backbone, end-to-end, via Docker.
# Mirrors the canonical recipe documented in KELLY_IT_HANDOFF.md §5, but
# does the bookkeeping automatically (run_name conventions, checkpoint
# discovery between substages, model-name sanitization) so that running
# additional backbones is a single CLI call — no YAML edits, no rebuilds.
#
# Stages run sequentially in this order:
#   1. CPT substage 1  (full_cpt/full_cpt_v3/stage1,  metadata @ 4096)
#   2. CPT substage 2  (full_cpt/full_cpt_v3/stage2,  all-files @ 10240,
#                       resumes from substage 1's latest checkpoint)
#   3. HCL substage 1  (pl_hcl/pl_hcl_v2/stage1,      metadata pairs @ 1024,
#                       chained on CPT substage 2's final_model adapter)
#   4. HCL substage 2  (pl_hcl/pl_hcl_v2/stage2,      all-files pairs @ 10240,
#                       resumes from HCL substage 1's latest checkpoint)
#   5. SFT misalignment — three ablation variants on sft/sft_align:
#        (a) base — raw <MODEL>,        no prior adapter merged
#        (b) cpt  — <MODEL> + CPT substage 2 adapter merged
#        (c) hcl  — <MODEL> + CPT substage 2 + HCL substage 2 adapters merged
#      Each gets its own output dir so they don't overwrite each other.
#
# Each stage runs as a separate `docker compose run --rm <service>` call.
# If any stage fails, the script aborts — later stages depend on earlier
# outputs (checkpoints, adapters) and would fail anyway.
#
# Usage:
#   ./scripts/run_full_pipeline_docker.sh                            # defaults to Foundation-Sec-8B-Reasoning
#   ./scripts/run_full_pipeline_docker.sh Qwen3-8B
#   ./scripts/run_full_pipeline_docker.sh llama3.1-8b
#
# Available backbones (keys in model_path.json):
#   Foundation-Sec-8B-Reasoning, Qwen3-8B, llama3.1-8b,
#   RedSage-Qwen3-8B-DPO, WhiteRabbitNeo-2-8B, gemma-4-E4B
#
# Required host-side env (sensible Jetstream defaults if unset, see
# docker-compose.yml; override in your .env on other hosts):
#   AGENTSKILLS_DATA_HOST    = host dir bind-mounted to /data inside the container
#   AGENTSKILLS_OUTPUTS_HOST = host dir bind-mounted to /app/outputs
#   HF_TOKEN                 = required for gated models (Llama, Gemma, ...)
#
# Output layout (on host, under ${AGENTSKILLS_OUTPUTS_HOST:-./outputs}):
#   outputs/runs/<safe_model_name>/
#       cpt_substage1/{checkpoint-*,final_model,...}
#       cpt_substage2/{checkpoint-*,final_model,...}
#       hcl_substage1/{checkpoint-*,final_model,...}
#       hcl_substage2/{checkpoint-*,final_model,...}
#       sft_misalignment_detection__sft_align_base/{...}
#       sft_misalignment_detection__sft_align_cpt/{...}
#       sft_misalignment_detection__sft_align_hcl/{...}

set -uo pipefail

MODEL="${1:-Foundation-Sec-8B-Reasoning}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Inside-container data dirs (must match the bind-mount in docker-compose.yml).
CPT_S1_DIR=/data/full_cpt/full_cpt_v3/stage1
CPT_S2_DIR=/data/full_cpt/full_cpt_v3/stage2
HCL_S1_DIR=/data/pl_hcl/pl_hcl_v2/stage1
HCL_S2_DIR=/data/pl_hcl/pl_hcl_v2/stage2
# Was sft_v2/ in older layouts; current upstream prep (commit 8d0b471
# "feat(data): split SFT into task-specific sft_mal/ and sft_align/") writes
# misalignment data here under sft_align/ as align_{train,val,test}.parquet.
SFT_ALIGN_DIR=/data/sft/sft_align

# Host-side output root — must match AGENTSKILLS_OUTPUTS_HOST in docker-compose.yml.
OUT_HOST="${AGENTSKILLS_OUTPUTS_HOST:-${REPO_ROOT}/outputs}"

# common/registry.py:sanitize_for_path replaces '/' with '__' and ' ' with '_'.
SAFE_MODEL="$(printf '%s' "${MODEL}" | sed 's|/|__|g; s| |_|g')"

# Run names (kept identical to KELLY_IT_HANDOFF.md so outputs are easy to find).
CPT_S1_RUN=cpt_substage1
CPT_S2_RUN=cpt_substage2
HCL_S1_RUN=hcl_substage1
HCL_S2_RUN=hcl_substage2
# Three SFT variants — each writes to its own output dir.
SFT_BASE_RUN=sft_align_base
SFT_CPT_RUN=sft_align_cpt
SFT_HCL_RUN=sft_align_hcl

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

step_header() {
  echo
  echo "============================================================"
  echo "[$(date '+%F %T')] MODEL=${MODEL}  STEP=${1}"
  echo "============================================================"
}

# Find the latest checkpoint dir in a host-side run output, return the
# corresponding container-side path. Returns nonzero if no checkpoint exists.
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
  echo "[$(date '+%F %T')] FAILED at: $1"
  echo "============================================================"
  exit 1
}

# ------------------------------------------------------------------
# 1/6 — CPT substage 1
# ------------------------------------------------------------------

step_header "1/6 CPT substage 1 (metadata @ 4096)"
docker compose run --rm pretraining \
    stages/pretraining/train.py \
    --config stages/pretraining/configs/full_cpt_v3_quarter.yaml \
    model.name="${MODEL}" \
    run.run_name="${CPT_S1_RUN}" \
    data.train_file="${CPT_S1_DIR}/train.parquet" \
    data.validation_file="${CPT_S1_DIR}/val.parquet" \
    data.test_file="${CPT_S1_DIR}/test.parquet" \
    data.max_seq_length=4096 \
  || die "CPT substage 1"

CPT_S1_CKPT="$(latest_checkpoint_in "${CPT_S1_RUN}")" \
  || die "no checkpoint written by CPT substage 1 — cannot resume into substage 2"

# ------------------------------------------------------------------
# 2/6 — CPT substage 2 (resumes from substage 1)
# ------------------------------------------------------------------

step_header "2/6 CPT substage 2 (all files @ 10240, resume from ${CPT_S1_CKPT})"
docker compose run --rm pretraining \
    stages/pretraining/train.py \
    --config stages/pretraining/configs/full_cpt_v3_quarter.yaml \
    model.name="${MODEL}" \
    run.run_name="${CPT_S2_RUN}" \
    run.resume_from_checkpoint="${CPT_S1_CKPT}" \
    data.train_file="${CPT_S2_DIR}/train.parquet" \
    data.validation_file="${CPT_S2_DIR}/val.parquet" \
    data.test_file="${CPT_S2_DIR}/test.parquet" \
    data.max_seq_length=10240 \
  || die "CPT substage 2"

CPT_FINAL_ADAPTER="/app/outputs/runs/${SAFE_MODEL}/${CPT_S2_RUN}/final_model"
if [[ ! -d "${OUT_HOST}/runs/${SAFE_MODEL}/${CPT_S2_RUN}/final_model" ]]; then
  die "CPT substage 2 ran but final_model/ is missing — cannot start HCL"
fi

# ------------------------------------------------------------------
# 3/6 — HCL substage 1 (chained on CPT substage 2's adapter)
# ------------------------------------------------------------------

step_header "3/6 HCL substage 1 (metadata pairs @ 1024, chained on ${CPT_FINAL_ADAPTER})"
docker compose run --rm hcl \
    stages/hcl/train.py \
    --config stages/hcl/configs/lp_hcl_v2_quarter.yaml \
    model.name="${MODEL}" \
    run.run_name="${HCL_S1_RUN}" \
    model.phase1_adapter_path="${CPT_FINAL_ADAPTER}" \
    data.train_file="${HCL_S1_DIR}/pairs_train.parquet" \
    data.validation_file="${HCL_S1_DIR}/pairs_val.parquet" \
    data.test_file="${HCL_S1_DIR}/pairs_test.parquet" \
    data.max_seq_length=1024 \
  || die "HCL substage 1"

HCL_S1_CKPT="$(latest_checkpoint_in "${HCL_S1_RUN}")" \
  || die "no checkpoint written by HCL substage 1 — cannot resume into substage 2"

# ------------------------------------------------------------------
# 4/6 — HCL substage 2 (resumes from substage 1)
# ------------------------------------------------------------------

step_header "4/6 HCL substage 2 (all-files pairs @ 10240, resume from ${HCL_S1_CKPT})"
docker compose run --rm hcl \
    stages/hcl/train.py \
    --config stages/hcl/configs/lp_hcl_v2_quarter.yaml \
    model.name="${MODEL}" \
    run.run_name="${HCL_S2_RUN}" \
    run.resume_from_checkpoint="${HCL_S1_CKPT}" \
    data.train_file="${HCL_S2_DIR}/pairs_train.parquet" \
    data.validation_file="${HCL_S2_DIR}/pairs_val.parquet" \
    data.test_file="${HCL_S2_DIR}/pairs_test.parquet" \
    data.max_seq_length=10240 \
  || die "HCL substage 2"

# ------------------------------------------------------------------
# 5a/5 — SFT misalignment on the RAW base (no prior adapter)
# ------------------------------------------------------------------
#
# This is the baseline: it shows what SFT can do on its own, without any
# CPT or HCL preconditioning. Used to measure the marginal lift from the
# upstream stages in 5b/5c.

step_header "5a/5 SFT misalignment — base (raw ${MODEL})"
docker compose run --rm sft-align \
    stages/downstream/sft/train.py \
    --config stages/downstream/configs/sft_misalignment_example.yaml \
    model.name="${MODEL}" \
    run.run_name="${SFT_BASE_RUN}" \
    model.phase1_adapter_path= \
    model.phase2_adapter_path= \
    data.train_file="${SFT_ALIGN_DIR}/align_train.parquet" \
    data.validation_file="${SFT_ALIGN_DIR}/align_val.parquet" \
    data.test_file="${SFT_ALIGN_DIR}/align_test.parquet" \
  || die "SFT misalignment (base)"

# ------------------------------------------------------------------
# 5b/5 — SFT misalignment chained on CPT substage 2's adapter
# ------------------------------------------------------------------
#
# The SFT trainer merges CPT_FINAL_ADAPTER into the base before applying
# its own fresh LoRA. Tests whether the CPT continuation alone helps the
# downstream misalignment classifier.

HCL_FINAL_ADAPTER="/app/outputs/runs/${SAFE_MODEL}/${HCL_S2_RUN}/final_model"
if [[ ! -d "${OUT_HOST}/runs/${SAFE_MODEL}/${HCL_S2_RUN}/final_model" ]]; then
  die "HCL substage 2 ran but final_model/ is missing — cannot start SFT (hcl variant)"
fi

step_header "5b/5 SFT misalignment — CPT (${MODEL} + CPT substage 2 adapter)"
docker compose run --rm sft-align \
    stages/downstream/sft/train.py \
    --config stages/downstream/configs/sft_misalignment_example.yaml \
    model.name="${MODEL}" \
    run.run_name="${SFT_CPT_RUN}" \
    model.phase1_adapter_path="${CPT_FINAL_ADAPTER}" \
    model.phase2_adapter_path= \
    data.train_file="${SFT_ALIGN_DIR}/align_train.parquet" \
    data.validation_file="${SFT_ALIGN_DIR}/align_val.parquet" \
    data.test_file="${SFT_ALIGN_DIR}/align_test.parquet" \
  || die "SFT misalignment (cpt)"

# ------------------------------------------------------------------
# 5c/5 — SFT misalignment chained on CPT + HCL substage 2 adapters
# ------------------------------------------------------------------
#
# Both prior adapters are merged in the order they were trained (CPT first,
# then HCL on top of CPT-merged base). This is the full curriculum: tests
# whether HCL on top of CPT adds value over just CPT (compare to 5b).

step_header "5c/5 SFT misalignment — HCL (${MODEL} + CPT + HCL substage 2 adapters)"
docker compose run --rm sft-align \
    stages/downstream/sft/train.py \
    --config stages/downstream/configs/sft_misalignment_example.yaml \
    model.name="${MODEL}" \
    run.run_name="${SFT_HCL_RUN}" \
    model.phase1_adapter_path="${CPT_FINAL_ADAPTER}" \
    model.phase2_adapter_path="${HCL_FINAL_ADAPTER}" \
    data.train_file="${SFT_ALIGN_DIR}/align_train.parquet" \
    data.validation_file="${SFT_ALIGN_DIR}/align_val.parquet" \
    data.test_file="${SFT_ALIGN_DIR}/align_test.parquet" \
  || die "SFT misalignment (hcl)"

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------

echo
echo "============================================================"
echo "[$(date '+%F %T')] DONE — full pipeline completed for ${MODEL}"
echo "Outputs under ${OUT_HOST}/runs/${SAFE_MODEL}/"
echo "============================================================"
