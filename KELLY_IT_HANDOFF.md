# Kelly IT handoff — agent-skills-training on a GPU host

This document describes everything Kelly IT needs to bring up the
`agent-skills-training` pipeline on a fresh GPU host. The procedure was
validated end-to-end on a Jetstream NVIDIA A100 40 GB on 2026-05-12 against
the legacy pre-refactor dataset, and a Foundation-Sec-8B-Reasoning backbone
pulled from HuggingFace.

---

## 1. Host requirements

| Component | Minimum | Notes |
|---|---|---|
| GPU | NVIDIA, ≥24 GB VRAM for 8B LoRA bf16 | A100 40 GB tested. H100/L40/RTX 6000 Ada all fine. |
| NVIDIA driver | 535+ (CUDA 12.1 runtime) | The image bundles CUDA libs; only the driver needs to be on the host. |
| `nvidia-container-toolkit` | Any recent | Required for `--gpus all`. Verify: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`. |
| Docker | 24+ with Compose v2 | Tested on Docker 29.1.3 + Compose 2.40.3. |
| Disk | ~80 GB free | Image ~12 GB, HF cache ~16 GB per 8B model, datasets ~25 GB legacy, outputs grow with checkpoint count. |
| RAM | ≥32 GB | Tokenization is the spiky part. |
| Network | Outbound HTTPS to `huggingface.co` | Only needed for the one-time prefetch. After that, all training is offline. |

---

## 2. Layout on the host

```
/opt/agentskills/                              # or wherever you check the repos out
├── agent-skills-training/                     # this repo
└── data/
    ├── full_cpt/full_cpt_v3/                  # Stage 1 (CPT) — challenge-scrubbed
    │   ├── stage1/{train,val,test,unseen}.parquet
    │   └── stage2/{train,val,test,unseen}.parquet
    ├── pl_hcl/pl_hcl_v2/                      # Stage 2 (HCL) — challenge-scrubbed
    │   ├── stage1/pairs_{train,val,test,unseen}.parquet    # JOINED pair files
    │   └── stage2/pairs_{train,val,test,unseen}.parquet
    └── sft/                                   # Stage 3 (SFT) — per-task layout
        ├── sft_mal/                           #   malicious-detection task
        │   ├── mal_{train,val,test}.parquet   #     label = overall_class (safe/malicious)
        │   ├── manifest.json                  #     schema + sha256 + source provenance
        │   └── stats.json                     #     per-split label distribution + char counts
        └── sft_align/                         #   misalignment-detection task
            ├── align_{train,val,test}.parquet #     label = alignment_class (ALIGNED/MISALIGNED)
            ├── manifest.json
            └── stats.json
```

> **HCL path note**: `stages/hcl/data.py` needs `anchor_text` + `pair_text` +
> `pair_kind` columns side-by-side. Those live in `pairs_*.parquet` only
> (built by agent-skills-preparation/pipeline/build_hcl_pairs.py). The
> sibling `{train,val,test}.parquet` and `*_t{1,2,3a,3b,3c}.parquet`
> files in the same dirs are SINGLE-VIEW intermediates — do **not** point
> the HCL trainer at them, it will fail with a column-not-found error.

**Two-substage continual training (CPT + HCL).** Both `full_cpt_v3/` and
`pl_hcl/pl_hcl_v2/` are split on disk into `stage1/` and `stage2/`:

- `stage1/` — metadata + instructions only. Train at `max_seq_length=4096`.
- `stage2/` — all files (incl. full code/content). **Continual-train**
  from sub-stage 1 at `max_seq_length=10240` — sub-stage 2 resumes from
  sub-stage 1's last checkpoint via `run.resume_from_checkpoint`, not a
  fresh base. See §5 for the run-it-twice commands.

`sft/sft_v2/` and `sft/challenge/` have no sub-stages; SFT is a single train/val/test pass.

The compose file expects exactly **two** host paths set via env (or `.env`):

- `AGENTSKILLS_DATA_HOST` — bind-mounted RO into the container at `/data`
- `AGENTSKILLS_OUTPUTS_HOST` — bind-mounted RW into the container at
  `/app/outputs` (checkpoints + HF cache + TensorBoard)

```bash
# .env in agent-skills-training/
AGENTSKILLS_DATA_HOST=/opt/agentskills/data
AGENTSKILLS_OUTPUTS_HOST=/opt/agentskills/training_outputs
AGENTSKILLS_MODELS_HOST=/opt/agentskills/hf_cache       # OPTIONAL; defaults to outputs/hf_cache
```

---

## 3. Get the image (two options)

### Option A — Load the prebuilt image (fastest, no build needed)

A prebuilt `agent-skills-training:latest` is shared as a 7.1 GB gzipped tarball:

```bash
# After downloading the tarball:
sha256sum agent-skills-training_2026-05-16.tar.gz   # verify against the .sha256 file
docker load -i agent-skills-training_2026-05-16.tar.gz
docker images agent-skills-training                  # should show :latest
```

### Option B — Build from source

```bash
cd /opt/agentskills/agent-skills-training
docker compose build pretraining     # ~10 min, ~12 GB image
```

### Warm the HF cache (one-time, either option)

```bash
docker compose run --rm --no-deps --entrypoint python pretraining \
    -m common.prefetch_models        # ~20 min on a fast link; one-time
```

`prefetch_models` walks every entry in `model_path.json` and downloads the
HF repo for any whose `local` path doesn't exist on this host. Resume-safe
(re-runs are no-ops once cached). To prefetch only specific models:

```bash
docker compose run --rm --no-deps --entrypoint python pretraining \
    -m common.prefetch_models Foundation-Sec-8B-Reasoning Qwen3-8B
```

---

## 4. Quick smoke (verify everything works in ~5 min)

The recommended way is the one-liner that runs **all five stages** end-to-end
(CPT → HCL → SFT misalignment → inference → eval_baseline) with smoke caps
and prints a pass/fail summary at the end. No need to remember individual
overrides:

```bash
bash scripts/run_pipeline_smoke_docker.sh                       # default backbone: Foundation-Sec-8B-Reasoning
bash scripts/run_pipeline_smoke_docker.sh llama3.1-8b           # or any model from model_path.json
```

Expected output on success (~5 min on a single A100 40 GB):

```
[pipe] SUMMARY  model=llama3.1-8b
  1_cpt              PASS
  2_hcl              PASS
  3_sft_align        PASS
  4_inference        PASS
  5_eval_baseline    PASS
```

If any step fails, the summary lists `FAIL(exit=N)` against that step and
the script exits non-zero. Per-step logs live under
`outputs/runs/<MODEL>/<step_run_name>/train.log`. The script is **idempotent**
— rerunning skips steps whose marker file already exists, so you can fix
one failure and just rerun.

Per-stage manual smoke commands (only if you need to bisect a failing step):

```bash
# Stage 1 — CPT
docker compose run --rm pretraining \
    stages/pretraining/train.py \
    --config stages/pretraining/configs/example.yaml \
    run.run_name=smoke_pretrain \
    training.max_steps=4 \
    data.max_train_rows=64 data.max_validation_rows=8 data.max_test_rows=8 \
    data.max_seq_length=1024

# Stage 2 — HCL (chains on Stage 1's adapter)
docker compose run --rm hcl \
    stages/hcl/train.py \
    --config stages/hcl/configs/example.yaml \
    run.run_name=smoke_hcl \
    model.phase1_adapter_path=/app/outputs/runs/Foundation-Sec-8B-Reasoning/smoke_pretrain/final_model \
    training.max_steps=4 \
    data.max_train_rows=64 data.max_validation_rows=16 data.max_test_rows=16 \
    data.max_seq_length=512

# Stage 3 — SFT misalignment
docker compose run --rm sft-align \
    stages/downstream/sft/train.py \
    --config stages/downstream/configs/sft_misalignment_example.yaml \
    run.run_name=smoke_sft_align \
    training.max_steps=4 \
    data.max_seq_length=512
```

Validated on Jetstream A100 40 GB (2026-05-16) end-to-end for both
`Foundation-Sec-8B-Reasoning` and `llama3.1-8b` — all 5/5 steps PASS,
total wall ~5 min per backbone.

### 4a. Full-pipeline Slurm smoke (Quartz / BR200, no Docker)

For IU's Slurm clusters where Docker isn't available, `scripts/run_pipeline_smoke.sh`
runs the same five stages back-to-back in one Slurm job against the
conda env at `/N/slate/cz1/conda/envs/AgentSkillsOSS` (sourced from
`/N/slate/cz1/miniconda3`). The companion launcher
`scripts/submit_smoke_all.sh` auto-detects cluster ↔ partition:

| Cluster   | Partition | QOS      | Auto-detected when hostname matches |
|-----------|-----------|----------|--------------------------------------|
| Quartz    | hopper    | hopper   | `h[0-9]*` / `*.quartz.*` (H100 80GB) |
| BigRed200 | gpu       | default  | `bigred*` / `*.bigred*` (A100 40GB)  |

```bash
# Stage-1-only smoke for every backbone in model_path.json (parallel jobs):
bash scripts/submit_smoke_all.sh

# Or a subset:
bash scripts/submit_smoke_all.sh Foundation-Sec-8B-Reasoning Qwen3-8B

# Full pipeline (CPT -> HCL -> SFT-align -> inference -> eval_baseline)
# for ONE backbone, one Slurm job, per-step pass/fail summary at the end:
sbatch -p hopper --qos=hopper scripts/run_pipeline_smoke.sh Foundation-Sec-8B-Reasoning   # Quartz
sbatch -p gpu                  scripts/run_pipeline_smoke.sh Foundation-Sec-8B-Reasoning   # BR200
```

What the full-pipeline smoke does (defaults under `scripts/run_pipeline_smoke.sh`):

1. **Stage 1 — CPT** on `full_cpt/full_cpt_v3/stage1/{train,val,test}.parquet`
   (challenge-scrubbed v3). 4 steps × 64 train rows.
2. **Stage 2 — HCL** on `pl_hcl/pl_hcl_v2/stage1/pairs_{train,val,test}.parquet`,
   chained on Stage 1's LoRA. 20 steps × 256 train rows, **stratified eval
   by `pair_kind`** (33 positive + 33 corrupted + 33 swapped) so the eval
   isn't dominated by the natural 88%-negative skew of the pair file.
   `ce_loss_weight=0.5` so the contrastive head moves past random init.
   Emits a per-pair_kind breakdown — see "HCL breakdown" below.
3. **Stage 3 — SFT misalignment**, trained on `sft/sft_align/align_*.parquet`
   (label = `alignment_class`), chained on Stage 1's LoRA. The parallel
   `sft/sft_mal/mal_*.parquet` (label = `overall_class`) is used by the
   `sft-mal` service when you want to train the malicious-detection task.
4. **Stage 3 — Inference** on `sft/sft_align/align_test.parquet` using
   the SFT-misalignment adapter from step 3.
5. **Stage 3 — eval_baseline** intrinsic CLM perplexity on
   `full_cpt/full_cpt_v3/stage1/test.parquet`.

Per-step markers (`final_model/` for trainers, `metrics_<task>.json` for
inference, `baseline_metrics.json` for eval) gate skip-on-rerun, so a
re-submit after a code change retries only the failed/new steps.

Logs land in `log/pipe_smoke_<MODEL>_<JOBID>.{log,err}`; outputs in
`outputs/runs/<MODEL>/<run_name>/`.

**HCL breakdown — for tuning `model.hcl.pair_kind_weights`.**
After the final test eval, `stages/hcl/train.py::evaluate_per_pair_kind`
runs a no-grad forward pass over the stratified test split and writes
`outputs/runs/<MODEL>/<hcl_run>/test_breakdown_by_pair_kind.json` with
per-kind `accuracy / precision / recall / f1 / pos_prob_mean /
neg_prob_mean / prob_gap / n` for `pair_kind ∈ {positive, corrupted,
swapped}`. Tuning hints:

- `corrupted` accuracy lags `swapped` → raise `pair_kind_weights[1]`
- `swapped` accuracy lags `corrupted` → raise `pair_kind_weights[2]`
- `positive` recall is the bottleneck → raise `pair_kind_weights[0]`

**Why the per-pair_kind stratified eval matters.** The pair test split
is ~12% `label=1` (positives) and ~88% `label=0` (corrupted + swapped).
A uniform random `max_test_rows=8` lands on all-negatives ~36% of the
time → `pos_prob_mean` collapses to `0.0` and accuracy/F1 look like a
bug. Stratifying by `pair_kind` (`data.stratify_by=pair_kind`) gives
~equal counts per kind and produces interpretable metrics on a
20-step smoke.

Expected timings on one H100 80GB (hopper) at the v4 smoke caps:

| Step | Wall time |
|------|-----------|
| 1 CPT          | ~30 s after HF cache is warm (first-ever run ~8 min for shard download) |
| 2 HCL          | ~5 min (20 steps + 99-row stratified eval + breakdown) |
| 3 SFT-align    | ~3 min |
| 4 Inference    | ~30 s (8-row smoke) |
| 5 eval_baseline| ~30 s |
| **Total**      | ~10–12 min for a single backbone |

If the summary block at the bottom of the log reads `1_cpt PASS … 5_eval_baseline PASS`,
the host is green end-to-end on the new datasets.

---

## 5. Real training run

CPT and HCL are **continual-curriculum** sequences: train on `stage1/`
data (metadata, `max_seq_length=4096`), then *continue* on `stage2/`
data (all files, `max_seq_length=10240`) starting from sub-stage 1's
checkpoint. SFT (Stage 3) has no sub-stages.

### Stage 1 (CPT) — sub-stage 1 → sub-stage 2

```bash
# sub-stage 1: metadata @ 4096
docker compose run --rm pretraining \
    stages/pretraining/train.py \
    --config stages/pretraining/configs/full_cpt_v3_quarter.yaml \
    run.run_name=cpt_substage1 \
    data.train_file=/data/full_cpt/full_cpt_v3/stage1/train.parquet \
    data.validation_file=/data/full_cpt/full_cpt_v3/stage1/val.parquet \
    data.test_file=/data/full_cpt/full_cpt_v3/stage1/test.parquet \
    data.max_seq_length=4096

# sub-stage 2: all files @ 10240, continues from sub-stage 1
docker compose run --rm pretraining \
    stages/pretraining/train.py \
    --config stages/pretraining/configs/full_cpt_v3_quarter.yaml \
    run.run_name=cpt_substage2 \
    run.resume_from_checkpoint=/app/outputs/runs/Foundation-Sec-8B-Reasoning/cpt_substage1/checkpoint-<N> \
    data.train_file=/data/full_cpt/full_cpt_v3/stage2/train.parquet \
    data.validation_file=/data/full_cpt/full_cpt_v3/stage2/val.parquet \
    data.test_file=/data/full_cpt/full_cpt_v3/stage2/test.parquet \
    data.max_seq_length=10240
```

Replace `<N>` with the actual checkpoint number from sub-stage 1's
output dir (latest one is usually right; the trainer writes
`checkpoint-N/` every `save_fraction_of_epoch`).

### Stage 2 (HCL) — same pattern, chained onto CPT sub-stage 2's adapter

```bash
# HCL sub-stage 1: chains on CPT sub-stage 2's adapter, metadata pairs @ 1024
#   NOTE: data files MUST be the joined pairs_*.parquet (anchor_text + pair_text
#   + pair_kind columns), NOT the single-view train.parquet in the same dir.
docker compose run --rm hcl \
    stages/hcl/train.py \
    --config stages/hcl/configs/lp_hcl_v2_quarter.yaml \
    run.run_name=hcl_substage1 \
    model.phase1_adapter_path=/app/outputs/runs/Foundation-Sec-8B-Reasoning/cpt_substage2/final_model \
    data.train_file=/data/pl_hcl/pl_hcl_v2/stage1/pairs_train.parquet \
    data.validation_file=/data/pl_hcl/pl_hcl_v2/stage1/pairs_val.parquet \
    data.test_file=/data/pl_hcl/pl_hcl_v2/stage1/pairs_test.parquet \
    data.max_seq_length=1024

# HCL sub-stage 2: all-files pairs @ 10240, continues from HCL sub-stage 1
docker compose run --rm hcl \
    stages/hcl/train.py \
    --config stages/hcl/configs/lp_hcl_v2_quarter.yaml \
    run.run_name=hcl_substage2 \
    run.resume_from_checkpoint=/app/outputs/runs/Foundation-Sec-8B-Reasoning/hcl_substage1/checkpoint-<N> \
    data.train_file=/data/pl_hcl/pl_hcl_v2/stage2/pairs_train.parquet \
    data.validation_file=/data/pl_hcl/pl_hcl_v2/stage2/pairs_val.parquet \
    data.test_file=/data/pl_hcl/pl_hcl_v2/stage2/pairs_test.parquet \
    data.max_seq_length=10240
```

### Stage 3 (SFT misalignment) — single pass, no sub-stages

**Train on the large scrubbed sft_v2 dataset** (heuristic labels, ~474K rows):

```bash
docker compose run --rm sft-align \
    stages/downstream/sft/train.py \
    --config stages/downstream/configs/sft_misalignment_example.yaml \
    run.run_name=sft_align_v2 \
    data.train_file=/data/sft/sft_v2/classifier_train.parquet \
    data.validation_file=/data/sft/sft_v2/classifier_val.parquet \
    data.test_file=/data/sft/sft_v2/classifier_test.parquet
```

### Stage 3b (Evaluation) — score the SFT-trained adapter on the GOLD challenge set

The challenge set (`sft/challenge/classifier_test.parquet`, 1,444 rows) is
**eval-only** — it contains every human-reviewed skill with the gold
`alignment_class` label. Never used for training; reserved as the
held-out evaluation corpus.

```bash
docker compose run --rm sft-align \
    stages/downstream/inference.py \
    --config stages/downstream/configs/inference_misalignment_challenge.yaml \
    model.adapter_path=/app/outputs/runs/Foundation-Sec-8B-Reasoning/sft_align_v2/final_model
```

This writes `predictions_misalignment_detection.jsonl` + `metrics_misalignment_detection.json`
(accuracy / precision / recall / F1 vs the gold labels) to the inference run's output dir.

Both sft_v2 and challenge parquets carry `skill_text` + `alignment_class`
columns, which is what `tasks/misalignment_detection.py` reads — so the
same task module powers both training (on sft_v2) and evaluation
(on challenge).

### Controlling the training-data fraction

`data.train_fraction`, `data.validation_fraction`, and
`data.test_fraction` are independent. To train on half the rows while
keeping the full validation and test splits, just override `train_fraction`:

```bash
... data.train_fraction=0.5
```

(`1.0` or omitted both mean "use all rows" — only fractions in `(0, 1)`
shrink a split, see `common/data.py:subsample_splits`.) Absolute caps
via `data.max_train_rows` / `data.max_validation_rows` /
`data.max_test_rows` are also available; when both a fraction and a cap
apply to the same split, the smaller of the two wins.

### Run-name discipline

**Use a distinct `run_name` per experiment.** The default
`auto_resume: true` keys off `<output_root>/<sanitized_model>/<run_name>/`,
so reusing a name silently extends an old run — and reusing across stages
will *fail* because checkpoint shapes differ (see §7 known issues).

### LR-schedule heads-up for `resume_from_checkpoint`

`run.resume_from_checkpoint` restores the optimizer + scheduler state.
With the default `cosine` schedule, the LR has decayed near zero by the
end of sub-stage 1 — so sub-stage 2 won't learn much unless you also
override one of:

- `training.num_train_epochs=<higher>` so the cosine extends past sub-stage
  1's endpoint, or
- `training.lr_scheduler_type=constant` to ignore the inherited schedule.

(A cleaner fix would be a CPT-side `model.phase1_adapter_path` knob
mirroring HCL's; that's tracked as a follow-up. Until it lands, override
the schedule explicitly when chaining sub-stages.)

### Scaling to multi-GPU

The docker entrypoint **auto-detects visible GPUs** and launches trainer
scripts (`stages/*/train.py`) under `torchrun --standalone --nproc_per_node=$N`
when `$N >= 2`. Single-GPU hosts and the non-trainer scripts
(`inference.py`, `eval_baseline.py`) stay single-process. Nothing to do
on a multi-GPU host beyond pointing `--gpus all` (already the default in
`docker-compose.yml`).

Inspect the launcher line at the top of any trainer's stdout to confirm:

```
[entrypoint] DDP launch: torchrun --standalone --nproc_per_node=4 -m stages.pretraining.train ...
```

**Manual override** — set `AGENTSKILLS_NPROC` in `.env` or the shell:

| Value | Effect |
|---|---|
| unset (default) | auto-detect; trainers use all visible GPUs, others single-process |
| `1` | force single-process even on a multi-GPU host (useful for debugging) |
| `<int> ≥ 2` | use exactly that many GPUs (e.g. `2` on an 8-GPU host) |
| `CUDA_VISIBLE_DEVICES=0,1` (separate var) | restrict which GPUs are visible; auto-detect then sees 2 |

**Effective batch size** is what the optimizer actually sees per step:

```
effective_batch = per_device_train_batch_size × world_size × gradient_accumulation_steps
```

The Jetstream validation used `per_device_train_batch_size=1`,
`gradient_accumulation_steps=4`, 2 GPUs → effective batch **8**. To
preserve that on a different GPU count:

| GPUs | `per_device_train_batch_size` | `gradient_accumulation_steps` | Effective batch |
|------|-------------------------------|-------------------------------|-----------------|
| 1    | 1                             | 8                             | 8 |
| 2    | 1                             | 4                             | 8 (validated)   |
| 4    | 1                             | 2                             | 8 |
| 8    | 1                             | 1                             | 8 |
| 8    | 2                             | 1                             | 16 (need LR rescale) |

If you let effective batch *grow* (e.g. keep `grad_accum=4` on 8 GPUs →
effective batch 32), rescale `training.learning_rate` proportionally —
linear-scale (×4) is the default heuristic.

**Memory headroom on bigger cards** — if you have ≥40 GB VRAM per GPU
and don't need 8-bit quantization for the chosen backbone, the
following two flips give a noticeable wall-clock win for real training
runs:

```
model.gradient_checkpointing=false         # ~30% throughput win
model.quantization=null                    # drop 8-bit base; bf16 LoRA fits in 40GB
```

(Both stay ON by default for safety on smaller cards; flip per-config or
per-CLI when you know your VRAM budget.)

### Full evaluation matrix across all backbones

`scripts/run_full_eval.sh` orchestrates the curated-dataset eval matrix
end-to-end across multiple backbones. Per model, it runs:

1. **Baseline ppl** (intrinsic next-token-prediction) — pre-CPT
2. **Zero-shot inference** misalignment_detection — pre-CPT
3. **Zero-shot inference** malicious_detection — pre-CPT
4. **CPT sub-stage 1** training (stage1 data, `max_seq_length=4096`)
5. **Baseline ppl** — post-CPT (adapter-aware; `eval_baseline.py` now
   accepts `model.adapter_path`)
6. **Zero-shot inference** misalignment_detection — post-CPT
7. **Zero-shot inference** malicious_detection — post-CPT
8. **SFT misalignment** training, chained from the CPT adapter
9. **SFT-adapter inference** misalignment_detection
10. **SFT malicious** training, chained from the CPT adapter
11. **SFT-adapter inference** malicious_detection

Then `scripts/aggregate_eval.py` rolls every run-name's metrics JSON
into one wide comparison table (Markdown + JSON).

Default model list (override on CLI):

- `Foundation-Sec-8B-Reasoning`
- `RedSage-Qwen3-8B-DPO`
- `WhiteRabbitNeo-2-8B`
- `llama3.1-8b`
- `Qwen3-8B`
- `gemma-4-E4B`

**Run it:**

```bash
# Make sure both env vars are exported (or in .env), plus HF_TOKEN for
# the gated backbones (Llama-3.1, Gemma, WhiteRabbitNeo).
export AGENTSKILLS_DATA_HOST=/opt/agentskills/data
export AGENTSKILLS_OUTPUTS_HOST=/opt/agentskills/training_outputs
export HF_TOKEN=hf_...

# All six default models:
bash scripts/run_full_eval.sh

# Or a subset:
bash scripts/run_full_eval.sh Foundation-Sec-8B-Reasoning Qwen3-8B

# After completion (or anytime to see partial results):
docker compose run --rm --no-deps --entrypoint python pretraining \
    scripts/aggregate_eval.py /app/outputs/runs
```

**Idempotent.** Each step checks for its expected output artifact
(`final_model/`, `baseline_metrics.json`, or `metrics_<task>.json`)
and skips if found. Re-running after a crash picks up where it left
off; you don't lose hours of work.

**Failure-resilient.** Per-model failures don't abort the loop — the
script reports a per-model success/fail summary at the end. The one
exception is a CPT failure: downstream steps for that model are
skipped (they all depend on the CPT adapter).

**Runtime budget.** Rough rule of thumb: per 8B-class model on a
single A100 40GB, CPT sub-stage 1 dominates at 8-30 hr depending on
`train_fraction`, the seven eval steps each add ~5-15 min, and each
SFT add ~30-90 min. **Multi-GPU helps proportionally** — the
auto-torchrun entrypoint will use every visible GPU for trainer
scripts. For all 6 models on a single A100, plan a week; on 8×A100,
plan a long weekend. Lower `data.train_fraction` (e.g. `0.1`) to
shorten end-to-end iteration when validating the pipeline rather than
producing publication numbers.

**Coverage caveat.** This is the **curated dataset** path only —
real-world / human-verified-set inference and few-shot (n-shot)
inference are not wired up in this orchestrator. Tracked as
follow-ups; both will land once the verified set is finalized and
the exemplar pool is decided.

---

## 6. Stages 2 & 3 against legacy pre-refactor data (smoke only)

The legacy `pl_hcl` and `sft` parquets use a different schema than the new
code expects:

| Stage | New schema (training code) | Legacy schema (pre-refactor) |
|---|---|---|
| 2 — HCL | `anchor_text`, `pair_text`, `pair_kind`, `label` | `text`, `label`, `sub_strategy`, `anchor_skill_id` |
| 3 — SFT misalignment | `skill_text`, `alignment_class` | `instruction`, `input_text`, `evidence_text`, `target_text` |
| 3 — SFT malicious | `skill_text`, `overall_class` | same as misalignment legacy |

`training_adapters/adapt_legacy_to_new_schema.py` writes a small adapted
parquet so the code paths can be smoke-tested:

```bash
docker compose run --rm --no-deps \
    -v /opt/agentskills/data:/data:ro \
    -v /opt/agentskills/data/adapted:/adapted \
    --entrypoint python pretraining \
    /app/training_adapters/adapt_legacy_to_new_schema.py \
        --legacy-root /data --out-root /adapted --rows 2000
```

The adapted parquets are **for smoke validation only** — the semantic
mapping is approximate (legacy `target_text=yes/no` collapses to
`ALIGNED/MISALIGNED` and `SAFE/MALICIOUS`). Real training runs must use
parquets produced by the refactored `agent-skills-preparation` pipeline.

---

## 7. Known issues (fixed on `main`)

During the Jetstream validation we found six regressions in the
downstream code + docker-compose. All six are fixed on `main` via
[PR #1](https://github.com/Rainaaaa/agent-skills-training/pull/1)
(merged 2026-05-12), so a fresh clone of `main` has them all. The
table below is retained as a record of what was wrong.

| File | Issue | Patched |
|---|---|---|
| `stages/downstream/sft/train.py:123-124` | Stale 2-arg call to `load_tokenizer` / 3-arg call to `load_causal_lm`. The `common.modeling` signatures were refactored to 1-arg + `(tokenizer, num_added)` tuple return; only pretraining + hcl were updated. | ✓ |
| `stages/downstream/inference.py:164-165` | Same regression as above. | ✓ |
| `stages/downstream/sft/train.py:141` | `compute_eval_steps_from_fraction(training_cfg, train_ds)` uses the old signature; the new one requires `fraction, n_train, per_device_batch, grad_accum, world_size`. Also: the SFT trainer should gate this behind `eval_fraction_of_epoch` like pretraining/hcl do. | ✓ |
| `docker-compose.yml` | Missing forwards for 9 `AGENTSKILLS_SFT_*` / `AGENTSKILLS_INFER_*` env vars that are referenced by the downstream example configs. | ✓ |
| `stages/downstream/inference.py:183-186` | `output_cfg.get(key, default)` returns the empty string from `${VAR:-}` interpolation, not the default — so `Path("")` becomes `Path(".")` and `.open("w")` raises `IsADirectoryError`. Switched to `or`-fallback. | ✓ |
| `stages/downstream/inference.py:165-167` | `load_causal_lm` only sets `device_map` when bnb quantization is on, so unquantized models stay on CPU. Pretraining/HCL get away with this because the HF `Trainer` moves the model on entry; inference doesn't use Trainer, so 8B forward passes ran on CPU and never finished (one row stuck for 12+ minutes at 100% CPU / 0% GPU). Added explicit `.to("cuda")` after adapter attach. | ✓ |

Additional smaller items observed but not fixed (low priority):

- SFT `data.py` does not honor `data.max_train_rows` / `data.max_validation_rows` /
  `data.max_test_rows` the way `common/data.py` does for pretraining. Smoke
  runs end up evaluating on the full split. Workaround: write a smaller
  parquet, or accept the longer eval time.

---

## 8. Model registry — new dict schema

`model_path.json` entries now support both `local` and `hf` keys:

```json
"Foundation-Sec-8B-Reasoning": {
  "local": "/N/project/ai4chips/.../snapshots/63c930...",
  "hf": "fdtn-ai/Foundation-Sec-8B-Reasoning"
}
```

Resolution order:

1. If `local` exists on this host's filesystem → use it.
2. Else if `hf` is set → return the repo id (transformers caches to HF cache).
3. Else raise.

Old flat-string entries (`"<name>": "<path-or-repo>"`) still work. To
register a new model on a host where the local share path is unavailable,
just add the `hf` key.

The first-time download for all 9 registered models is one
`-m common.prefetch_models` call.

---

## 9. Production-readiness checklist for Kelly IT

- [ ] `nvidia-smi` from inside the container shows your target GPU(s).
- [ ] Data zip extracted to `${AGENTSKILLS_DATA_HOST}` matches the layout in §2.
- [ ] `prefetch_models` ran clean (`downloaded=N skipped(local)=N failed=0 no-target=0`).
- [ ] Smoke runs for all 3 stages (§4) exit 0 and write `final_model/`.
- [ ] Bind-mount permissions: outputs dir is writable by the container user
      (the image runs as root; not an issue on most setups).
- [ ] If running concurrent stages on the same host, give each its own
      `run_name` and `output_root` to avoid the auto-resume footgun (§5).
- [ ] TensorBoard logs in `${AGENTSKILLS_OUTPUTS_HOST}/runs/.../tb/` — point a
      TensorBoard process at this dir if you want live metrics.

---

## 10. Quick reference — Jetstream validation evidence

End-to-end smoke results from 2026-05-12 on the Jetstream A100 (Foundation-Sec-8B
LoRA `q_proj,v_proj` r=8, bf16, gradient_checkpointing). All six entry
points exercised:

| Entry point | Wall | Metric |
|---|---|---|
| `stages/pretraining/train.py`         | 14.2 s   | val_ppl 4.41, test_ppl 6.67 |
| `stages/hcl/train.py`                 | 9.4 s    | acc 1.0 (synthetic adapter, trivial signal) |
| `stages/downstream/sft/train.py` (misalignment) | ~3 min   | test_loss 13.81 |
| `stages/downstream/sft/train.py` (malicious)    | ~3 min   | test_loss 13.73 |
| `stages/downstream/inference.py`      | 38 s for 32 rows | acc 0.469, F1 0.0 (4-step adapter; code path verified) |
| `stages/downstream/eval_baseline.py`  | 5 s      | val ppl_token 7.50, top-1 59.7%, bpb 0.69 (untrained baseline) |

The SFT eval/test numbers reflect Foundation-Sec on an out-of-domain
yes/no→aligned/misaligned mapping; they're a code-path proof, not a
quality measurement. Real numbers come from real preparation output.
