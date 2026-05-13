# agent-skills-training

Three-stage training pipeline for AgentSkills-OSS misalignment models.
Each stage is a stand-alone CLI with its own YAML config; all three share
one set of building blocks in `common/` and one Docker image.

```
              ┌──────────────────────────────┐
   Stage 1    │  pretraining  (CLM CPT)      │  → checkpoints + final_model
              └──────────────┬───────────────┘
                             │ Phase 1 LoRA (optional)
                             ▼
              ┌──────────────────────────────┐
   Stage 2    │  hcl  (hierarchical          │  → contrastive head + adapter
              │       contrastive learning)  │
              └──────────────┬───────────────┘
                             │ Phase 1 + Phase 2 LoRAs
              ┌──────────────┴───────────────┐
              ▼                              ▼
       ┌────────────┐                 ┌──────────────────────┐
   3a  │ downstream │  3b  downstream/inference (predict)
       │   sft/     │       eval_baseline (CLM perplexity)
       │  • malicious_detection
       │  • misalignment_detection
       └────────────┘
```

All three stages consume datasets produced by
[agent-skills-preparation](https://github.com/Rainaaaa/agent-skills-preparation):

| Stage          | Input                                                     |
| -------------- | --------------------------------------------------------- |
| pretraining    | `full_cpt/<v>/<stage>/<split>.parquet`  (Phase 1 outputs) |
| hcl            | `pl_hcl/<v>/<stage>/<split>{,_t*}.parquet`                |
| downstream sft | scanner-labeled or human-reviewed CSV/parquet             |

`<stage>` for **pretraining** and **hcl** is one of `stage1` or `stage2` —
both trainers run as a **two-sub-stage continual curriculum**
(metadata-only @ `max_seq_length=4096` → all files @ `max_seq_length=10240`,
with sub-stage 2 resuming from sub-stage 1's checkpoint). See
[Two-substage continual training](#two-substage-continual-training) below.
SFT has no sub-stages.

## Layout

```
agent-skills-training/
├── README.md
├── Dockerfile                       # GPU image (pytorch:2.3.0-cuda12.1)
├── docker-compose.yml               # one service per stage
├── docker-entrypoint.sh
├── .dockerignore
├── .gitignore
├── requirements.txt
├── cleanup_local.sh
├── model_path.json                  # model registry (logical name → local/HF path)
│
├── common/                          # shared building blocks
│   ├── config.py                    # YAML + env-var interp + CLI key=val overrides
│   ├── registry.py                  # model_path.json resolver
│   ├── modeling.py                  # load_causal_lm / load_tokenizer / maybe_wrap_lora
│   ├── data.py                      # tokenize_and_pack, subsample_splits, …
│   ├── trainer_utils.py             # callbacks, build_training_args, auto-resume
│   ├── lm_eval.py                   # compute_metrics for perplexity / bpb
│   └── logging_utils.py
│
├── stages/
│   ├── pretraining/                 # Stage 1: CLM continued pretraining
│   │   ├── train.py
│   │   ├── configs/
│   │   │   ├── example.yaml                # smoke (env-var paths)
│   │   │   ├── full_cpt_v1_quarter.yaml    # production
│   │   │   └── full_cpt_v1_half.yaml
│   │   ├── scripts/
│   │   │   ├── run_pretraining.sh
│   │   │   └── submit_models.sh
│   │   └── README.md
│   │
│   ├── hcl/                         # Stage 2: hierarchical contrastive learning
│   │   ├── train.py
│   │   ├── data.py
│   │   ├── collator.py
│   │   ├── modeling.py
│   │   ├── configs/
│   │   │   ├── example.yaml
│   │   │   └── lp_hcl_v1_quarter.yaml
│   │   ├── scripts/
│   │   │   └── run_hcl.sh
│   │   └── README.md
│   │
│   └── downstream/                  # Stage 3: inference + classification SFT
│       ├── inference.py             # batch predict + accuracy/F1
│       ├── eval_baseline.py         # CLM perplexity sanity check
│       ├── aggregate_baselines.py   # roll up baseline runs into one summary
│       ├── sft/
│       │   ├── train.py             # generative-Q&A SFT trainer (task-parameterized)
│       │   ├── data.py
│       │   └── tasks/               # plug-in: drop a file here to add a task
│       │       ├── malicious_detection.py
│       │       └── misalignment_detection.py
│       ├── configs/
│       │   ├── sft_malicious_example.yaml
│       │   ├── sft_misalignment_example.yaml
│       │   ├── inference_example.yaml
│       │   └── eval_baseline_example.yaml
│       ├── scripts/
│       │   ├── run_sft.sh
│       │   ├── run_inference.sh
│       │   └── run_eval_baseline.sh
│       └── README.md
│
├── inputs/                          # gitignored placeholder
└── outputs/                         # gitignored placeholder
```

## What's new vs. the legacy `model_training/`

| Change | Why |
| --- | --- |
| **Top-level `stages/` package** with `pretraining / hcl / downstream` | The three sections from your design map 1:1 to subpackages; new stages get added by dropping a `stages/<name>/` folder. |
| **Downstream SFT with a task registry** | Generative-Q&A SFT trainer plus pluggable task modules. Built-ins: `malicious_detection`, `misalignment_detection`. Add a new task by writing one file under `stages/downstream/sft/tasks/` — no orchestrator edits. |
| **`stages/downstream/inference.py` (NEW)** | Batch predict with constrained scoring of `response_vocab`; reports accuracy / precision / recall / F1. Same task plug-in as SFT — so every new task gets eval for free. |
| **`${VAR:-default}` env interpolation in configs** | Every config in the repo runs on a fresh clone with just env-var overrides; no YAML edits required. Same convention as agent-skills-collection / -scanning / -preparation. |
| **Single GPU Dockerfile + compose** | `pytorch:2.3.0-cuda12.1` base; one image covers every stage. `docker compose run --rm sft-mal …` etc. |

## First-time setup

```bash
git clone git@github.com:Rainaaaa/agent-skills-training.git
cd agent-skills-training

# Local (no Docker)
pip install -r requirements.txt
export AGENTSKILLS_TRAIN_OUTPUT_ROOT=$(pwd)/outputs/runs
export AGENTSKILLS_FULL_CPT_TRAIN=/path/.../full_cpt/train.parquet
# ... see config.example.yaml for the full env-var list

# Docker (recommended for portability)
docker build -t agent-skills-training .
docker compose run --rm pretraining --config stages/pretraining/configs/example.yaml \
    training.max_steps=20
```

## Stages

### Stage 1 — Pretraining (CLM CPT)

```bash
python -m stages.pretraining.train \
    --config stages/pretraining/configs/example.yaml
```

Decoder-only continued pretraining on Phase 1 packed sequences. LoRA on
`q_proj, v_proj` by default; flip `model.use_lora: false` for
full-parameter CPT. See
[`stages/pretraining/README.md`](stages/pretraining/README.md).

### Stage 2 — HCL (post-CPT contrastive)

```bash
python -m stages.hcl.train \
    --config stages/hcl/configs/example.yaml
```

Contrastive learning on Phase 2 HCL pairs (anchor + corruption). Reads
the trained CPT LoRA via `model.phase1_adapter_path` and merges it into
the base before attaching a fresh Phase 2 LoRA. See
[`stages/hcl/README.md`](stages/hcl/README.md).

### Stage 3 — Downstream

Three independent entry points, all sharing the same backbone-loading
machinery and task registry.

```bash
# SFT for malicious detection
python -m stages.downstream.sft.train \
    --config stages/downstream/configs/sft_malicious_example.yaml

# SFT for misalignment detection
python -m stages.downstream.sft.train \
    --config stages/downstream/configs/sft_misalignment_example.yaml

# Inference (after SFT) — emits predictions + classification metrics
python -m stages.downstream.inference \
    --config stages/downstream/configs/inference_example.yaml

# Optional: CLM-perplexity sanity baseline
python -m stages.downstream.eval_baseline \
    --config stages/downstream/configs/eval_baseline_example.yaml
```

See [`stages/downstream/README.md`](stages/downstream/README.md).

## Two-substage continual training

**Stage 1 (CPT)** and **Stage 2 (HCL)** are each delivered as a
two-sub-stage **continual** curriculum. On disk both `full_cpt_v2/` and
`pl_hcl/pl_hcl_v1/` contain `stage1/` and `stage2/` subdirectories:

| Sub-stage | Data                                          | `max_seq_length` |
| --------- | --------------------------------------------- | ---------------- |
| 1         | metadata + instructions only                  | 4096             |
| 2         | all files (incl. full code/content), **continues from 1** | 10240 |

Sub-stage 2 is *not* an independent run on a fresh base — it loads
sub-stage 1's last checkpoint via `run.resume_from_checkpoint` and
continues training with the new (longer-context, larger-content) data.
The dataloader rebuilds with the new `data.train_file` and the new
`data.max_seq_length`; the LoRA adapter, tokenizer, and optimizer
state carry over from sub-stage 1.

Stage 3 (SFT) has no sub-stages — single train/val/test pass.

### CPT sub-stage 1 → sub-stage 2

```bash
# sub-stage 1: metadata @ 4096
docker compose run --rm pretraining stages/pretraining/train.py \
    --config stages/pretraining/configs/full_cpt_v1_quarter.yaml \
    run.run_name=cpt_substage1 \
    data.train_file=/data/full_cpt_v2/stage1/train.parquet \
    data.validation_file=/data/full_cpt_v2/stage1/val.parquet \
    data.test_file=/data/full_cpt_v2/stage1/test.parquet \
    data.max_seq_length=4096

# sub-stage 2: all files @ 10240, continues from sub-stage 1
docker compose run --rm pretraining stages/pretraining/train.py \
    --config stages/pretraining/configs/full_cpt_v1_quarter.yaml \
    run.run_name=cpt_substage2 \
    run.resume_from_checkpoint=/app/outputs/runs/<model>/cpt_substage1/checkpoint-<N> \
    data.train_file=/data/full_cpt_v2/stage2/train.parquet \
    data.validation_file=/data/full_cpt_v2/stage2/val.parquet \
    data.test_file=/data/full_cpt_v2/stage2/test.parquet \
    data.max_seq_length=10240
```

HCL stage1 → stage2 follows the same pattern under
`stages/hcl/configs/` and `pl_hcl/pl_hcl_v1/{stage1,stage2}/`.

### Controlling the training-data fraction independently of eval

`data.train_fraction`, `data.validation_fraction`, and
`data.test_fraction` are independent knobs. To train on half the rows
while keeping the full validation and test splits, set only
`train_fraction`:

```bash
... data.train_fraction=0.5
```

`1.0` or omitted both mean "use all rows" — only fractions in `(0, 1)`
shrink a split (`common/data.py:subsample_splits`). Absolute caps with
`max_train_rows` / `max_validation_rows` / `max_test_rows` work in
parallel; when both a fraction and a cap apply, the smaller wins.

### LR-schedule heads-up

`resume_from_checkpoint` restores the optimizer + scheduler state
alongside the LoRA weights. With the default `cosine` schedule the LR
has decayed near zero by the end of sub-stage 1, so sub-stage 2 needs
either `training.num_train_epochs=<higher>` (extends the cosine past
sub-stage 1's endpoint) or `training.lr_scheduler_type=constant`
(ignore the inherited schedule). A cleaner fix would be adding a
CPT-side `model.phase1_adapter_path` knob mirroring HCL's, which loads
the adapter onto a fresh base + fresh optimizer; tracked as a follow-up.

## Extending the pipeline

### Add a new SFT task

```bash
$EDITOR stages/downstream/sft/tasks/my_task.py
# Then register in stages/downstream/sft/tasks/__init__.py `_TASKS` list.
```

`my_task.py` exports a module-level `TASK: SFTTask` with:
- `prompt_template`: format string referencing row columns
- `input_column`: the row's main text column
- `label_column`: the gold-label column
- `response_for(row)`: maps a row to a response in `response_vocab`
- `response_vocab`: the closed set of allowed answers
- (optional) `system_prompt`

Both the SFT trainer and `inference.py` pick it up automatically.

### Add a new training stage

Drop `stages/<your_stage>/train.py` with the same shape as
`stages/pretraining/train.py` (parse `--config`, call `common.config.load_config`,
build a Trainer, save to `output_dir`). The Dockerfile copies the whole
`stages/` tree, so the new stage gets a zero-config container image
slot — add a `services.<your_stage>` entry to `docker-compose.yml` and
you have a named target.

## Running on a remote machine (Docker + SMB)

The common deployment story for the training stage: you ship a Docker
image plus a zip of the prepared datasets to a remote GPU host, and the
admin gives you a mounted SMB share to point the data at. The pipeline
is built for this — you set **two host paths** and the configs do the
rest, no YAML edits.

### Mental model

```
SMB share (admin-side, network)
    │
    ▼
host path on the remote machine, e.g. /mnt/agentskills-data
    │  (Docker bind-mount, defined in docker-compose.yml)
    ▼
container path /data:ro
    │  (env-var defaults in configs, e.g. ${AGENTSKILLS_FULL_CPT_TRAIN:-/data/full_cpt/train.parquet})
    ▼
trainer reads /data/full_cpt/train.parquet
```

You don't need to know the SMB protocol — the remote admin mounts the
share at some host path. From your side it's just "a directory on the
host." Docker bind-mounts that directory into the container, and the
configs read from the container path.

### Canonical dataset layout

Lay the zip out so the in-container defaults Just Work — that way you
only override the host-side mount point, not every individual file
path:

```
agentskills-data/                       # unzip here on the remote host
├── full_cpt_v2/                        # Stage 1 (CPT) — two sub-stages
│   ├── stage1/                         # metadata + instructions; train @ max_seq_length=4096
│   │   ├── train.parquet
│   │   ├── val.parquet
│   │   ├── test.parquet
│   │   └── unseen.parquet
│   └── stage2/                         # all files; continual-train @ max_seq_length=10240
│       └── {train,val,test,unseen}.parquet
├── pl_hcl/pl_hcl_v1/                   # Stage 2 (HCL pairs) — same two-sub-stage shape
│   ├── stage1/                         # metadata-pair anchors @ 4096
│   │   ├── {train,val,test,unseen}.parquet
│   │   └── {train,val,test,unseen}_t{1,2,3a,3b,3c}.parquet   # per-strategy splits
│   └── stage2/                         # all-files-pair anchors @ 10240
│       └── ...                         # (same shape as stage1)
├── sft/sft_v1/                         # Stage 3 (SFT) — no sub-stages
│   ├── classifier_{train,val,test}.parquet
│   └── sft_{train,val,test}.jsonl
├── adapted/                            # optional, legacy→new schema for smoke runs (see KELLY_IT_HANDOFF.md §6)
│   └── {hcl,sft_align,sft_malicious}/{train,val,test}.parquet
└── models/                             # optional, see "Backbone weights" below
    └── Foundation-Sec-8B-Reasoning/
```

See [Two-substage continual training](#two-substage-continual-training)
above for what `stage1/` vs `stage2/` mean and how to chain them.

### On the remote machine

```bash
# 1. Admin mounts the SMB share, e.g. at /mnt/agentskills-data.
#    Verify it's readable:
ls /mnt/agentskills-data/full_cpt/

# 2. Tell docker-compose where to find data + where to write outputs.
#    Put these in agent-skills-training/.env  (or export in your shell).
cat > .env <<'EOF'
AGENTSKILLS_DATA_HOST=/mnt/agentskills-data
AGENTSKILLS_OUTPUTS_HOST=/mnt/agentskills-outputs
AGENTSKILLS_BACKBONE=/data/models/Foundation-Sec-8B-Reasoning   # see below
EOF

# 3. Build + run as usual. Compose picks up .env automatically.
docker compose build
docker compose run --rm pretraining --config stages/pretraining/configs/example.yaml
```

That's the entire deploy story. If your data zip uses the canonical
layout above, you do not need to override any of the per-file env vars
(`AGENTSKILLS_FULL_CPT_TRAIN`, etc.) — their defaults already point at
`/data/full_cpt/train.parquet` and friends.

### Backbone weights — the one gotcha

`model_path.json` ships pointing at the IU BR200 cluster paths
(`/N/project/ai4chips/...`). Those won't exist on the remote machine.
Three ways to handle it:

1. **Bundle the backbone in the zip** (simplest, ~16GB for an 8B model).
   Put it at `agentskills-data/models/<name>/` and set
   `AGENTSKILLS_BACKBONE=/data/models/<name>`. The config sees an
   absolute path and skips the registry lookup.
2. **HF download at runtime** — set `AGENTSKILLS_BACKBONE` to a HF repo
   id like `fdtn-ai/Foundation-Sec-8B-Reasoning`. Needs internet on the
   remote machine, and the `models:` volume in `docker-compose.yml`
   caches the download between runs.
3. **Separate SMB share for models** — same trick as data: ask the
   admin for a second mount, point `AGENTSKILLS_MODELS_HOST` at it, and
   the compose file binds it to `/app/outputs/hf_cache`.

### Why the configs don't break across hosts

Every host-specific path in the configs is wrapped in
`${VAR:-default}`. The `default` is the in-container path under
`/data`, so a fresh clone runs with no edits. On the remote machine you
only override the *host*-side mount point (`AGENTSKILLS_DATA_HOST`).
The same trick is used in agent-skills-collection / -scanning /
-preparation, so the pattern is consistent across the four repos.

## Pipeline guarantees

- **Auto-resume from checkpoints.** `run.auto_resume: true` (default)
  resumes from the latest `checkpoint-*` under `run.output_dir`. Pick a
  new `run.run_name` (which threads into `output_dir`) for new
  experiments to avoid silently extending an old run.
- **Configs are portable.** Every `/N/slate` or `/N/project` path is
  parameterized via `${VAR:-default}` so a clone works on any host once
  env vars are set.
- **Single shared `common/` package.** New stages reuse the same
  config loader, trainer-arg builder, callbacks, perplexity metrics,
  and model registry — no duplication.
- **Lean checkpoints in HCL.** Phase 2 only writes the LoRA adapter
  plus a tiny `hcl_head.pt` (temperature, bias, optional projection),
  not the full base every save.
- **Auto multi-GPU.** The docker entrypoint inspects
  `torch.cuda.device_count()` and launches trainer scripts
  (`stages/*/train.py`) under `torchrun --standalone --nproc_per_node=$N`
  when ≥2 GPUs are visible. Inference / eval stay single-process. Override
  with `AGENTSKILLS_NPROC=<int>` in the env (e.g. `1` to force
  single-process). Effective batch =
  `per_device_batch × world_size × grad_accum` — see
  `KELLY_IT_HANDOFF.md §5` "Scaling to multi-GPU" for the rescale table.
