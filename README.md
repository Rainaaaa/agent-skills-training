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
