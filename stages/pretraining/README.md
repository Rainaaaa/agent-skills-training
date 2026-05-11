# Pretraining (Phase 1 — Continued Pretraining)

Decoder-only CLM continued pretraining over the Phase 1 packed corpus
emitted by [agent-skills-preparation](https://github.com/Rainaaaa/agent-skills-preparation)
(`full_cpt/<v>/<stage>/<split>.parquet`).

Run:

```bash
python -m stages.pretraining.train \
    --config stages/pretraining/configs/example.yaml
```

## What it does

1. Loads the parquet/JSONL splits via `common/data.py::load_splits`.
2. Resolves the backbone via `common/registry.py` (`model_path.json`).
3. Tokenizes the `text` column and packs into `max_seq_length`-long blocks.
4. Wraps the model in LoRA (`q_proj, v_proj`, rank 8 by default).
5. Runs `Trainer.train()` with our shared callbacks:
   - `MetricsJsonlCallback` → `metrics.jsonl`
   - `EpochEndEvalCallback` (opt-in via `run.train_eval_at_epoch_end`)
   - auto-resume from the latest `checkpoint-*` under `run.output_dir`
6. Final pass on validation + test, perplexity reported.

## Config knobs

See the top of `configs/example.yaml`. Key dials:

- `model.use_lora` — flip to `false` for full-parameter CPT.
- `data.train_fraction` / `data.max_train_rows` — staged rollouts (1/4 → 1/2 → 1.0).
- `training.eval_fraction_of_epoch` — eval cadence in fractions of an epoch.
- `model.attn_implementation` — `sdpa` / `flash_attention_2` / `eager`.

## Output

```
<run.output_dir>/
├── train.log               # Python log mirror
├── metrics.jsonl           # all trainer.log() entries + derived perplexity
├── checkpoint-<N>/         # periodic
├── final_model/            # final adapter + tokenizer
├── train_results.json
├── validation_results.json
├── test_results.json
└── tb/                     # if report_to: [tensorboard]
```

`final_model/` is what the HCL stage merges via `model.phase1_adapter_path`.
