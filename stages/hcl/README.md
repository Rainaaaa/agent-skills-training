# HCL — Hierarchical Contrastive Learning (post-pretraining)

Phase 2 in the AgentSkills-OSS training recipe. Trains a contrastive
head on `(anchor, pair, label)` triples produced by
[agent-skills-preparation](https://github.com/Rainaaaa/agent-skills-preparation)
(`pl_hcl/<v>/<stage>/<split>{,_t1,_t2,_t3*}.parquet`), on top of the
backbone produced by Stage 1.

Run:

```bash
python -m stages.hcl.train \
    --config stages/hcl/configs/example.yaml
```

## What it does

For each pair:

1. Tokenize anchor and pair independently (right-padded by `HclPairCollator`).
2. Forward both through the LoRA-wrapped CausalLM with `output_hidden_states=True`.
3. Pool the last hidden state (`last_token` by default; `mean` available).
4. Optionally project through `Linear(hidden, proj_dim)`.
5. Score with `cos(anchor, pair) / temperature + bias`; temperature and
   bias are learnable.
6. Loss: `BCE(logit, label)`, optionally re-weighted by `pair_kind`.

Optional in-batch InfoNCE term over the positive subset
(`model.hcl.ce_loss_weight`, default 0).

## Phase 1 chaining

Set `model.phase1_adapter_path` to a `final_model/` directory written
by Stage 1. The adapter is loaded with `is_trainable=False` and folded
into the base via `merge_and_unload()` before a fresh Phase 2 LoRA is
attached. Drop the key to start from the raw backbone.

> **Quantization caveat**: `merge_and_unload()` requires unquantized
> weights. If `model.quantization` is set, leave `phase1_adapter_path`
> empty.

## Lean checkpoints

Each `checkpoint-*/` contains:

- LoRA adapter (`adapter_config.json`, `adapter_model.safetensors`)
- `hcl_head.pt` — `{log_temperature, bias, [proj_state_dict], pooling}`
- Trainer artifacts (`optimizer.pt`, `scheduler.pt`, …)

To reload for inference, recreate `HclPairModel(build_hcl_model(...))`
with the same config (so LoRA shapes match) and load `hcl_head.pt`.
