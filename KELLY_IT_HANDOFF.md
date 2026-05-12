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
    ├── full_cpt_v2/                           # Stage 1 (CPT) parquets
    │   ├── stage1/{train,val,test,unseen}.parquet
    │   └── stage2/{train,val,test,unseen}.parquet
    ├── pl_hcl/pl_hcl_v1/                      # Stage 2 (HCL pairs) parquets
    │   └── stage{1,2}/{train,val,test,...}.parquet
    ├── sft/sft_v1/                            # Stage 3 (SFT) parquets + jsonl
    │   ├── classifier_{train,val,test}.parquet
    │   └── sft_{train,val,test}.jsonl
    └── adapted/                               # OPTIONAL: legacy-schema → new-schema
                                               # only needed if you must run stages 2/3
                                               # on legacy data; see §6.
```

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

## 3. Build + warm the HF cache (one-time)

```bash
cd /opt/agentskills/agent-skills-training
docker compose build pretraining     # ~10 min, ~12 GB image
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

# Stage 3 — SFT
docker compose run --rm sft-align \
    stages/downstream/sft/train.py \
    --config stages/downstream/configs/sft_misalignment_example.yaml \
    run.run_name=smoke_sft_align \
    training.max_steps=4 \
    data.max_seq_length=512
```

Expected timings on an A100 40 GB at the smoke caps above:

| Stage | Wall time | Notes |
|---|---|---|
| 1 (CPT)        | ~15 s   | 4 steps + final eval/test |
| 2 (HCL)        | ~10 s   | Phase 1 LoRA merge + 4 contrastive steps |
| 3 (SFT)        | ~3 min  | Dominated by the start-/end-of-training full evals (see §7 known issues) |

If all three exit 0 and write a `final_model/` directory, the host is good.

---

## 5. Real training run

Same commands, drop the smoke caps:

```bash
docker compose run --rm pretraining \
    stages/pretraining/train.py \
    --config stages/pretraining/configs/full_cpt_v1_quarter.yaml \
    run.run_name=full_cpt_v1_quarter
```

**Use a distinct `run_name` per experiment.** The default
`auto_resume: true` keys off `<output_root>/<sanitized_model>/<run_name>/`,
so reusing a name silently extends an old run — and reusing across stages
will *fail* because checkpoint shapes differ (see §7 known issues).

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

## 7. Known issues (patched locally; need upstream PRs)

During the Jetstream validation we found four regressions in the
downstream code and docker-compose. All four are fixed in this checkout;
they should be pushed back to `github.com/Rainaaaa/agent-skills-training`
so future clones don't hit them.

| File | Issue | Patched |
|---|---|---|
| `stages/downstream/sft/train.py:123-124` | Stale 2-arg call to `load_tokenizer` / 3-arg call to `load_causal_lm`. The `common.modeling` signatures were refactored to 1-arg + `(tokenizer, num_added)` tuple return; only pretraining + hcl were updated. | ✓ |
| `stages/downstream/inference.py:164-165` | Same regression as above. | ✓ |
| `stages/downstream/sft/train.py:141` | `compute_eval_steps_from_fraction(training_cfg, train_ds)` uses the old signature; the new one requires `fraction, n_train, per_device_batch, grad_accum, world_size`. Also: the SFT trainer should gate this behind `eval_fraction_of_epoch` like pretraining/hcl do. | ✓ |
| `docker-compose.yml` | Missing forwards for 9 `AGENTSKILLS_SFT_*` / `AGENTSKILLS_INFER_*` env vars that are referenced by the downstream example configs. | ✓ |

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
LoRA `q_proj,v_proj` r=8, bf16, gradient_checkpointing):

| Stage | Steps | Wall | train_loss | eval/test |
|---|---|---|---|---|
| Pretraining (CPT) | 4 | 14.2 s | 1.96 → 1.85 | val_ppl 4.41, test_ppl 6.67 |
| HCL (synthetic adapter) | 4 | 9.4 s  | ~0 (trivial task) | acc 1.0 (synthetic) |
| SFT (synthetic adapter) | 4 | ~3 s training + ~100 s eval | 14.99 | test_loss 13.81 |

The SFT eval/test numbers reflect Foundation-Sec on an out-of-domain
yes/no→aligned/misaligned mapping; they're a code-path proof, not a
quality measurement. Real numbers come from real preparation output.
