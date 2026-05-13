#!/usr/bin/env python3
"""
Evaluate one baseline causal-LM on the Phase 1 test split.

Reuses the same building blocks as Phase 1 training (`common/*`) so metrics are
comparable to mid-training and post-training evaluations of the trained model.

Each baseline model produces:

  baseline/output/<run_name>/
    baseline_metrics.json    # final metric dict
    baseline_metrics.jsonl   # one-line copy (easy to concatenate across runs)
    train.log                # full Python log
    metrics.jsonl            # raw trainer.log entries
    config_snapshot.json     # the effective YAML (including model path)

Token-level perplexity is NOT directly comparable across tokenizers; for
cross-model comparison rely on `ppl_byte`, `ppl_word`, `ppl_char`, and
`bits_per_byte`, which are normalized by raw text length.
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path

import yaml
from transformers import DataCollatorForLanguageModeling, Trainer, set_seed

from common.config import load_config
from common.data import compute_corpus_stats, load_splits, subsample_splits, tokenize_and_pack
from common.lm_eval import make_compute_metrics, preprocess_logits_for_metrics
from common.logging_utils import configure_logging
from common.modeling import load_causal_lm, load_tokenizer
from common.registry import resolve_model_path, sanitize_for_path
from common.trainer_utils import (
    MetricsJsonlCallback,
    build_training_args,
    maybe_disable_implicit_deepspeed,
)


def _derive_output_dir(run_cfg, model_name):
    if run_cfg.get("output_dir"):
        from pathlib import Path
        return Path(run_cfg["output_dir"]).expanduser().resolve()
    output_root = run_cfg.get("output_root")
    run_name = run_cfg.get("run_name") or run_cfg.get("experiment_name")
    if not output_root or not run_name or not model_name:
        raise ValueError(
            "Either set run.output_dir explicitly, or all of: "
            "run.output_root, run.run_name (or experiment_name), and model.name."
        )
    from pathlib import Path
    return (Path(output_root).expanduser() / sanitize_for_path(model_name) / run_name).resolve()


LOGGER = logging.getLogger("baseline.eval")


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline LLM evaluation runner")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Ad-hoc overrides, e.g. data.max_test_rows=200 model.attn_implementation=sdpa",
    )
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    run_cfg = config["run"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    training_cfg = config["training"]

    if not model_cfg.get("model_name_or_path") and model_cfg.get("name"):
        resolved_path, _ = resolve_model_path(model_cfg)
        model_cfg["model_name_or_path"] = resolved_path

    output_dir = _derive_output_dir(run_cfg, model_cfg.get("name"))
    log_path = configure_logging(output_dir, run_cfg.get("log_level", "INFO"))
    LOGGER.info("Writing log to %s", log_path)
    LOGGER.info("Effective config:\n%s", yaml.safe_dump(config, sort_keys=False))

    with (output_dir / "config_snapshot.json").open("w", encoding="utf-8") as h:
        json.dump(_to_jsonable(config), h, ensure_ascii=False, indent=2)

    set_seed(int(run_cfg.get("seed", 42)))

    tokenizer, num_added = load_tokenizer(model_cfg)
    raw_datasets = load_splits(data_cfg)

    # Drop any train split before subsampling — baseline eval never touches it.
    raw_datasets = subsample_splits(
        raw_datasets,
        max_rows={
            "train": data_cfg.get("max_train_rows"),
            "validation": data_cfg.get("max_validation_rows"),
            "test": data_cfg.get("max_test_rows"),
        },
        fractions={
            "train": data_cfg.get("train_fraction"),
            "validation": data_cfg.get("validation_fraction"),
            "test": data_cfg.get("test_fraction"),
        },
        seed=int(run_cfg.get("seed", 42)),
    )

    text_column = data_cfg.get("text_column", "text")
    eval_split = run_cfg.get("eval_split", "test")  # "test" or "validation"
    if eval_split not in raw_datasets:
        raise ValueError(
            f"eval_split={eval_split!r} not in dataset; available={list(raw_datasets)}"
        )

    # Baseline eval only touches `eval_split` — drop the others before
    # tokenization so we don't waste minutes packing a 100K-row train split
    # the Trainer will never read. (Was a noticeable cost in the
    # full-eval-orchestrator pre/post-CPT baseline pairs.)
    for split in list(raw_datasets.keys()):
        if split != eval_split:
            del raw_datasets[split]

    corpus_stats = compute_corpus_stats(
        {eval_split: raw_datasets[eval_split]}, text_column=text_column
    )
    tokenized = tokenize_and_pack(raw_datasets, tokenizer, data_cfg)

    model = load_causal_lm(model_cfg, num_added)
    # Optional: load a trained LoRA adapter on top of the base for
    # post-training intrinsic eval (e.g. measure perplexity *after* CPT).
    # Empty / unset → eval the raw backbone (pre-training baseline).
    adapter_path = model_cfg.get("adapter_path")
    if adapter_path:
        from peft import PeftModel
        LOGGER.info("Attaching adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.config.use_cache = False  # eval only; turn off KV cache for memory

    # Eval-only run; force eval_strategy="no" to suppress trainer's mid-train logic.
    training_cfg = dict(training_cfg)
    training_cfg.setdefault("eval_strategy", "no")
    training_cfg.setdefault("save_strategy", "no")
    training_cfg.setdefault("report_to", [])
    training_cfg.setdefault("do_train", False)
    training_cfg.setdefault("do_eval", True)
    training_cfg.setdefault("per_device_eval_batch_size", 1)
    training_cfg.pop("num_train_epochs", None)
    maybe_disable_implicit_deepspeed(training_cfg)
    training_args = build_training_args(training_cfg, output_dir)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    eval_stats = corpus_stats.get(eval_split)
    compute_metrics_fn = make_compute_metrics(eval_corpus_stats=eval_stats, prefix="eval")
    metrics_cb = MetricsJsonlCallback(metrics_path=output_dir / "metrics.jsonl")

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized[eval_split],
        data_collator=data_collator,
        callbacks=[metrics_cb],
        compute_metrics=compute_metrics_fn,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    LOGGER.info("Running baseline evaluate() on split=%s ...", eval_split)
    metrics = trainer.evaluate(
        eval_dataset=tokenized[eval_split],
        metric_key_prefix="eval",
    )

    payload = {
        "run_name": run_cfg.get("experiment_name", output_dir.name),
        "model_name_or_path": model_cfg["model_name_or_path"],
        "eval_split": eval_split,
        "eval_split_size_rows": len(raw_datasets[eval_split]),
        "eval_split_size_packed_blocks": len(tokenized[eval_split]),
        "eval_corpus_stats": eval_stats,
        "metrics": _to_jsonable(metrics),
    }
    out_json = output_dir / "baseline_metrics.json"
    with out_json.open("w", encoding="utf-8") as h:
        json.dump(payload, h, ensure_ascii=False, indent=2)

    line_path = output_dir / "baseline_metrics.jsonl"
    with line_path.open("w", encoding="utf-8") as h:
        h.write(json.dumps(payload, ensure_ascii=False))
        h.write("\n")

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    LOGGER.info("Wrote %s", out_json)
    LOGGER.info("Done. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
