#!/usr/bin/env python3
"""
Phase 1 / Full CPT entrypoint for AgentSkills-OSS.

Loads Phase 1 `full_cpt/<output_version>/{train,val,test}.parquet` splits,
runs causal-LM continued pretraining on a local or HF model, and writes
checkpoints + metrics into `run.output_dir`.

Reusable building blocks live in `common.*`; this file wires them together
for the Phase 1 recipe:

- LoRA on `q_proj, v_proj` (configurable), rank 8 default
- bf16 + gradient checkpointing
- token packing at `max_seq_length`
- eval every `training.eval_fraction_of_epoch` of an epoch (if set)
- JSONL metric log + epoch-end train/test eval callback
- explicit or auto checkpoint resume
"""

import argparse
import inspect
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from transformers import DataCollatorForLanguageModeling, Trainer, set_seed

from common.config import load_config
from common.data import compute_corpus_stats, load_splits, subsample_splits, tokenize_and_pack
from common.lm_eval import make_compute_metrics, preprocess_logits_for_metrics
from common.logging_utils import configure_logging
from common.modeling import load_causal_lm, load_tokenizer, maybe_wrap_lora
from common.registry import resolve_model_path, sanitize_for_path
from common.trainer_utils import (
    EpochEndEvalCallback,
    MetricsJsonlCallback,
    build_training_args,
    compute_eval_steps_from_fraction,
    maybe_disable_implicit_deepspeed,
    resolve_resume,
)


def _derive_output_dir(run_cfg: Dict[str, Any], model_name: Optional[str]) -> Path:
    """Pick the run's output directory.

    Precedence:
      1. `run.output_dir` if explicitly set — used as-is (back-compat).
      2. Otherwise: `<run.output_root>/<sanitized_model_name>/<run.run_name>`.

    `run_name` falls back to `experiment_name` so existing configs still work.
    """
    if run_cfg.get("output_dir"):
        return Path(run_cfg["output_dir"]).expanduser().resolve()
    output_root = run_cfg.get("output_root")
    run_name = run_cfg.get("run_name") or run_cfg.get("experiment_name")
    if not output_root or not run_name or not model_name:
        raise ValueError(
            "Either set run.output_dir explicitly, or provide all of: "
            "run.output_root, run.run_name (or experiment_name), and model.name."
        )
    return (Path(output_root).expanduser() / sanitize_for_path(model_name) / run_name).resolve()


LOGGER = logging.getLogger("full_cpt.train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full CPT trainer (Phase 1)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Ad-hoc overrides: e.g. training.max_steps=50 data.max_seq_length=1024",
    )
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    run_cfg = config["run"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    training_cfg = config["training"]

    # Resolve model path (via registry) ahead of output-dir derivation so we
    # can use the registry name as a directory component.
    if not model_cfg.get("model_name_or_path") and model_cfg.get("name"):
        resolved_path, _ = resolve_model_path(model_cfg)
        model_cfg["model_name_or_path"] = resolved_path

    output_dir = _derive_output_dir(run_cfg, model_cfg.get("name"))
    log_path = configure_logging(output_dir, run_cfg.get("log_level", "INFO"))
    LOGGER.info("Writing log to %s", log_path)
    LOGGER.info("Effective config:\n%s", yaml.safe_dump(config, sort_keys=False))

    set_seed(int(run_cfg.get("seed", 42)))

    tokenizer, num_added = load_tokenizer(model_cfg)
    raw_datasets = load_splits(data_cfg)

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

    # Capture raw byte/word/char counts BEFORE packing — used for byte- and
    # word-level perplexity in compute_metrics.
    text_column = data_cfg.get("text_column", "text")
    eval_split_for_stats = "validation" if "validation" in raw_datasets else "test"
    corpus_stats = compute_corpus_stats(
        {k: v for k, v in raw_datasets.items() if k in {"validation", "test"}},
        text_column=text_column,
    )

    tokenized = tokenize_and_pack(raw_datasets, tokenizer, data_cfg)

    model = load_causal_lm(model_cfg, num_added)
    if bool(model_cfg.get("gradient_checkpointing", True)):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False
    model = maybe_wrap_lora(model, model_cfg)

    eval_fraction = training_cfg.pop("eval_fraction_of_epoch", None)
    save_fraction = training_cfg.pop("save_fraction_of_epoch", None)
    world_size = max(int(os.environ.get("WORLD_SIZE", "1")), 1)
    per_device_batch = int(training_cfg.get("per_device_train_batch_size", 1))
    grad_accum = int(training_cfg.get("gradient_accumulation_steps", 1))

    if eval_fraction is not None:
        eval_steps, steps_per_epoch = compute_eval_steps_from_fraction(
            fraction=float(eval_fraction),
            n_train=len(tokenized["train"]),
            per_device_batch=per_device_batch,
            grad_accum=grad_accum,
            world_size=world_size,
        )
        LOGGER.info(
            "eval_fraction_of_epoch=%s -> steps_per_epoch=%d, eval_steps=%d",
            eval_fraction, steps_per_epoch, eval_steps,
        )
        training_cfg["eval_strategy"] = "steps"
        training_cfg["eval_steps"] = eval_steps
        # Default: align save cadence to eval cadence; can be overridden below.
        training_cfg["save_strategy"] = "steps"
        training_cfg["save_steps"] = eval_steps

    if save_fraction is not None:
        save_steps, steps_per_epoch_s = compute_eval_steps_from_fraction(
            fraction=float(save_fraction),
            n_train=len(tokenized["train"]),
            per_device_batch=per_device_batch,
            grad_accum=grad_accum,
            world_size=world_size,
        )
        LOGGER.info(
            "save_fraction_of_epoch=%s -> steps_per_epoch=%d, save_steps=%d",
            save_fraction, steps_per_epoch_s, save_steps,
        )
        training_cfg["save_strategy"] = "steps"
        training_cfg["save_steps"] = save_steps

    maybe_disable_implicit_deepspeed(training_cfg)
    training_args = build_training_args(training_cfg, output_dir)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    eval_callback = EpochEndEvalCallback(
        trainer_ref=[],
        train_eval_dataset=tokenized.get("train") if bool(run_cfg.get("train_eval_at_epoch_end", False)) else None,
        test_dataset=tokenized.get("test"),
    )
    metrics_cb = MetricsJsonlCallback(metrics_path=output_dir / "metrics.jsonl")

    eval_stats = corpus_stats.get(eval_split_for_stats)
    compute_metrics_fn = make_compute_metrics(eval_corpus_stats=eval_stats, prefix="eval")

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=data_collator,
        callbacks=[metrics_cb, eval_callback],
        compute_metrics=compute_metrics_fn,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(
        **trainer_kwargs,
    )
    eval_callback.trainer_ref.append(trainer)

    resume = resolve_resume(run_cfg, output_dir)
    LOGGER.info("Starting trainer.train(resume=%s)", resume)
    train_result = trainer.train(resume_from_checkpoint=resume)

    train_metrics = train_result.metrics
    if "train_loss" in train_metrics and train_metrics["train_loss"] < 30:
        train_metrics["train_ppl"] = math.exp(train_metrics["train_loss"])
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()
    trainer.save_model(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))

    if "validation" in tokenized:
        LOGGER.info("Final validation-split eval")
        val_metrics = trainer.evaluate(eval_dataset=tokenized["validation"], metric_key_prefix="validation")
        if "validation_loss" in val_metrics and val_metrics["validation_loss"] < 30:
            val_metrics["validation_ppl"] = math.exp(val_metrics["validation_loss"])
        trainer.log_metrics("validation", val_metrics)
        trainer.save_metrics("validation", val_metrics)

    if "test" in tokenized:
        LOGGER.info("Final test-split eval")
        test_metrics = trainer.evaluate(eval_dataset=tokenized["test"], metric_key_prefix="test")
        if "test_loss" in test_metrics and test_metrics["test_loss"] < 30:
            test_metrics["test_ppl"] = math.exp(test_metrics["test_loss"])
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)

    LOGGER.info("Done. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
