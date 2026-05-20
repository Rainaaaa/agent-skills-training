#!/usr/bin/env python3
"""SFT trainer for a registered classification task.

Trains a decoder-only LM to generate a single answer token (`malicious` /
`safe`, or `aligned` / `misaligned`) given a templated prompt. Same
backbone-loading + LoRA + auto-resume + JSONL-metrics machinery as the
pretraining and HCL stages — only the data layout is different.

Run:

    python -m stages.downstream.sft.train \\
        --config stages/downstream/configs/sft_malicious_example.yaml

Config knobs (see configs/sft_*_example.yaml for full templates):

    data.task: malicious_detection | misalignment_detection | <my_pkg.my_task>
    data.train_file, data.validation_file, data.test_file: parquet/jsonl
    data.balance_by_label: true | false   # equalize pos/neg counts at load
    data.max_seq_length: 2048
    model.name: <registry id>   OR  model.model_name_or_path: <path>
    model.use_lora: true        # rank/alpha defaults from common/modeling
    model.phase1_adapter_path:  # optional CPT-LoRA to merge first
    model.phase2_adapter_path:  # optional HCL-LoRA to merge first
    training.*: forwarded to TrainingArguments
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from peft import PeftModel
from transformers import Trainer, set_seed
from transformers.data.data_collator import DataCollatorForLanguageModeling

from common.config import load_config
from common.logging_utils import configure_logging
from common.modeling import load_causal_lm, load_tokenizer, maybe_wrap_lora
from common.registry import resolve_model_path, sanitize_for_path
from common.trainer_utils import (
    MetricsJsonlCallback,
    build_training_args,
    compute_eval_steps_from_fraction,
    maybe_disable_implicit_deepspeed,
    resolve_resume,
)

from stages.downstream.sft.data import load_splits, tokenize_for_sft, _balance_split
from stages.downstream.sft.tasks import load_task


LOGGER = logging.getLogger("downstream.sft.train")


def _derive_output_dir(run_cfg: Dict[str, Any], model_name: Optional[str], task_name: str) -> Path:
    if run_cfg.get("output_dir"):
        return Path(run_cfg["output_dir"]).expanduser().resolve()
    output_root = run_cfg.get("output_root")
    run_name = run_cfg.get("run_name") or run_cfg.get("experiment_name") or f"sft_{task_name}"
    if not output_root or not model_name:
        raise ValueError(
            "Either set run.output_dir, or all of: run.output_root, run.run_name, "
            "and model.name (so we can build <root>/<model>/<run>)."
        )
    return (Path(output_root).expanduser()
            / sanitize_for_path(model_name)
            / f"sft_{task_name}__{run_name}").resolve()


def _load_and_merge_phase_adapter(model, model_cfg: Dict[str, Any], key: str):
    """Load a prior-stage LoRA adapter from model_cfg[key] and merge it.

    Mirrors stages/hcl/modeling.load_phase1_adapter_if_any. Used to chain SFT
    on top of CPT (phase1) and/or HCL (phase2) adapters: each named adapter
    is loaded in eval mode and folded into the base via merge_and_unload, so
    the fresh SFT LoRA wraps cleanly on top of the merged weights.

    Returns the model unchanged when the config value is empty/missing.
    """
    adapter_path = model_cfg.get(key)
    if not adapter_path:
        return model
    adapter_path = str(Path(adapter_path).expanduser().resolve())
    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"{key} not found: {adapter_path}")
    quantized = bool(getattr(model, "is_loaded_in_4bit", False)) or bool(
        getattr(model, "is_loaded_in_8bit", False)
    )
    if quantized:
        raise ValueError(
            f"Cannot merge {key} into a quantized base — merge_and_unload "
            "requires unquantized weights. Drop model.quantization or omit "
            f"{key}."
        )
    LOGGER.info("Loading %s from %s", key, adapter_path)
    merged = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    LOGGER.info("Merging %s into base weights (merge_and_unload)", key)
    return merged.merge_and_unload()


class _DropDataCollator:
    """Pad to longest in batch; preserve `labels` field with -100 mask."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        ids, labels, mask = [], [], []
        for f in features:
            pad_n = max_len - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * pad_n)
            labels.append(f["labels"] + [-100] * pad_n)
            mask.append(f["attention_mask"] + [0] * pad_n)
        return {
            "input_ids":      torch.tensor(ids, dtype=torch.long),
            "labels":         torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="Optional section.key=value overrides.")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    run_cfg = config["run"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    training_cfg = config["training"]

    seed = int(run_cfg.get("seed", 42))
    set_seed(seed)
    maybe_disable_implicit_deepspeed(training_cfg)

    task_name = data_cfg.get("task")
    if not task_name:
        raise ValueError("data.task is required (e.g. 'malicious_detection').")
    task = load_task(task_name)

    model_path, model_name = resolve_model_path(model_cfg)
    output_dir = _derive_output_dir(run_cfg, model_name, task.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir, run_cfg.get("log_level", "INFO"))

    LOGGER.info("task=%s model=%s output=%s", task.name, model_path, output_dir)

    tokenizer, num_added = load_tokenizer(model_cfg)
    model = load_causal_lm(model_cfg, num_added)
    # Chain on prior-stage adapters before applying the fresh SFT LoRA, so SFT
    # can train on top of {raw base, CPT-merged base, HCL-merged base}.
    model = _load_and_merge_phase_adapter(model, model_cfg, "phase1_adapter_path")
    model = _load_and_merge_phase_adapter(model, model_cfg, "phase2_adapter_path")
    model = maybe_wrap_lora(model, model_cfg)

    ds = load_splits(data_cfg)

    if data_cfg.get("balance_by_label", False):
        for split in list(ds.keys()):
            ds[split] = _balance_split(ds[split], task, seed=seed)

    max_seq_length = int(data_cfg.get("max_seq_length", 2048))
    tokenized = tokenize_for_sft(
        ds, tokenizer, task,
        max_seq_length=max_seq_length,
        drop_no_label=bool(data_cfg.get("drop_no_label", True)),
    )

    eval_fraction = training_cfg.pop("eval_fraction_of_epoch", None)
    save_fraction = training_cfg.pop("save_fraction_of_epoch", None)
    import os as _os
    world_size = max(int(_os.environ.get("WORLD_SIZE", "1")), 1)
    per_device_batch = int(training_cfg.get("per_device_train_batch_size", 1))
    grad_accum = int(training_cfg.get("gradient_accumulation_steps", 1))
    if eval_fraction is not None and "train" in tokenized:
        eval_steps, _ = compute_eval_steps_from_fraction(
            fraction=float(eval_fraction),
            n_train=len(tokenized["train"]),
            per_device_batch=per_device_batch,
            grad_accum=grad_accum,
            world_size=world_size,
        )
        training_cfg["eval_strategy"] = "steps"
        training_cfg["eval_steps"] = eval_steps
        training_cfg["save_strategy"] = "steps"
        training_cfg["save_steps"] = eval_steps
    if save_fraction is not None and "train" in tokenized:
        save_steps, _ = compute_eval_steps_from_fraction(
            fraction=float(save_fraction),
            n_train=len(tokenized["train"]),
            per_device_batch=per_device_batch,
            grad_accum=grad_accum,
            world_size=world_size,
        )
        training_cfg["save_strategy"] = "steps"
        training_cfg["save_steps"] = save_steps

    training_args = build_training_args(training_cfg, output_dir)

    collator = _DropDataCollator(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized.get("train"),
        eval_dataset=tokenized.get("validation"),
        data_collator=collator,
        callbacks=[MetricsJsonlCallback(output_dir / "metrics.jsonl")],
    )

    resume = resolve_resume(run_cfg, output_dir)
    train_result = trainer.train(resume_from_checkpoint=resume)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    trainer.save_model(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))

    if "validation" in tokenized:
        val_metrics = trainer.evaluate(eval_dataset=tokenized["validation"], metric_key_prefix="validation")
        trainer.log_metrics("validation", val_metrics)
        trainer.save_metrics("validation", val_metrics)

    if "test" in tokenized:
        test_metrics = trainer.evaluate(eval_dataset=tokenized["test"], metric_key_prefix="test")
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)

    LOGGER.info("Done. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
