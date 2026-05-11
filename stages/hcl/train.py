#!/usr/bin/env python3
"""
Phase 2 / LP-HCL CPT entrypoint for AgentSkills-OSS.

Runs hierarchical pair-discriminative contrastive learning on the Phase 2
pair dataset emitted by `data_preparation.run_phase2`. The recipe:

  - load contrastive pair splits (parquet/jsonl) and tokenize anchor + pair
    independently
  - load a CausalLM backbone (registry-resolved or explicit path), optionally
    starting from a Phase 1 LoRA adapter that gets merged into the base
  - apply a fresh Phase 2 LoRA on top
  - train with BCE on `cos(anchor, pair) / temperature + bias`, with optional
    per-pair-kind loss weighting and an optional in-batch InfoNCE term
  - save the trained backbone (LoRA adapter or merged weights) plus the small
    contrastive head (`hcl_head.pt`) at every checkpoint

Reusable building blocks (config loader, logging, model loading, callbacks,
TrainingArguments builder) are imported from `common.*`.
"""

import argparse
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from transformers import EvalPrediction, Trainer, set_seed

from common.config import load_config
from common.logging_utils import configure_logging
from common.modeling import load_tokenizer
from common.registry import resolve_model_path, sanitize_for_path
from common.trainer_utils import (
    EpochEndEvalCallback,
    MetricsJsonlCallback,
    build_training_args,
    compute_eval_steps_from_fraction,
    maybe_disable_implicit_deepspeed,
    resolve_resume,
)

from stages.hcl.collator import HclPairCollator
from stages.hcl.data import load_pair_splits, subsample_pair_splits, tokenize_pairs
from stages.hcl.modeling import HclPairModel, build_hcl_model


LOGGER = logging.getLogger("lp_hcl.train")


def _derive_output_dir(run_cfg: Dict[str, Any], model_name: Optional[str]) -> Path:
    """Pick the run's output directory.

    Precedence:
      1. `run.output_dir` if explicitly set — used as-is.
      2. Otherwise: `<run.output_root>/<sanitized_model_name>/<run.run_name>`.
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
    return (
        Path(output_root).expanduser() / sanitize_for_path(model_name) / run_name
    ).resolve()


def _make_compute_metrics():
    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = np.asarray(logits).reshape(-1).astype(np.float32)
        labels = np.asarray(labels).reshape(-1).astype(np.float32)
        if labels.size == 0:
            return {"accuracy": 0.0}
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= 0.5).astype(np.float32)
        acc = float((preds == labels).mean())
        tp = float(((preds == 1) & (labels == 1)).sum())
        fp = float(((preds == 1) & (labels == 0)).sum())
        fn = float(((preds == 0) & (labels == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        pos_mean = float(probs[labels == 1].mean()) if (labels == 1).any() else 0.0
        neg_mean = float(probs[labels == 0].mean()) if (labels == 0).any() else 0.0
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pos_prob_mean": pos_mean,
            "neg_prob_mean": neg_mean,
            "prob_gap": pos_mean - neg_mean,
        }

    return compute_metrics


class HclTrainer(Trainer):
    """Saves the LoRA adapter (or merged backbone) plus the small HCL head.

    Trainer's default `_save` falls back to `torch.save(model.state_dict())`
    for non-PreTrainedModel modules, which would dump the full 8B-param base
    every checkpoint. We override it to:

      1. delegate to the backbone's `save_pretrained` (writes only the LoRA
         adapter when peft-wrapped), so users can load the adapter directly;
      2. write the temperature/bias/projection head to `hcl_head.pt`;
      3. write a `pytorch_model.bin` containing **only the trainable
         parameters** (LoRA + head) so Trainer's standard
         `_load_from_checkpoint` can resume training. The frozen base
         weights are reconstructed from `model.name` (+ optional Phase 1
         adapter merge) at startup, and `model.load_state_dict(..., strict=False)`
         leaves them untouched.
    """

    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model = self.model
        backbone = getattr(model, "backbone", None)
        if backbone is not None and hasattr(backbone, "save_pretrained"):
            backbone.save_pretrained(output_dir)

        head_state = {
            "log_temperature": model.log_temperature.detach().cpu(),
            "bias": model.bias.detach().cpu(),
            "pooling": getattr(model, "pooling", "last_token"),
        }
        if isinstance(model.proj, torch.nn.Linear):
            head_state["proj_state_dict"] = {
                k: v.detach().cpu() for k, v in model.proj.state_dict().items()
            }
        torch.save(head_state, os.path.join(output_dir, "hcl_head.pt"))

        # Trainable-only state dict: drives Trainer.resume_from_checkpoint
        # without writing the frozen 8B base every time.
        trainable = {
            name: param.detach().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        torch.save(trainable, os.path.join(output_dir, "pytorch_model.bin"))

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        tok = getattr(self, "tokenizer", None) or getattr(self, "processing_class", None)
        if tok is not None:
            tok.save_pretrained(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="LP-HCL CPT trainer (Phase 2)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Ad-hoc overrides: e.g. training.max_steps=50 data.max_seq_length=512",
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

    set_seed(int(run_cfg.get("seed", 42)))

    tokenizer, num_added = load_tokenizer(model_cfg)
    raw_datasets = load_pair_splits(data_cfg)
    raw_datasets = subsample_pair_splits(
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
    tokenized = tokenize_pairs(raw_datasets, tokenizer, data_cfg)

    model: HclPairModel = build_hcl_model(model_cfg, num_added)

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
    data_collator = HclPairCollator(pad_token_id=tokenizer.pad_token_id)

    eval_callback = EpochEndEvalCallback(
        trainer_ref=[],
        train_eval_dataset=(
            tokenized.get("train")
            if bool(run_cfg.get("train_eval_at_epoch_end", False))
            else None
        ),
        test_dataset=tokenized.get("test"),
    )
    metrics_cb = MetricsJsonlCallback(metrics_path=output_dir / "metrics.jsonl")

    trainer_kwargs: Dict[str, Any] = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=data_collator,
        callbacks=[metrics_cb, eval_callback],
        compute_metrics=_make_compute_metrics(),
    )
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = HclTrainer(**trainer_kwargs)
    eval_callback.trainer_ref.append(trainer)

    resume = resolve_resume(run_cfg, output_dir)
    LOGGER.info("Starting trainer.train(resume=%s)", resume)
    train_result = trainer.train(resume_from_checkpoint=resume)
    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()
    trainer.save_model(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))

    if "validation" in tokenized:
        LOGGER.info("Final validation-split eval")
        val_metrics = trainer.evaluate(
            eval_dataset=tokenized["validation"], metric_key_prefix="validation"
        )
        trainer.log_metrics("validation", val_metrics)
        trainer.save_metrics("validation", val_metrics)

    if "test" in tokenized:
        LOGGER.info("Final test-split eval")
        test_metrics = trainer.evaluate(
            eval_dataset=tokenized["test"], metric_key_prefix="test"
        )
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)

    LOGGER.info("Done. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
