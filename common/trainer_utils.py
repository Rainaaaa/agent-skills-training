"""Callbacks, TrainingArguments builder, and checkpoint resume helper."""

import json
import logging
import math
import os
from importlib.util import find_spec
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint


LOGGER = logging.getLogger(__name__)


def maybe_disable_implicit_deepspeed(training_cfg: Dict[str, Any]) -> None:
    """
    `accelerate` unconditionally tries to import DeepSpeed during model unwrap if
    it detects the package on the environment. On clusters where DeepSpeed is
    installed but CUDA build metadata is incomplete, that import can crash even
    when the current training run is not using DeepSpeed at all.

    If the config does not explicitly request `training.deepspeed`, proactively
    turn off the implicit DeepSpeed path after detecting that `import deepspeed`
    is broken in the current runtime.
    """

    if training_cfg.get("deepspeed"):
        LOGGER.info("DeepSpeed explicitly configured; leaving DeepSpeed integration enabled.")
        return
    if find_spec("deepspeed") is None:
        return

    cuda_home = os.environ.get("CUDA_HOME")
    if not cuda_home or not Path(cuda_home).exists():
        LOGGER.warning(
            "DeepSpeed is installed but CUDA_HOME is unset or invalid and this run does not "
            "request DeepSpeed. Disabling implicit DeepSpeed detection."
        )
        try:
            import accelerate.utils.other as accelerate_other

            accelerate_other.is_deepspeed_available = lambda: False
        except Exception as exc:
            LOGGER.warning("Failed to patch accelerate DeepSpeed detection: %s", exc)

        try:
            import transformers.integrations.deepspeed as hf_deepspeed

            hf_deepspeed.is_deepspeed_available = lambda: False
        except Exception as exc:
            LOGGER.warning("Failed to patch transformers DeepSpeed detection: %s", exc)
        return

    try:
        import deepspeed  # noqa: F401
        return
    except Exception as exc:
        LOGGER.warning(
            "DeepSpeed is installed but unusable in this environment and this run does not "
            "request it. Disabling implicit DeepSpeed detection. Root cause: %s",
            exc,
        )

    try:
        import accelerate.utils.other as accelerate_other

        accelerate_other.is_deepspeed_available = lambda: False
    except Exception as exc:
        LOGGER.warning("Failed to patch accelerate DeepSpeed detection: %s", exc)

    try:
        import transformers.integrations.deepspeed as hf_deepspeed

        hf_deepspeed.is_deepspeed_available = lambda: False
    except Exception as exc:
        LOGGER.warning("Failed to patch transformers DeepSpeed detection: %s", exc)


@dataclass
class MetricsJsonlCallback(TrainerCallback):
    """Append every `trainer.log()` entry to a metrics.jsonl file, with perplexity."""

    metrics_path: Path

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not logs:
            return
        payload = dict(logs)
        payload["step"] = int(state.global_step)
        payload["epoch"] = float(state.epoch) if state.epoch is not None else None
        for key, value in list(payload.items()):
            if "loss" in key and isinstance(value, (int, float)) and value >= 0:
                try:
                    payload[f"{key}_ppl"] = float(math.exp(value))
                except OverflowError:
                    payload[f"{key}_ppl"] = float("inf")
        with open(self.metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")


@dataclass
class EpochEndEvalCallback(TrainerCallback):
    """At each epoch end, evaluate on the training split and the test split."""

    trainer_ref: List[Trainer] = field(default_factory=list)
    train_eval_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None

    def _eval_and_log(self, trainer: Trainer, dataset: Dataset, prefix: str, epoch):
        LOGGER.info("[epoch %s] %s-split eval", epoch, prefix)
        metrics = trainer.evaluate(eval_dataset=dataset, metric_key_prefix=prefix)
        loss_key = f"{prefix}_loss"
        if loss_key in metrics and metrics[loss_key] < 30:
            metrics[f"{prefix}_ppl"] = math.exp(metrics[loss_key])
        trainer.log_metrics(prefix, metrics)
        trainer.save_metrics(prefix, metrics)

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if not self.trainer_ref:
            return
        trainer = self.trainer_ref[0]
        if self.train_eval_dataset is not None:
            self._eval_and_log(trainer, self.train_eval_dataset, "train_eval", state.epoch)
        if self.test_dataset is not None:
            self._eval_and_log(trainer, self.test_dataset, "test", state.epoch)


def build_training_args(training_cfg: Dict[str, Any], output_dir: Path) -> TrainingArguments:
    args = dict(training_cfg)
    args.setdefault("output_dir", str(output_dir))
    args.setdefault("logging_dir", str(output_dir / "tb"))
    args.setdefault("report_to", ["tensorboard"])
    args.setdefault("num_train_epochs", 1)
    args.setdefault("per_device_train_batch_size", 1)
    args.setdefault("per_device_eval_batch_size", 1)
    args.setdefault("gradient_accumulation_steps", 8)
    args.setdefault("learning_rate", 2e-4)
    args.setdefault("lr_scheduler_type", "cosine")
    args.setdefault("warmup_ratio", 0.03)
    args.setdefault("logging_steps", 50)
    args.setdefault("save_strategy", "steps")
    args.setdefault("save_steps", 500)
    args.setdefault("save_total_limit", 3)
    # transformers >= 4.40 renamed evaluation_strategy -> eval_strategy
    args.setdefault("eval_strategy", args.pop("evaluation_strategy", "steps"))
    args.setdefault("eval_steps", 500)
    args.setdefault("bf16", True)
    args.setdefault("dataloader_num_workers", 2)
    args.setdefault("remove_unused_columns", False)
    args.setdefault("ddp_find_unused_parameters", False)
    args.setdefault("gradient_checkpointing", True)
    args.setdefault("gradient_checkpointing_kwargs", {"use_reentrant": False})
    return TrainingArguments(**args)


def resolve_resume(run_cfg: Dict[str, Any], output_dir: Path) -> Optional[str]:
    explicit = run_cfg.get("resume_from_checkpoint")
    if explicit:
        ckpt = Path(explicit)
        if not ckpt.exists():
            raise FileNotFoundError(f"resume_from_checkpoint not found: {ckpt}")
        LOGGER.info("Resuming from explicit checkpoint: %s", ckpt)
        return str(ckpt)
    if run_cfg.get("auto_resume", False) and output_dir.exists():
        last = get_last_checkpoint(str(output_dir))
        if last:
            LOGGER.info("Auto-resuming from latest checkpoint: %s", last)
            return last
        LOGGER.info("auto_resume=true but no checkpoint-* found in %s", output_dir)
    return None


def compute_eval_steps_from_fraction(
    fraction: float,
    n_train: int,
    per_device_batch: int,
    grad_accum: int,
    world_size: int = 1,
) -> int:
    """Translate `eval every X fraction of epoch` into concrete step counts."""
    effective_batch = max(1, per_device_batch * grad_accum * world_size)
    steps_per_epoch = max(1, math.ceil(n_train / effective_batch))
    return max(1, round(steps_per_epoch * float(fraction))), steps_per_epoch
