"""Tokenizer/model loading + LoRA / QLoRA wrapping shared across phases."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common.registry import resolve_model_path


LOGGER = logging.getLogger(__name__)


def resolve_local_or_hub(path_or_id: str) -> Tuple[str, bool]:
    candidate = Path(path_or_id).expanduser()
    if candidate.exists():
        return str(candidate), True
    return path_or_id, False


def torch_dtype(spec: Optional[str]) -> Optional[torch.dtype]:
    if spec is None:
        return None
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = spec.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported torch_dtype={spec}")
    return mapping[key]


def load_tokenizer(model_cfg: Dict[str, Any]):
    # If only `model.name` is set (no explicit model_name_or_path), resolve via
    # the registry. The tokenizer falls back to the same path.
    if not model_cfg.get("model_name_or_path") and model_cfg.get("name"):
        path, _ = resolve_model_path(model_cfg)
        model_cfg = dict(model_cfg)
        model_cfg["model_name_or_path"] = path
    tokenizer_ref = model_cfg.get("tokenizer_name_or_path") or model_cfg["model_name_or_path"]
    resolved, is_local = resolve_local_or_hub(tokenizer_ref)
    LOGGER.info("Loading tokenizer from %s (local=%s)", resolved, is_local)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        use_fast=True,
    )
    added = 0
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            LOGGER.info("Set pad_token = eos_token (%s)", tokenizer.eos_token)
        else:
            added = tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            LOGGER.info("Added new pad token <|pad|> (added=%d)", added)
    return tokenizer, added


def _build_bnb_config(quant_cfg: Dict[str, Any]) -> Optional[BitsAndBytesConfig]:
    """Translate a `model.quantization` block into a BitsAndBytesConfig.

    Supported keys:
      - load_in_4bit: bool   (default True if any quantization block is present)
      - load_in_8bit: bool   (mutually exclusive with 4-bit)
      - bnb_4bit_compute_dtype: bfloat16 | float16 | float32 (default bfloat16)
      - bnb_4bit_use_double_quant: bool (default True)
      - bnb_4bit_quant_type: nf4 | fp4 (default nf4)
    """
    if not quant_cfg:
        return None
    if quant_cfg.get("load_in_8bit"):
        return BitsAndBytesConfig(load_in_8bit=True)
    if not quant_cfg.get("load_in_4bit", True):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch_dtype(quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=bool(quant_cfg.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_quant_type=str(quant_cfg.get("bnb_4bit_quant_type", "nf4")),
    )


def load_causal_lm(model_cfg: Dict[str, Any], num_added_tokens: int):
    if not model_cfg.get("model_name_or_path") and model_cfg.get("name"):
        path, _ = resolve_model_path(model_cfg)
        model_cfg = dict(model_cfg)
        model_cfg["model_name_or_path"] = path
    ref = model_cfg["model_name_or_path"]
    resolved, is_local = resolve_local_or_hub(ref)
    LOGGER.info("Loading model from %s (local=%s)", resolved, is_local)

    kwargs: Dict[str, Any] = {
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
    }
    dtype = torch_dtype(model_cfg.get("torch_dtype"))
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    attn = model_cfg.get("attn_implementation")
    if attn:
        kwargs["attn_implementation"] = attn

    bnb = _build_bnb_config(model_cfg.get("quantization") or {})
    if bnb is not None:
        kwargs["quantization_config"] = bnb
        # In DDP each rank owns its own GPU and needs its own model copy.
        # `device_map="auto"` would shard across ranks instead. Default to
        # `{"": local_rank}` when WORLD_SIZE > 1; allow explicit override.
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            default_dm: Any = {"": local_rank}
        else:
            default_dm = "auto"
        kwargs["device_map"] = model_cfg.get("device_map", default_dm)
        LOGGER.info(
            "Quantization enabled: %s; device_map=%s; world_size=%d",
            "8-bit" if bnb.load_in_8bit else f"4-bit ({bnb.bnb_4bit_quant_type}, double_quant={bnb.bnb_4bit_use_double_quant})",
            kwargs["device_map"],
            world_size,
        )

    model = AutoModelForCausalLM.from_pretrained(resolved, **kwargs)
    if num_added_tokens:
        new_size = model.get_input_embeddings().weight.shape[0] + num_added_tokens
        model.resize_token_embeddings(new_size)
    return model


def maybe_wrap_lora(model, model_cfg: Dict[str, Any]):
    if not bool(model_cfg.get("use_lora", True)):
        LOGGER.info("LoRA disabled; training all parameters")
        return model

    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        LOGGER.info("k-bit quantized base detected; calling prepare_model_for_kbit_training")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", True)),
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    lora_cfg = model_cfg.get("lora", {}) or {}
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("alpha", 16)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias=str(lora_cfg.get("bias", "none")),
        task_type="CAUSAL_LM",
        target_modules=list(lora_cfg.get("target_modules", ["q_proj", "v_proj"])),
    )
    model = get_peft_model(model, peft_config)
    trainable = total = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    LOGGER.info(
        "LoRA attached: trainable=%s (%.4f%% of %s)",
        f"{trainable:,}",
        100.0 * trainable / max(total, 1),
        f"{total:,}",
    )
    return model
