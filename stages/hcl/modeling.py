"""LP-HCL model: decoder-only LM as encoder + contrastive pair-scoring head.

The wrapper keeps the underlying CausalLM intact (tokenizer, hidden size,
attention impl, LoRA wrapping all unchanged) and adds:

  - a pooling step over the last hidden state (`last_token` or `mean`)
  - an optional projection (defaults to identity)
  - a pair-discriminative score: cos(anchor, pair) / temperature + bias
  - a BCE loss against the pair label, optionally per-pair-kind weighted
  - an optional in-batch InfoNCE term for the positive subset

It also knows how to start from a Phase 1 LoRA adapter: `model.phase1_adapter_path`
points at a `final_model/` produced by full_cpt; we load it via
`PeftModel.from_pretrained` and merge it into the base before applying the
fresh Phase 2 LoRA.
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from common.modeling import load_causal_lm, maybe_wrap_lora


LOGGER = logging.getLogger(__name__)


def _last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Last non-pad token per row — assumes right-padding (the collator's default)."""
    last_idx = attention_mask.long().sum(dim=1) - 1
    last_idx = last_idx.clamp(min=0)
    batch_idx = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch_idx, last_idx]


def _mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
    summed = (hidden * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1.0)
    return summed / count


class HclPairModel(nn.Module):
    """Decoder-only encoder + contrastive head.

    Forward inputs (see HclPairCollator):
      - anchor_input_ids, anchor_attention_mask
      - pair_input_ids, pair_attention_mask
      - labels (float 0/1)
      - optional: pair_kind_id, anchor_stage_id, pair_stage_id  (ignored unless
        the loss uses them)
    """

    # HF Trainer's _issue_warnings_after_load (resume path) reads this attr
    # directly. PreTrainedModel defines it at the class level; an nn.Module
    # wrapper has to too, or Module.__getattr__ raises AttributeError on
    # resume_from_checkpoint.
    _keys_to_ignore_on_save = None

    def __init__(
        self,
        backbone: PreTrainedModel,
        hidden_size: int,
        pooling: str = "last_token",
        proj_dim: Optional[int] = None,
        init_temperature: float = 0.07,
        pair_kind_loss_weights: Optional[Tuple[float, float, float]] = None,
        ce_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if pooling not in {"last_token", "mean"}:
            raise ValueError(f"Unsupported pooling: {pooling}")
        self.backbone = backbone
        self.pooling = pooling
        if proj_dim and int(proj_dim) > 0:
            self.proj: nn.Module = nn.Linear(hidden_size, int(proj_dim), bias=False)
        else:
            self.proj = nn.Identity()
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(float(init_temperature)), dtype=torch.float32)
        )
        self.bias = nn.Parameter(torch.zeros(1))
        if pair_kind_loss_weights is not None:
            self.register_buffer(
                "pair_kind_weights",
                torch.tensor(list(pair_kind_loss_weights), dtype=torch.float32),
            )
        else:
            self.pair_kind_weights = None
        self.ce_loss_weight = float(ce_loss_weight)

    @property
    def config(self):
        return getattr(self.backbone, "config", None)

    @property
    def is_loaded_in_8bit(self) -> bool:
        return bool(getattr(self.backbone, "is_loaded_in_8bit", False))

    @property
    def is_loaded_in_4bit(self) -> bool:
        return bool(getattr(self.backbone, "is_loaded_in_4bit", False))

    def gradient_checkpointing_enable(self, *args, **kwargs):
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()

    def _encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        if self.pooling == "mean":
            pooled = _mean_pool(hidden, attention_mask)
        else:
            pooled = _last_token_pool(hidden, attention_mask)
        return self.proj(pooled)

    def forward(
        self,
        anchor_input_ids,
        anchor_attention_mask,
        pair_input_ids,
        pair_attention_mask,
        labels=None,
        pair_kind_id=None,
        anchor_stage_id=None,
        pair_stage_id=None,
        **kwargs,
    ):
        a = self._encode(anchor_input_ids, anchor_attention_mask)
        p = self._encode(pair_input_ids, pair_attention_mask)
        a_n = F.normalize(a.float(), dim=-1)
        p_n = F.normalize(p.float(), dim=-1)
        cos = (a_n * p_n).sum(dim=-1)
        temperature = self.log_temperature.exp().clamp(min=1e-3, max=10.0)
        logits = cos / temperature + self.bias

        loss = None
        if labels is not None:
            label_f = labels.to(logits.dtype)
            per_example = F.binary_cross_entropy_with_logits(
                logits, label_f, reduction="none"
            )
            if self.pair_kind_weights is not None and pair_kind_id is not None:
                kind = pair_kind_id.clone()
                kind[kind < 0] = 0
                w = self.pair_kind_weights.to(per_example.device)[kind]
                loss = (per_example * w).mean()
            else:
                loss = per_example.mean()

            if self.ce_loss_weight > 0.0:
                pos_mask = labels > 0.5
                if pos_mask.any():
                    sim_matrix = torch.matmul(a_n, p_n.t()) / temperature
                    pos_idx = torch.nonzero(pos_mask, as_tuple=False).squeeze(-1)
                    sim_pos = sim_matrix[pos_idx]
                    target = pos_idx
                    ce = F.cross_entropy(sim_pos, target, reduction="mean")
                    loss = loss + self.ce_loss_weight * ce

        return SequenceClassifierOutput(loss=loss, logits=logits)


def load_phase1_adapter_if_any(
    model: PreTrainedModel, model_cfg: Dict[str, Any]
) -> PreTrainedModel:
    """Attach a Phase 1 LoRA adapter and merge it into the base.

    The Phase 1 launcher writes its trained LoRA to `<run.output_dir>/final_model/`.
    Pass that path via `model.phase1_adapter_path` here. The adapter is loaded
    in `is_trainable=False` mode and folded into the base via
    `merge_and_unload()` so a fresh Phase 2 LoRA can be applied cleanly on top.
    """
    adapter_path = model_cfg.get("phase1_adapter_path")
    if not adapter_path:
        return model
    adapter_path = str(Path(adapter_path).expanduser().resolve())
    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"phase1_adapter_path not found: {adapter_path}")
    LOGGER.info("Loading Phase 1 LoRA adapter from %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)

    if not bool(model_cfg.get("merge_phase1_adapter", True)):
        raise ValueError(
            "merge_phase1_adapter=false is not supported in this build "
            "(Phase 2 expects to add a fresh LoRA on top of merged weights). "
            "Either keep the default (true) or omit phase1_adapter_path."
        )

    quantized = bool(getattr(model, "is_loaded_in_4bit", False)) or bool(
        getattr(model, "is_loaded_in_8bit", False)
    )
    if quantized:
        raise ValueError(
            "Cannot merge a Phase 1 LoRA adapter into a quantized base "
            "(merge_and_unload requires unquantized weights). Remove "
            "model.quantization or drop phase1_adapter_path."
        )
    LOGGER.info("Merging Phase 1 adapter into base weights (merge_and_unload)")
    return model.merge_and_unload()


def build_hcl_model(model_cfg: Dict[str, Any], num_added_tokens: int) -> HclPairModel:
    backbone = load_causal_lm(model_cfg, num_added_tokens)

    # Capture hidden_size before any peft / merge wrapping changes the type.
    hidden_size = getattr(backbone.config, "hidden_size", None)
    if hidden_size is None:
        raise RuntimeError(
            "Could not infer backbone hidden_size from model.config — "
            "the model class may not expose it."
        )

    backbone = load_phase1_adapter_if_any(backbone, model_cfg)
    if bool(model_cfg.get("gradient_checkpointing", True)):
        if hasattr(backbone, "enable_input_require_grads"):
            backbone.enable_input_require_grads()
        cfg = getattr(backbone, "config", None)
        if cfg is not None:
            cfg.use_cache = False
    backbone = maybe_wrap_lora(backbone, model_cfg)

    hcl_cfg = model_cfg.get("hcl", {}) or {}
    pair_kind_weights = hcl_cfg.get("pair_kind_weights")
    if pair_kind_weights is not None and len(pair_kind_weights) != 3:
        raise ValueError(
            "model.hcl.pair_kind_weights must be 3 floats: "
            "[positive, corrupted, swapped]"
        )

    return HclPairModel(
        backbone=backbone,
        hidden_size=int(hidden_size),
        pooling=str(hcl_cfg.get("pooling", "last_token")),
        proj_dim=hcl_cfg.get("proj_dim"),
        init_temperature=float(hcl_cfg.get("init_temperature", 0.07)),
        pair_kind_loss_weights=tuple(pair_kind_weights) if pair_kind_weights else None,
        ce_loss_weight=float(hcl_cfg.get("ce_loss_weight", 0.0)),
    )
