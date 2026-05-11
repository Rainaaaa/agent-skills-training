"""Collator for Phase 2 / LP-HCL pair batches.

Pads anchors and pairs to their own per-batch max length (right-padding) and
stacks them as `(anchor_*, pair_*, labels, [pair_kind_id, anchor_stage_id,
pair_stage_id])`. The model encodes each side separately then computes a pair
score, so we don't concatenate anchor and pair into a single sequence.
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class HclPairCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        def pad(items, pad_value):
            max_len = max(len(x) for x in items)
            return [list(x) + [pad_value] * (max_len - len(x)) for x in items]

        anchor_ids = pad([f["anchor_input_ids"] for f in features], self.pad_token_id)
        anchor_mask = pad([f["anchor_attention_mask"] for f in features], 0)
        pair_ids = pad([f["pair_input_ids"] for f in features], self.pad_token_id)
        pair_mask = pad([f["pair_attention_mask"] for f in features], 0)

        batch: Dict[str, torch.Tensor] = {
            "anchor_input_ids": torch.tensor(anchor_ids, dtype=torch.long),
            "anchor_attention_mask": torch.tensor(anchor_mask, dtype=torch.long),
            "pair_input_ids": torch.tensor(pair_ids, dtype=torch.long),
            "pair_attention_mask": torch.tensor(pair_mask, dtype=torch.long),
            "labels": torch.tensor([f["labels"] for f in features], dtype=torch.float),
        }
        for opt_key in ("pair_kind_id", "anchor_stage_id", "pair_stage_id"):
            if opt_key in features[0]:
                batch[opt_key] = torch.tensor(
                    [f[opt_key] for f in features], dtype=torch.long
                )
        return batch
