"""Phase 2 / LP-HCL data loading and tokenization.

Reads contrastive pair parquet/jsonl produced by `data_preparation.run_phase2`
and tokenizes anchor and pair texts independently. The collator (see
`lp_hcl.collator`) pads each side and stacks them into a single batch.

Pair-row schema produced by data_preparation:
  - anchor_text, pair_text   : tokenizable text
  - label (0/1)              : 1 = pair from same skill, 0 = different
  - pair_kind                : "positive" | "corrupted" | "swapped"
  - anchor_stage, pair_stage : "stage_name" | "stage_description" |
                               "stage_markdown" | "stage_files"
  - split, normalized_version, output_version : lineage
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import DatasetDict, load_dataset


LOGGER = logging.getLogger(__name__)

PAIR_KIND_TO_ID: Dict[str, int] = {"positive": 0, "corrupted": 1, "swapped": 2}
STAGE_TO_ID: Dict[str, int] = {
    "stage_name": 0,
    "stage_description": 1,
    "stage_markdown": 2,
    "stage_files": 3,
}


def _file_loader_name(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix in {".json", ".jsonl"}:
        return "json"
    raise ValueError(f"Unsupported data file extension: {path}")


def load_pair_splits(data_cfg: Dict[str, Any]) -> DatasetDict:
    files: Dict[str, str] = {}
    for split_name, key in (
        ("train", "train_file"),
        ("validation", "validation_file"),
        ("test", "test_file"),
    ):
        path = data_cfg.get(key)
        if not path:
            continue
        if not Path(path).exists():
            raise FileNotFoundError(f"{key}={path} does not exist")
        files[split_name] = path

    if "train" not in files:
        raise ValueError("data.train_file is required")

    loaders = {split: _file_loader_name(path) for split, path in files.items()}
    distinct = set(loaders.values())
    if len(distinct) != 1:
        raise ValueError(f"All split files must share a format; got {loaders}")

    dataset = load_dataset(next(iter(distinct)), data_files=files)
    LOGGER.info("Loaded pair splits: %s", {k: len(v) for k, v in dataset.items()})
    return dataset


def subsample_pair_splits(
    dataset: DatasetDict,
    max_rows: Dict[str, Optional[int]],
    fractions: Dict[str, Optional[float]],
    seed: int = 42,
) -> DatasetDict:
    """Per-split row cap and/or fractional subsample (smaller keep-size wins)."""
    for split, ds in dataset.items():
        n = len(ds)
        keep = n
        cap = max_rows.get(split)
        frac = fractions.get(split)
        if frac is not None and 0 < float(frac) < 1:
            keep = min(keep, max(1, int(n * float(frac))))
        if cap is not None and cap > 0:
            keep = min(keep, int(cap))
        if keep < n:
            LOGGER.info(
                "Subsampling %s: %d -> %d (cap=%s, fraction=%s)",
                split, n, keep, cap, frac,
            )
            dataset[split] = ds.shuffle(seed=seed).select(range(keep))
    return dataset


def tokenize_pairs(
    dataset: DatasetDict,
    tokenizer,
    data_cfg: Dict[str, Any],
) -> DatasetDict:
    anchor_col = data_cfg.get("anchor_column", "anchor_text")
    pair_col = data_cfg.get("pair_column", "pair_text")
    label_col = data_cfg.get("label_column", "label")
    pair_kind_col = data_cfg.get("pair_kind_column", "pair_kind")
    anchor_stage_col = data_cfg.get("anchor_stage_column", "anchor_stage")
    pair_stage_col = data_cfg.get("pair_stage_column", "pair_stage")
    max_seq_length = int(data_cfg.get("max_seq_length", 1024))
    num_proc = int(data_cfg.get("preprocessing_num_workers", 8))

    first_split = next(iter(dataset.values()))
    for col in (anchor_col, pair_col, label_col):
        if col not in first_split.column_names:
            raise ValueError(
                f"Required column '{col}' not in {first_split.column_names}"
            )

    has_kind = pair_kind_col in first_split.column_names
    has_anchor_stage = anchor_stage_col in first_split.column_names
    has_pair_stage = pair_stage_col in first_split.column_names

    def encode(batch):
        anchor = tokenizer(
            batch[anchor_col],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            add_special_tokens=True,
        )
        pair = tokenizer(
            batch[pair_col],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            add_special_tokens=True,
        )
        out = {
            "anchor_input_ids": anchor["input_ids"],
            "anchor_attention_mask": anchor["attention_mask"],
            "pair_input_ids": pair["input_ids"],
            "pair_attention_mask": pair["attention_mask"],
            "labels": [int(x) for x in batch[label_col]],
        }
        if has_kind:
            out["pair_kind_id"] = [
                PAIR_KIND_TO_ID.get(k, -1) for k in batch[pair_kind_col]
            ]
        if has_anchor_stage:
            out["anchor_stage_id"] = [
                STAGE_TO_ID.get(s, -1) for s in batch[anchor_stage_col]
            ]
        if has_pair_stage:
            out["pair_stage_id"] = [
                STAGE_TO_ID.get(s, -1) for s in batch[pair_stage_col]
            ]
        return out

    columns_to_drop = first_split.column_names
    encoded = dataset.map(
        encode,
        batched=True,
        num_proc=num_proc,
        remove_columns=columns_to_drop,
        desc="tokenize pairs",
    )
    LOGGER.info("Tokenized pair counts: %s", {k: len(v) for k, v in encoded.items()})
    return encoded
