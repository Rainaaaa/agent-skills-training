"""Data split loading, fractional subsampling, and token packing for CPT."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import DatasetDict, load_dataset


LOGGER = logging.getLogger(__name__)


def _file_loader_name(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix in {".json", ".jsonl"}:
        return "json"
    raise ValueError(f"Unsupported data file extension: {path}")


def load_splits(data_cfg: Dict[str, Any]) -> DatasetDict:
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
    LOGGER.info("Loaded splits: %s", {k: len(v) for k, v in dataset.items()})
    return dataset


def subsample_splits(
    dataset: DatasetDict,
    max_rows: Dict[str, Optional[int]],
    fractions: Dict[str, Optional[float]],
    seed: int = 42,
) -> DatasetDict:
    """Apply per-split row cap and/or fractional subsample; smaller keep-size wins."""
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


def compute_corpus_stats(dataset: DatasetDict, text_column: str = "text") -> Dict[str, Dict[str, int]]:
    """Count raw bytes / whitespace-words / chars per split, used for byte- and word-level PPL.

    Runs a single pass per split and returns a dict like:
      {"train": {"chars": ..., "bytes": ..., "words": ..., "rows": ...}, ...}
    """
    stats: Dict[str, Dict[str, int]] = {}
    for split, ds in dataset.items():
        chars = bytes_total = words = 0
        for text in ds[text_column]:
            if not text:
                continue
            s = str(text)
            chars += len(s)
            bytes_total += len(s.encode("utf-8"))
            words += len(s.split())
        stats[split] = {
            "chars": chars,
            "bytes": bytes_total,
            "words": words,
            "rows": len(ds),
        }
        LOGGER.info(
            "Corpus stats[%s]: rows=%d chars=%d bytes=%d words=%d",
            split, len(ds), chars, bytes_total, words,
        )
    return stats


def tokenize_and_pack(
    dataset: DatasetDict,
    tokenizer,
    data_cfg: Dict[str, Any],
) -> DatasetDict:
    text_column = data_cfg.get("text_column", "text")
    max_seq_length = int(data_cfg.get("max_seq_length", 2048))
    line_by_line = bool(data_cfg.get("line_by_line", False))
    num_proc = int(data_cfg.get("preprocessing_num_workers", 8))

    first_split = next(iter(dataset.values()))
    if text_column not in first_split.column_names:
        raise ValueError(
            f"text_column='{text_column}' not found. Columns: {first_split.column_names}"
        )

    if line_by_line:
        def tokenize_row(batch):
            return tokenizer(
                batch[text_column],
                truncation=True,
                max_length=max_seq_length,
                return_attention_mask=True,
            )

        columns_to_drop = first_split.column_names
        return dataset.map(
            tokenize_row,
            batched=True,
            num_proc=num_proc,
            remove_columns=columns_to_drop,
            desc="tokenize (line_by_line)",
        )

    def tokenize_only(batch):
        return tokenizer(batch[text_column], add_special_tokens=False)

    columns_to_drop = first_split.column_names
    tokenized = dataset.map(
        tokenize_only,
        batched=True,
        num_proc=num_proc,
        remove_columns=columns_to_drop,
        desc="tokenize",
    )

    eos = tokenizer.eos_token_id

    def pack(batch):
        concatenated: List[int] = []
        for ids in batch["input_ids"]:
            concatenated.extend(ids)
            if eos is not None:
                concatenated.append(eos)
        total_length = (len(concatenated) // max_seq_length) * max_seq_length
        concatenated = concatenated[:total_length]
        blocks = [
            concatenated[i : i + max_seq_length]
            for i in range(0, total_length, max_seq_length)
        ]
        return {
            "input_ids": blocks,
            "attention_mask": [[1] * max_seq_length for _ in blocks],
            "labels": [list(b) for b in blocks],
        }

    # Drop *all* upstream columns before packing — some tokenizers emit extras
    # like `token_type_ids` whose row count would mismatch the packed-block count.
    first_key = next(iter(tokenized))
    remove_cols = list(tokenized[first_key].column_names)
    packed = tokenized.map(
        pack,
        batched=True,
        num_proc=num_proc,
        remove_columns=remove_cols,
        desc=f"pack blocks of {max_seq_length}",
    )
    LOGGER.info("Packed block counts: %s", {k: len(v) for k, v in packed.items()})
    return packed
