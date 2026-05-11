"""SFT data loading — turn raw rows into (input_ids, labels) tensors with
loss masked to the response span only.

Same conventions as `common/data.py` so dataset files (parquet / jsonl)
load identically across stages. The differences vs CPT:

  - Rows are NOT packed; one example per row (binary classification).
  - Loss is masked on the prompt span (labels = -100 on the prompt) so
    the model is only graded on its short answer.
  - The optional `data.balance_by_label` flag downsamples the majority
    class so the training set is 50/50 — useful for the small malicious
    set which would otherwise drown in 'safe' rows.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Tuple

from datasets import DatasetDict, load_dataset

from stages.downstream.sft.tasks import SFTTask


LOGGER = logging.getLogger("downstream.sft.data")


def _file_loader_name(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".parquet"):
        return "parquet"
    if lower.endswith(".jsonl") or lower.endswith(".json"):
        return "json"
    raise ValueError(f"Unsupported input file format: {path}")


def load_splits(data_cfg: Dict[str, Any]) -> DatasetDict:
    """Dispatches by extension; mirrors `common/data.load_splits`."""
    files: Dict[str, str] = {}
    for split, key in (("train", "train_file"), ("validation", "validation_file"), ("test", "test_file")):
        path = data_cfg.get(key)
        if path:
            files[split] = path
    if not files:
        raise ValueError("data.train_file / validation_file / test_file is required")
    loaders = {f: _file_loader_name(p) for f, p in files.items()}
    unique_loaders = set(loaders.values())
    if len(unique_loaders) != 1:
        raise ValueError(f"Mixed loaders across splits: {loaders}")
    return load_dataset(next(iter(unique_loaders)), data_files=files)


def _balance_split(ds, task: SFTTask, seed: int):
    """Downsample to equal positive/negative count."""
    pos_idx, neg_idx = [], []
    pos_label = task.response_vocab[0]  # arbitrarily the "positive" class
    for i, row in enumerate(ds):
        if task.response_for(row) == pos_label:
            pos_idx.append(i)
        else:
            neg_idx.append(i)
    if not pos_idx or not neg_idx:
        LOGGER.warning(
            "[sft.data] cannot balance split: pos=%d neg=%d (need both > 0)",
            len(pos_idx), len(neg_idx),
        )
        return ds
    rng = random.Random(seed)
    target = min(len(pos_idx), len(neg_idx))
    keep = rng.sample(pos_idx, target) + rng.sample(neg_idx, target)
    rng.shuffle(keep)
    LOGGER.info(
        "[sft.data] balanced split: pos %d→%d, neg %d→%d",
        len(pos_idx), target, len(neg_idx), target,
    )
    return ds.select(keep)


def _format_one(row: Dict[str, Any], task: SFTTask) -> Tuple[str, str]:
    """Render the (prompt, response) pair for a single row."""
    response = task.response_for(row)
    if response not in task.response_vocab:
        raise ValueError(
            f"Task '{task.name}' produced response '{response}' which is "
            f"not in response_vocab={task.response_vocab}; row keys: {list(row)[:8]}"
        )
    prompt = task.prompt_template.format(**{
        **{"skill_text": row.get(task.input_column, "")},
        **row,  # pass everything through so custom templates can reach other cols
    })
    if task.system_prompt:
        prompt = f"{task.system_prompt}\n\n{prompt}"
    return prompt, response


def tokenize_for_sft(
    dataset: DatasetDict,
    tokenizer,
    task: SFTTask,
    *,
    max_seq_length: int,
    response_suffix: str = " ",  # space between prompt and response
    drop_no_label: bool = True,
):
    """Tokenize each example as a single sequence, with labels masked on
    the prompt span so loss is computed only on the answer.

    Returns a tokenized DatasetDict ready for `Trainer`. Each row:
      input_ids: [<prompt> <space> <response> <eos>]
      labels:    [-100, …, -100,   ans_token_0, ans_token_1, …, eos]
    """
    eos = tokenizer.eos_token_id

    def _encode(batch):
        out_ids, out_labels, out_mask = [], [], []
        for i in range(len(next(iter(batch.values())))):
            row = {k: v[i] for k, v in batch.items()}
            try:
                prompt, response = _format_one(row, task)
            except ValueError:
                if drop_no_label:
                    continue
                raise
            prompt_ids = tokenizer(prompt + response_suffix, add_special_tokens=False)["input_ids"]
            resp_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
            if eos is not None:
                resp_ids = resp_ids + [eos]
            total = prompt_ids + resp_ids
            if len(total) > max_seq_length:
                # Right-truncate the prompt to keep the response intact.
                room = max_seq_length - len(resp_ids)
                if room <= 0:
                    continue
                prompt_ids = prompt_ids[:room]
                total = prompt_ids + resp_ids
            labels = [-100] * len(prompt_ids) + list(resp_ids)
            out_ids.append(total)
            out_labels.append(labels)
            out_mask.append([1] * len(total))
        return {"input_ids": out_ids, "labels": out_labels, "attention_mask": out_mask}

    return dataset.map(
        _encode,
        batched=True,
        remove_columns=dataset[next(iter(dataset))].column_names,
        desc=f"tokenize SFT [{task.name}]",
    )
