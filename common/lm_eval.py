"""Causal-LM evaluation metrics.

Provides:

- `preprocess_logits_for_metrics(logits, labels)` — runs once per eval batch on
  GPU. Reduces full vocab logits to a small per-token feature vector before
  Trainer concatenates predictions across batches.

- `make_compute_metrics(corpus_stats, ...)` — builds a `compute_metrics` function
  that aggregates the per-token features into:
    * cross-entropy loss
    * token / word / byte / char level perplexity
    * top-1, top-5, top-10 next-token accuracy
    * top-p coverage at p=0.9 and p=0.95
    * mean true-token probability (a.k.a. confidence)

The byte/word/char perplexities are derived from the token-level loss using
corpus normalisation ratios computed once during data preparation:

    PPL_byte = exp(loss_per_token * tokens_per_byte)
    PPL_word = exp(loss_per_token * tokens_per_word)
"""

import math
from typing import Dict, Optional

import numpy as np
import torch


# Index layout for the per-token feature tensor that
# `preprocess_logits_for_metrics` returns. compute_metrics depends on this.
FEATURE_NAMES = [
    "valid",          # 0  — 1.0 if the position contributes (label != -100)
    "top1_correct",   # 1
    "top5_correct",   # 2
    "top10_correct",  # 3
    "in_topp_090",    # 4
    "in_topp_095",    # 5
    "prob_true",      # 6  — softmax probability of the true next token
    "neg_log_prob",   # 7  — -log p(true token); i.e. per-token CE loss
]
NUM_FEATURES = len(FEATURE_NAMES)


def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Reduce per-token logits/labels to a small feature vector.

    Returns shape `[B, T-1, NUM_FEATURES]` (float32 on GPU). Trainer then moves
    it to CPU and concatenates across batches.
    """
    # Some models return a (logits, ...) tuple via return_dict=False
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    # `logits` shape: [B, T, V]; `labels` shape: [B, T]. Apply the standard
    # causal-LM shift so position t predicts token t+1.
    pred = logits[..., :-1, :]                  # [B, T-1, V]
    targ = labels[..., 1:]                      # [B, T-1]
    valid = (targ != -100)
    # Replace ignored tokens with 0 so gather() doesn't crash.
    targ_safe = targ.clamp(min=0)

    # log_softmax — numerically stable, avoids materialising a separate softmax.
    log_probs = pred.float().log_softmax(dim=-1)                          # [B, T-1, V]
    log_p_true = log_probs.gather(-1, targ_safe.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
    p_true = log_p_true.exp()

    # Top-k correctness (single topk reused for k=1, 5, 10).
    _, top10_idx = log_probs.topk(10, dim=-1)                              # [B, T-1, 10]
    targ_exp = targ_safe.unsqueeze(-1)
    top1_correct = (top10_idx[..., 0:1] == targ_exp).any(-1)
    top5_correct = (top10_idx[..., :5] == targ_exp).any(-1)
    top10_correct = (top10_idx == targ_exp).any(-1)

    # Top-p coverage. mass_above_strict = sum of probs of tokens strictly more
    # likely than the true token. true token is in nucleus(p) iff that mass < p.
    mask = log_probs > log_p_true.unsqueeze(-1)
    mass_above = (log_probs.exp() * mask).sum(dim=-1)                      # [B, T-1]
    in_topp_090 = mass_above < 0.90
    in_topp_095 = mass_above < 0.95

    valid_f = valid.float()
    feats = torch.stack(
        [
            valid_f,
            top1_correct.float() * valid_f,
            top5_correct.float() * valid_f,
            top10_correct.float() * valid_f,
            in_topp_090.float() * valid_f,
            in_topp_095.float() * valid_f,
            p_true * valid_f,
            (-log_p_true) * valid_f,
        ],
        dim=-1,
    )
    return feats


def _safe_perplexity(loss: float) -> float:
    if loss is None or not math.isfinite(loss):
        return float("inf")
    if loss > 30:
        return float("inf")
    return float(math.exp(loss))


def make_compute_metrics(
    eval_corpus_stats: Optional[Dict[str, int]] = None,
    prefix: str = "eval",
):
    """Build a compute_metrics(eval_pred) callable.

    `eval_corpus_stats` is the per-split dict produced by
    `common.data.compute_corpus_stats`, e.g. `{"chars": ..., "bytes": ..., "words": ...}`.
    When provided, byte/word/char-level perplexities are added.
    """

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        # Trainer turned our GPU output into a numpy array on CPU.
        # Shape: [N_examples_in_eval, T-1, NUM_FEATURES] OR flattened across
        # examples. Handle both by reshaping to 2D.
        arr = np.asarray(preds)
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        elif arr.ndim != 2:
            raise ValueError(f"Unexpected predictions shape {arr.shape}")
        if arr.shape[-1] != NUM_FEATURES:
            raise ValueError(
                f"Expected last dim {NUM_FEATURES} matching FEATURE_NAMES, got {arr.shape[-1]}"
            )

        valid = arr[:, 0]
        n_tokens = float(valid.sum())
        if n_tokens <= 0:
            return {f"{prefix}_n_tokens": 0.0}

        sums = arr.sum(axis=0)
        # sums[i] / n_tokens is the masked mean of feature i.
        mean_top1 = float(sums[1]) / n_tokens
        mean_top5 = float(sums[2]) / n_tokens
        mean_top10 = float(sums[3]) / n_tokens
        mean_topp090 = float(sums[4]) / n_tokens
        mean_topp095 = float(sums[5]) / n_tokens
        mean_prob_true = float(sums[6]) / n_tokens
        mean_loss = float(sums[7]) / n_tokens

        metrics = {
            "loss": mean_loss,
            "ppl_token": _safe_perplexity(mean_loss),
            "top1_acc": mean_top1,
            "top5_acc": mean_top5,
            "top10_acc": mean_top10,
            "topp_0.90_coverage": mean_topp090,
            "topp_0.95_coverage": mean_topp095,
            "mean_prob_true": mean_prob_true,
            "mean_log_prob_true": -mean_loss,
            "n_tokens_evaluated": n_tokens,
        }

        if eval_corpus_stats:
            n_bytes = float(eval_corpus_stats.get("bytes", 0) or 0)
            n_words = float(eval_corpus_stats.get("words", 0) or 0)
            n_chars = float(eval_corpus_stats.get("chars", 0) or 0)
            if n_bytes > 0:
                loss_per_byte = mean_loss * n_tokens / n_bytes
                metrics["ppl_byte"] = _safe_perplexity(loss_per_byte)
                metrics["bits_per_byte"] = loss_per_byte / math.log(2)
            if n_words > 0:
                loss_per_word = mean_loss * n_tokens / n_words
                metrics["ppl_word"] = _safe_perplexity(loss_per_word)
            if n_chars > 0:
                loss_per_char = mean_loss * n_tokens / n_chars
                metrics["ppl_char"] = _safe_perplexity(loss_per_char)

        return metrics

    return compute_metrics
