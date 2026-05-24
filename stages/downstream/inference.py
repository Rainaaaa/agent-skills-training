#!/usr/bin/env python3
"""Batch inference + classification metrics for SFT-trained models.

Loads a (possibly LoRA-adapted) causal LM, walks a test parquet, renders
each row through the same prompt template the SFT trainer used, runs
constrained decoding to one of the response vocab tokens, and reports
accuracy / precision / recall / F1 against the gold label.

Designed to work with **any** registered SFT task, not just the two
built-ins, so a new task only needs its own `tasks/<name>.py` module —
no changes to this file.

Run:

    python -m stages.downstream.inference \\
        --config stages/downstream/configs/inference_example.yaml

Config knobs:

    data.task:        which task module to load (must match SFT training)
    data.test_file:   parquet or jsonl with the input + gold label columns
    data.max_rows:    cap for smoke tests
    model.name / model.model_name_or_path: backbone
    model.adapter_path: trained LoRA dir (e.g. .../final_model)
    inference.batch_size: 8
    inference.max_new_tokens: 8     # plenty for one-word answers
    inference.constrain_vocab: true # only sample from task.response_vocab
    output.predictions_jsonl: where to write per-row predictions
    output.metrics_json:      where to write the summary
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from common.config import load_config
from common.logging_utils import configure_logging
from common.modeling import load_causal_lm, load_tokenizer
from common.registry import resolve_model_path, sanitize_for_path

from stages.downstream.sft.data import load_splits, _format_one
from stages.downstream.sft.tasks import SFTTask, load_task


LOGGER = logging.getLogger("downstream.inference")


def _render_one_no_sys(row: Dict[str, Any], task: SFTTask) -> Tuple[str, str]:
    """Like sft.data._format_one but WITHOUT prepending task.system_prompt.

    Used to build few-shot exemplar blocks where the system prompt should
    appear once at the very top, not duplicated per exemplar.
    """
    response = task.response_for(row)
    if response not in task.response_vocab:
        raise ValueError(
            f"Task '{task.name}' produced response '{response}' which is "
            f"not in response_vocab={task.response_vocab}"
        )
    prompt = task.prompt_template.format(**{
        **{"skill_text": row.get(task.input_column, "")},
        **row,
    })
    return prompt, response


def _load_fewshot_exemplars(
    exemplar_file: str, task: SFTTask, k: int, seed: int = 42,
) -> List[Dict[str, Any]]:
    """Sample k exemplars from `exemplar_file`, balanced 50/50 across the
    task's two response classes (rounding to ceil for the positive class).

    Returns a list of plain dict rows in shuffled order so the positive /
    negative aren't all clustered at the start of the few-shot block.
    """
    if k <= 0 or not exemplar_file:
        return []
    if not Path(exemplar_file).exists():
        raise FileNotFoundError(f"exemplar_file does not exist: {exemplar_file}")

    from datasets import load_dataset
    loader = "parquet" if exemplar_file.lower().endswith(".parquet") else "json"
    ds = load_dataset(loader, data_files={"train": exemplar_file})["train"]

    pos_label = task.response_vocab[0]
    pos_idx, neg_idx = [], []
    for i in range(len(ds)):
        row = ds[i]
        try:
            r = task.response_for(row)
        except Exception:
            continue
        if r not in task.response_vocab:
            continue
        (pos_idx if r == pos_label else neg_idx).append(i)

    rng = random.Random(seed)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    k_pos = (k + 1) // 2     # ceil(k/2)
    k_neg = k - k_pos
    if len(pos_idx) < k_pos or len(neg_idx) < k_neg:
        LOGGER.warning(
            "Few-shot exemplar pool short: pos=%d (need %d), neg=%d (need %d)",
            len(pos_idx), k_pos, len(neg_idx), k_neg,
        )
        k_pos = min(k_pos, len(pos_idx))
        k_neg = min(k_neg, len(neg_idx))

    picked_idx = pos_idx[:k_pos] + neg_idx[:k_neg]
    rng.shuffle(picked_idx)
    rows = [ds[i] for i in picked_idx]
    LOGGER.info(
        "[fewshot] loaded k=%d exemplars from %s (pos=%d, neg=%d, seed=%d)",
        len(rows), exemplar_file, k_pos, k_neg, seed,
    )
    return rows


def _build_fewshot_prefix(exemplars: List[Dict[str, Any]], task: SFTTask) -> str:
    """Render k (prompt, response) pairs joined by blank lines.

    Each pair is `<prompt_template_filled> <response>`. Matches the
    response_suffix=" " convention used by SFT training (sft.data line 103),
    so a model that saw `<prompt><space><response>` during SFT sees the
    same token boundaries here.
    """
    if not exemplars:
        return ""
    parts = []
    for row in exemplars:
        prompt, resp = _render_one_no_sys(row, task)
        parts.append(f"{prompt} {resp}")
    return "\n\n".join(parts) + "\n\n"


def _format_one_with_fewshot(
    row: Dict[str, Any], task: SFTTask, fewshot_prefix: str,
) -> Tuple[str, str]:
    """Compose the final test prompt: `<system_prompt>\\n\\n<exemplars><test_prompt>`.

    system_prompt appears exactly once (at the top), so a 5-shot run doesn't
    repeat the system header 6 times.
    """
    test_prompt, gold = _render_one_no_sys(row, task)
    if task.system_prompt:
        return f"{task.system_prompt}\n\n{fewshot_prefix}{test_prompt}", gold
    return f"{fewshot_prefix}{test_prompt}", gold


def _derive_output_dir(run_cfg: Dict[str, Any], model_name: Optional[str], task_name: str) -> Path:
    if run_cfg.get("output_dir"):
        return Path(run_cfg["output_dir"]).expanduser().resolve()
    output_root = run_cfg.get("output_root") or "./outputs"
    run_name = run_cfg.get("run_name") or f"infer_{task_name}"
    return (Path(output_root).expanduser()
            / sanitize_for_path(model_name or "unknown")
            / f"infer_{task_name}__{run_name}").resolve()


def _maybe_attach_adapter(model, model_cfg: Dict[str, Any]):
    """Load chained adapters into a base model for inference.

    Convention (matches stages/downstream/sft/train.py build order):
      - `model.phase1_adapter_path` (CPT LoRA)  -> load and MERGE
      - `model.phase2_adapter_path` (HCL LoRA)  -> load and MERGE
      - `model.adapter_path`        (active)    -> load and ATTACH (keep as LoRA)

    Merging snaps the adapter into the base weights so subsequent adapters
    don't see PEFT-wrapping conflicts. The final `adapter_path` is left
    attached because zero-shot / few-shot / SFT inference each want a
    distinct active adapter (or none, for raw-base zero-shot).
    """
    paths_to_merge = [
        (key, model_cfg.get(key))
        for key in ("phase1_adapter_path", "phase2_adapter_path")
        if model_cfg.get(key)
    ]
    adapter_path = model_cfg.get("adapter_path") or model_cfg.get("sft_adapter_path")

    if not paths_to_merge and not adapter_path:
        return model

    try:
        from peft import PeftModel  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "peft is required when any of phase1/phase2/adapter_path is set"
        ) from exc

    for key, path in paths_to_merge:
        LOGGER.info("Loading %s from %s (will merge into base)", key, path)
        model = PeftModel.from_pretrained(model, path, is_trainable=False)
        model = model.merge_and_unload()

    if adapter_path:
        LOGGER.info("Attaching adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    return model


def _score_answer(
    model, tokenizer, prompt: str, response_vocab: List[str],
    constrain_vocab: bool, max_new_tokens: int, device,
) -> Tuple[str, Dict[str, float]]:
    """Return (predicted_response, per_class_logprob).

    With `constrain_vocab=True` we score each candidate response by its
    log-prob given the prompt and pick argmax. Robust for one-word
    answers; avoids decoding noise.
    With `constrain_vocab=False` we greedy-decode and post-match the
    output to the closest vocab item (substring match).
    """
    prompt_ids = tokenizer(prompt + " ", return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    if constrain_vocab:
        scores: Dict[str, float] = {}
        with torch.no_grad():
            for resp in response_vocab:
                resp_ids = tokenizer(resp, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
                full = torch.cat([prompt_ids, resp_ids], dim=1)
                out = model(full)
                logits = out.logits  # [1, T, V]
                # log-likelihood of resp_ids conditioned on prompt
                log_probs = torch.log_softmax(logits[0, prompt_ids.size(1) - 1 : -1, :], dim=-1)
                idx = resp_ids[0]
                ll = log_probs.gather(-1, idx.unsqueeze(-1)).squeeze(-1).sum().item()
                scores[resp] = ll
        predicted = max(scores, key=scores.get)
        return predicted, scores

    # Free-form generation path.
    with torch.no_grad():
        out = model.generate(
            prompt_ids, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    text = tokenizer.decode(out[0, prompt_ids.size(1):], skip_special_tokens=True).strip().lower()
    for resp in response_vocab:
        if resp in text:
            return resp, {resp: 0.0}
    return text, {}


def _binary_metrics(rows: List[Dict[str, Any]], positive_label: str) -> Dict[str, float]:
    tp = sum(1 for r in rows if r["pred"] == positive_label and r["gold"] == positive_label)
    tn = sum(1 for r in rows if r["pred"] != positive_label and r["gold"] != positive_label)
    fp = sum(1 for r in rows if r["pred"] == positive_label and r["gold"] != positive_label)
    fn = sum(1 for r in rows if r["pred"] != positive_label and r["gold"] == positive_label)
    n = tp + tn + fp + fn
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    config = load_config(args.config, args.overrides,
                         required_sections=("run", "data", "model", "inference", "output"))
    run_cfg = config["run"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    inference_cfg = config.get("inference", {})
    output_cfg = config.get("output", {})

    task_name = data_cfg.get("task")
    if not task_name:
        raise ValueError("data.task is required.")
    task = load_task(task_name)

    model_path, model_name = resolve_model_path(model_cfg)
    output_dir = _derive_output_dir(run_cfg, model_name, task.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir, run_cfg.get("log_level", "INFO"))
    LOGGER.info("task=%s model=%s output=%s", task.name, model_path, output_dir)

    tokenizer, num_added = load_tokenizer(model_cfg)
    model = load_causal_lm(model_cfg, num_added)
    model = _maybe_attach_adapter(model, model_cfg)
    model.eval()
    # load_causal_lm leaves an unquantized model on CPU (no device_map set).
    # The Trainer in pretraining/hcl moves it for us; inference has to do it
    # itself or the 8B forward passes run on CPU and take minutes per row.
    if torch.cuda.is_available() and next(model.parameters()).device.type == "cpu":
        model = model.to("cuda")
    device = next(model.parameters()).device
    LOGGER.info("[infer] model on device: %s", device)

    ds = load_splits(data_cfg)
    # Only the test split is used; if it's missing, fall back to validation
    # so the user can smoke-test with the same config they used for SFT.
    split_name = "test" if "test" in ds else "validation"
    test = ds[split_name]
    max_rows = int(data_cfg.get("max_rows", 0))
    if max_rows:
        test = test.select(range(min(len(test), max_rows)))
    LOGGER.info("[infer] %d rows from split=%s", len(test), split_name)

    max_new_tokens = int(inference_cfg.get("max_new_tokens", 8))
    constrain_vocab = bool(inference_cfg.get("constrain_vocab", True))

    # Few-shot config — n_shot=0 keeps the original zero-shot behavior.
    n_shot = int(inference_cfg.get("n_shot", 0))
    exemplar_seed = int(inference_cfg.get("exemplar_seed", 42))
    exemplar_file = (
        data_cfg.get("exemplar_file")
        or data_cfg.get("train_file")
        or ""
    )
    fewshot_exemplars: List[Dict[str, Any]] = []
    fewshot_prefix = ""
    if n_shot > 0:
        fewshot_exemplars = _load_fewshot_exemplars(
            exemplar_file, task, n_shot, exemplar_seed,
        )
        fewshot_prefix = _build_fewshot_prefix(fewshot_exemplars, task)

    predictions_path = Path(
        output_cfg.get("predictions_jsonl") or output_dir / f"predictions_{task.name}.jsonl"
    )
    metrics_path = Path(
        output_cfg.get("metrics_json") or output_dir / f"metrics_{task.name}.json"
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    rows_out: List[Dict[str, Any]] = []
    t0 = time.time()
    with predictions_path.open("w", encoding="utf-8") as fout:
        for i in range(len(test)):
            row = test[i]
            try:
                if n_shot > 0:
                    prompt, gold = _format_one_with_fewshot(
                        row, task, fewshot_prefix,
                    )
                else:
                    prompt, gold = _format_one(row, task)
            except ValueError:
                continue
            pred, scores = _score_answer(
                model, tokenizer, prompt,
                response_vocab=task.response_vocab,
                constrain_vocab=constrain_vocab,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            rec = {
                "row_index": i,
                "skill_id":  row.get("skill_id"),
                "gold":      gold,
                "pred":      pred,
                "scores":    scores,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows_out.append({"pred": pred, "gold": gold})
            if (i + 1) % 50 == 0:
                LOGGER.info("[infer] %d/%d elapsed=%.1fs", i + 1, len(test), time.time() - t0)

    # Binary metrics, treating response_vocab[0] as the positive class.
    positive_label = task.response_vocab[0]
    metrics = _binary_metrics(rows_out, positive_label)
    metrics["positive_label"] = positive_label
    metrics["task"] = task.name
    metrics["model"] = model_name
    metrics["adapter_path"] = model_cfg.get("adapter_path") or ""
    metrics["constrain_vocab"] = constrain_vocab
    metrics["split"] = split_name
    metrics["elapsed_sec"] = round(time.time() - t0, 2)
    metrics["n_shot"] = n_shot
    if n_shot > 0:
        metrics["exemplar_file"] = exemplar_file
        metrics["exemplar_seed"] = exemplar_seed
        metrics["n_exemplars_used"] = len(fewshot_exemplars)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, sort_keys=True)
    LOGGER.info("[infer] metrics: %s", metrics)
    LOGGER.info("[infer] predictions: %s", predictions_path)
    LOGGER.info("[infer] metrics_json: %s", metrics_path)


if __name__ == "__main__":
    main()
