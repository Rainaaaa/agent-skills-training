#!/usr/bin/env python3
"""Aggregate the per-model artifacts written by `scripts/run_full_eval.sh`
into one comparison table covering both intrinsic (perplexity) and
extrinsic (alignment / maliciousness classification) eval.

Walks <runs_root> with the run-name conventions baked into the
orchestrator and emits:

  eval_summary.json   — full per-(model,phase,task) records
  eval_summary.md     — wide pivot table, easy to paste into a PR/review

Usage (inside the docker image, where pyarrow is available):

  docker compose run --rm --no-deps --entrypoint python pretraining \
      scripts/aggregate_eval.py /app/outputs/runs

Outside docker (host Python with json stdlib is enough):

  python3 scripts/aggregate_eval.py /path/to/runs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# Must match the NAME_* constants in scripts/run_full_eval.sh.
RUN_NAMES = {
    "baseline_pre":   "eval_baseline_pre_cpt",
    "baseline_post":  "eval_baseline_post_cpt",
    "zs_align_pre":   "eval_zs_misalignment_pre_cpt",
    "zs_align_post":  "eval_zs_misalignment_post_cpt",
    "zs_mal_pre":     "eval_zs_malicious_pre_cpt",
    "zs_mal_post":    "eval_zs_malicious_post_cpt",
    "sft_align":      "eval_sft_misalignment",
    "sft_mal":        "eval_sft_malicious",
}

# Subset of baseline_metrics.json keys worth surfacing.
BASELINE_KEYS = ["eval_ppl_token", "eval_ppl_byte", "eval_bits_per_byte", "eval_top1_acc"]

# Inference metrics_<task>.json keys we want.
EXTRINSIC_KEYS = ["accuracy", "precision", "recall", "f1", "n"]

# Map run-name → (file basename, schema label) so aggregator knows what to load.
ARTIFACTS = {
    "baseline_pre":  ("baseline_metrics.json",                       "intrinsic"),
    "baseline_post": ("baseline_metrics.json",                       "intrinsic"),
    "zs_align_pre":  ("metrics_misalignment_detection.json",         "extrinsic"),
    "zs_align_post": ("metrics_misalignment_detection.json",         "extrinsic"),
    "zs_mal_pre":    ("metrics_malicious_detection.json",            "extrinsic"),
    "zs_mal_post":   ("metrics_malicious_detection.json",            "extrinsic"),
    "sft_align":     ("metrics_misalignment_detection.json",         "extrinsic"),
    "sft_mal":       ("metrics_malicious_detection.json",            "extrinsic"),
}


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"warn: failed to read {path}: {exc}", file=sys.stderr)
        return None


def _extract(record: Dict[str, Any], kind: str) -> Dict[str, Any]:
    """Pull the subset of keys we care about, defaulting missing to None."""
    if kind == "intrinsic":
        metrics = record.get("metrics", record)
        return {k: metrics.get(k) for k in BASELINE_KEYS}
    return {k: record.get(k) for k in EXTRINSIC_KEYS}


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def collect(runs_root: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not runs_root.is_dir():
        sys.exit(f"runs_root not found or not a directory: {runs_root}")
    for model_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        model = model_dir.name
        row: Dict[str, Any] = {"model": model}
        for short, run_name in RUN_NAMES.items():
            fname, kind = ARTIFACTS[short]
            data = _load(model_dir / run_name / fname)
            row[short] = _extract(data, kind) if data else None
        out.append(row)
    return out


def write_json(rows: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.write_text(json.dumps(rows, indent=2, sort_keys=False))


# Columns surfaced in the markdown pivot table.
MD_COLUMNS = [
    ("ppl_pre",            "baseline_pre",   "eval_ppl_token"),
    ("ppl_post",           "baseline_post",  "eval_ppl_token"),
    ("bpb_pre",            "baseline_pre",   "eval_bits_per_byte"),
    ("bpb_post",           "baseline_post",  "eval_bits_per_byte"),
    ("zs_align_acc_pre",   "zs_align_pre",   "accuracy"),
    ("zs_align_acc_post",  "zs_align_post",  "accuracy"),
    ("zs_align_f1_pre",    "zs_align_pre",   "f1"),
    ("zs_align_f1_post",   "zs_align_post",  "f1"),
    ("zs_mal_acc_pre",     "zs_mal_pre",     "accuracy"),
    ("zs_mal_acc_post",    "zs_mal_post",    "accuracy"),
    ("zs_mal_f1_pre",      "zs_mal_pre",     "f1"),
    ("zs_mal_f1_post",     "zs_mal_post",    "f1"),
    ("sft_align_acc",      "sft_align",      "accuracy"),
    ("sft_align_f1",       "sft_align",      "f1"),
    ("sft_mal_acc",        "sft_mal",        "accuracy"),
    ("sft_mal_f1",         "sft_mal",        "f1"),
]


def write_markdown(rows: List[Dict[str, Any]], out_path: Path) -> None:
    headers = ["model"] + [c[0] for c in MD_COLUMNS]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        cells = [row["model"]]
        for _, short, key in MD_COLUMNS:
            cells.append(_fmt((row.get(short) or {}).get(key)))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Empty cells (`—`) = artifact missing on disk "
                 "(step failed, skipped, or not yet run).")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs_root", type=Path,
                    help="Directory containing per-model run subdirectories.")
    ap.add_argument("--json-out", type=Path, default=None,
                    help="Output JSON path (default: <runs_root>/eval_summary.json)")
    ap.add_argument("--md-out", type=Path, default=None,
                    help="Output Markdown path (default: <runs_root>/eval_summary.md)")
    args = ap.parse_args()

    rows = collect(args.runs_root)
    json_out = args.json_out or (args.runs_root / "eval_summary.json")
    md_out = args.md_out or (args.runs_root / "eval_summary.md")
    write_json(rows, json_out)
    write_markdown(rows, md_out)
    print(f"wrote {json_out}")
    print(f"wrote {md_out}")
    print()
    print(md_out.read_text())


if __name__ == "__main__":
    main()
