#!/usr/bin/env python3
"""Aggregate per-model baseline_metrics.json files into one comparison table.

Usage:
  python aggregate.py <output_root>          # writes baseline_summary.{json,md}
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


COMPARABLE_KEYS = [
    "eval_loss",
    "eval_ppl_token",
    "eval_ppl_byte",
    "eval_ppl_word",
    "eval_ppl_char",
    "eval_bits_per_byte",
    "eval_top1_acc",
    "eval_top5_acc",
    "eval_top10_acc",
    "eval_topp_0.90_coverage",
    "eval_topp_0.95_coverage",
    "eval_mean_prob_true",
]


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)
    root = Path(sys.argv[1]).expanduser().resolve()
    # Match both layouts:
    #   <root>/<model>/baseline_metrics.json                (legacy flat)
    #   <root>/<model>/<run_name>/baseline_metrics.json     (registry-organized)
    files = sorted(set(root.glob("*/baseline_metrics.json")) | set(root.glob("*/*/baseline_metrics.json")))
    if not files:
        print(f"No baseline_metrics.json files under {root}", file=sys.stderr)
        sys.exit(1)

    rows: List[Dict[str, Any]] = []
    for path in files:
        with path.open("r", encoding="utf-8") as h:
            payload = json.load(h)
        flat = {
            "run_name": payload.get("run_name"),
            "model_name_or_path": payload.get("model_name_or_path"),
            "eval_split": payload.get("eval_split"),
            "rows": payload.get("eval_split_size_rows"),
            "packed_blocks": payload.get("eval_split_size_packed_blocks"),
        }
        flat.update(payload.get("metrics", {}))
        rows.append(flat)

    summary = {"runs": rows}
    json_out = root / "baseline_summary.json"
    with json_out.open("w", encoding="utf-8") as h:
        json.dump(summary, h, ensure_ascii=False, indent=2)

    md_out = root / "baseline_summary.md"
    cols = ["run_name"] + COMPARABLE_KEYS
    with md_out.open("w", encoding="utf-8") as h:
        h.write("# Baseline LLM Comparison\n\n")
        h.write("Eval split: same Phase 1 test parquet, capped to `max_test_rows=2000`.\n")
        h.write("Token-level PPL is **not** comparable across tokenizers; rely on `ppl_byte` / `bits_per_byte`.\n\n")
        h.write("| " + " | ".join(cols) + " |\n")
        h.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in rows:
            cells = []
            for key in cols:
                val = r.get(key)
                if isinstance(val, float):
                    cells.append(f"{val:.4f}")
                elif val is None:
                    cells.append("—")
                else:
                    cells.append(str(val))
            h.write("| " + " | ".join(cells) + " |\n")

    print(f"Wrote {json_out}")
    print(f"Wrote {md_out}")


if __name__ == "__main__":
    main()
