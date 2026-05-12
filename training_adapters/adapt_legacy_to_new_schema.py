"""Legacy → new-schema adapter for Jetstream smoke testing.

This script is a *smoke-test convenience only* — it lets Stages 2 (HCL) and
3 (SFT) run end-to-end against pre-refactor data so Kelly IT can verify the
code paths on their GPUs. The semantic mapping is approximate; real training
runs MUST use parquets produced by the refactored agent-skills-preparation.

Adaptations:

  Stage 2 (HCL)
    legacy pl_hcl rows  -> new HCL pair rows
    For each anchor_skill_id we materialize the anchor's `positive` row and
    pair it once with itself (label=1, pair_kind=positive) and once with
    each corruption row (label=0, pair_kind=corrupted). T1/T2/T3b legacy
    labels collapse to "corrupted" since the new code only distinguishes
    positive/corrupted/swapped.

  Stage 3 SFT (misalignment)
    legacy classifier rows -> new misalignment-detection rows
      skill_text       = instruction + "\n\n" + input_text + "\n\n" + evidence_text
      alignment_class  = "ALIGNED" if target_text=="yes" else "MISALIGNED"

  Stage 3 SFT (malicious)
    Same source rows -> malicious-detection schema
      skill_text   = (same composition as above)
      overall_class = "SAFE" if target_text=="yes" else "MALICIOUS"

Usage (from inside the training container, with legacy data mounted at /data):

    python /app/training_adapters/adapt_legacy_to_new_schema.py \\
        --legacy-root /data \\
        --out-root    /data/adapted \\
        --rows        2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def adapt_hcl(legacy_root: Path, out_dir: Path, max_rows_per_split: int) -> None:
    src = legacy_root / "pl_hcl" / "pl_hcl_v1" / "stage2"
    for split_name, src_file in (
        ("train", "train.parquet"),
        ("val",   "val.parquet"),
        ("test",  "test.parquet"),
    ):
        df = pd.read_parquet(src / src_file)
        positives = (
            df[df.label == "positive"][["anchor_skill_id", "text"]]
            .drop_duplicates("anchor_skill_id")
            .rename(columns={"text": "anchor_text"})
        )
        joined = df.merge(positives, on="anchor_skill_id", how="inner")
        joined["pair_text"] = joined["text"]
        joined["pair_kind"] = joined["label"].map(
            lambda lbl: "positive" if lbl == "positive" else "corrupted"
        )
        joined["label_int"] = (joined["label"] == "positive").astype(int)
        out = (
            joined[["anchor_text", "pair_text", "pair_kind", "label_int"]]
            .rename(columns={"label_int": "label"})
            .sample(n=min(len(joined), max_rows_per_split), random_state=42)
            .reset_index(drop=True)
        )
        out_path = out_dir / f"{split_name}.parquet"
        out.to_parquet(out_path, compression="zstd")
        print(f"  HCL/{split_name}: {len(df)} legacy -> {len(out)} pairs -> {out_path}")


def _compose_skill_text(row) -> str:
    parts = []
    for col in ("instruction", "input_text", "evidence_text"):
        v = row.get(col)
        if isinstance(v, str) and v:
            parts.append(v)
    return "\n\n".join(parts)


def adapt_sft(legacy_root: Path, out_align_dir: Path, out_mal_dir: Path, max_rows_per_split: int) -> None:
    src = legacy_root / "sft" / "sft_v1"
    splits = (
        ("train", "classifier_train.parquet"),
        ("val",   "classifier_val.parquet"),
        ("test",  "classifier_test.parquet"),
    )
    for split_name, src_file in splits:
        df = pd.read_parquet(src / src_file)
        if len(df) > max_rows_per_split:
            df = df.sample(n=max_rows_per_split, random_state=42).reset_index(drop=True)
        df["skill_text"] = df.apply(_compose_skill_text, axis=1)
        df["alignment_class"] = df["target_text"].map(
            lambda t: "ALIGNED" if str(t).strip().lower() == "yes" else "MISALIGNED"
        )
        df["overall_class"] = df["target_text"].map(
            lambda t: "SAFE" if str(t).strip().lower() == "yes" else "MALICIOUS"
        )
        align = df[["skill_text", "alignment_class"]]
        mal   = df[["skill_text", "overall_class"]]
        align_path = out_align_dir / f"{split_name}.parquet"
        mal_path   = out_mal_dir   / f"{split_name}.parquet"
        align.to_parquet(align_path, compression="zstd")
        mal.to_parquet(mal_path, compression="zstd")
        print(f"  SFT/{split_name}: {len(df)} rows -> {align_path}, {mal_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Legacy → new-schema adapter (smoke only)")
    parser.add_argument("--legacy-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=2000,
                        help="Max rows per split (caps both HCL and SFT outputs)")
    args = parser.parse_args()

    hcl_out   = args.out_root / "hcl"
    align_out = args.out_root / "sft_align"
    mal_out   = args.out_root / "sft_malicious"
    for d in (hcl_out, align_out, mal_out):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Adapting HCL pl_hcl_v1/stage2 -> {hcl_out}")
    adapt_hcl(args.legacy_root, hcl_out, args.rows)
    print(f"Adapting SFT sft_v1/classifier -> {align_out} (align), {mal_out} (mal)")
    adapt_sft(args.legacy_root, align_out, mal_out, args.rows)
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
