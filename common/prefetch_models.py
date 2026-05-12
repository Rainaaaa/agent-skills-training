"""Pre-download every HF-backed model in `model_path.json` into the bound
HuggingFace cache.

Use this to warm `hf_cache/` ahead of a long training run so the trainer
never blocks on a download mid-step. Entries with a usable `local` path
are skipped. Entries with neither a present local path nor an `hf` repo
are reported and counted as failures.

Usage (from inside the container, with the HF cache bound):

    docker compose run --rm --no-deps --entrypoint python pretraining \\
        -m common.prefetch_models

    # Limit to specific names:
    docker compose run --rm --no-deps --entrypoint python pretraining \\
        -m common.prefetch_models Foundation-Sec-8B-Reasoning Qwen3-8B

    # Use a non-default registry file:
    docker compose run --rm --no-deps --entrypoint python pretraining \\
        -m common.prefetch_models --registry /app/model_path.kelly.json
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from common.registry import (
    DEFAULT_REGISTRY_PATH,
    Entry,
    _is_marker,
    load_registry,
)


LOGGER = logging.getLogger("prefetch_models")

# Default allowlist mirrors what training actually needs. Override per-call
# via `--allow-patterns` if you want safetensors-only / no consolidated, etc.
DEFAULT_ALLOW_PATTERNS = [
    "*.json", "*.safetensors", "*.txt", "*.model",
    "tokenizer*", "special_tokens_map*", "*.py",
]


def _entry_summary(name: str, entry: Entry) -> str:
    if isinstance(entry, str):
        return f"{name}: <string> {entry}"
    if isinstance(entry, dict):
        return f"{name}: local={entry.get('local')!r} hf={entry.get('hf')!r}"
    return f"{name}: <unknown> {entry!r}"


def _local_present(entry: Entry) -> Optional[str]:
    """Return the local path if the entry has one that exists on disk, else None."""
    if isinstance(entry, dict):
        local = entry.get("local")
        if local and not _is_marker(local) and Path(str(local)).expanduser().exists():
            return str(Path(str(local)).expanduser())
    elif isinstance(entry, str):
        if not _is_marker(entry):
            p = Path(entry).expanduser()
            if p.exists():
                return str(p)
    return None


def _hf_repo(entry: Entry) -> Optional[str]:
    if isinstance(entry, dict):
        hf = entry.get("hf")
        if hf and not _is_marker(hf):
            return str(hf)
    return None


def _prefetch_one(name: str, repo: str, allow_patterns: List[str]) -> bool:
    from huggingface_hub import snapshot_download

    LOGGER.info("[%s] downloading from HF: %s", name, repo)
    try:
        path = snapshot_download(
            repo_id=repo,
            max_workers=8,
            allow_patterns=allow_patterns,
        )
        LOGGER.info("[%s] cached at %s", name, path)
        return True
    except Exception as exc:  # huggingface_hub raises various concrete types
        LOGGER.error("[%s] download failed: %s", name, exc)
        return False


def prefetch(
    registry_path: Path,
    only: Optional[Iterable[str]] = None,
    allow_patterns: Optional[List[str]] = None,
) -> int:
    registry = load_registry(registry_path)
    only_set = set(only) if only else None
    if only_set:
        missing = only_set - set(registry)
        if missing:
            LOGGER.error("requested names not in registry: %s", sorted(missing))
            return 2

    patterns = list(allow_patterns or DEFAULT_ALLOW_PATTERNS)
    n_skipped = n_downloaded = n_failed = n_no_target = 0

    for name, entry in registry.items():
        if only_set is not None and name not in only_set:
            continue
        local = _local_present(entry)
        if local:
            LOGGER.info("[%s] local present, skipping (%s)", name, local)
            n_skipped += 1
            continue
        repo = _hf_repo(entry)
        if not repo:
            LOGGER.warning(
                "[%s] no usable local path AND no `hf` fallback (%s)",
                name, _entry_summary(name, entry),
            )
            n_no_target += 1
            continue
        if _prefetch_one(name, repo, patterns):
            n_downloaded += 1
        else:
            n_failed += 1

    LOGGER.info(
        "prefetch summary: downloaded=%d skipped(local)=%d failed=%d no-target=%d",
        n_downloaded, n_skipped, n_failed, n_no_target,
    )
    return 0 if (n_failed == 0 and n_no_target == 0) else 1


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Prefetch registered models from HuggingFace.")
    parser.add_argument(
        "names", nargs="*",
        help="Only prefetch these model names (default: all entries in the registry).",
    )
    parser.add_argument(
        "--registry", type=Path, default=DEFAULT_REGISTRY_PATH,
        help=f"Path to model_path.json (default: {DEFAULT_REGISTRY_PATH}).",
    )
    parser.add_argument(
        "--allow-patterns", nargs="*", default=None,
        help="Override the HF allow_patterns list (default: tokenizer + safetensors + json).",
    )
    args = parser.parse_args(argv)
    return prefetch(args.registry, args.names or None, args.allow_patterns)


if __name__ == "__main__":
    sys.exit(main())
