"""Model registry — resolves a short logical name (e.g. `Foundation-Sec-8B-Reasoning`)
into a filesystem path or HF repo id, by reading `model_path.json` at the repo root.

Schema:

    {
      "GeneralLLMs":      { "<name>": "<path-or-hub-id>", ... },
      "CybersecurityLLM": { "<name>": "<path-or-hub-id>", ... }
    }

A value of `"downloading"` (or any non-existent path) is treated as not yet
available and raises a clear error.

Usage from a config:

    model:
      name: Foundation-Sec-8B-Reasoning
      # registry_path defaults to <repo>/model_path.json

    # Explicit override always wins:
    model:
      name: Foundation-Sec-8B-Reasoning
      model_name_or_path: /path/to/some/local/checkpoint
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple


LOGGER = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "model_path.json"
NOT_AVAILABLE_MARKERS = {"downloading", "not_available", "tbd", "todo"}


def _iter_entries(data: Dict[str, Dict[str, str]]) -> Iterator[Tuple[str, str, str]]:
    """Yield (category, name, path) triples; tolerates flat shape too."""
    for category, models in data.items():
        if isinstance(models, dict):
            for name, path in models.items():
                yield category, name, path
        else:
            yield "_root", category, models


def load_registry(path: Optional[Path] = None) -> Dict[str, str]:
    """Return a flat `{name: path}` dict from `model_path.json`."""
    p = Path(path) if path else DEFAULT_REGISTRY_PATH
    if not p.exists():
        raise FileNotFoundError(f"Model registry not found at {p}")
    with p.open("r", encoding="utf-8") as h:
        data = json.load(h)
    flat: Dict[str, str] = {}
    for _, name, target in _iter_entries(data):
        flat[name] = target
    return flat


def resolve(name: str, registry_path: Optional[Path] = None) -> str:
    """Resolve `name` -> path via the registry. Raises if missing or marked unavailable."""
    registry = load_registry(registry_path)
    if name not in registry:
        raise KeyError(
            f"Model '{name}' not found in registry. Available: {sorted(registry)}"
        )
    target = registry[name]
    if isinstance(target, str) and target.strip().lower() in NOT_AVAILABLE_MARKERS:
        raise FileNotFoundError(
            f"Model '{name}' is marked '{target}' in registry — not yet downloaded."
        )
    return target


def resolve_model_path(
    model_cfg: Dict[str, Any],
    registry_path: Optional[Path] = None,
) -> Tuple[str, Optional[str]]:
    """Pick the model path to load + the model's logical name (if any).

    Precedence:
      1. `model.model_name_or_path` if set — used as-is (no registry lookup).
      2. `model.name` — looked up in the registry.

    Returns (path, name_or_none).
    """
    explicit = model_cfg.get("model_name_or_path")
    name = model_cfg.get("name")
    if explicit:
        if name:
            LOGGER.info(
                "Both model.name=%s and model.model_name_or_path=%s set; using explicit path.",
                name, explicit,
            )
        return explicit, name
    if name:
        path = resolve(name, registry_path or model_cfg.get("registry_path"))
        LOGGER.info("Resolved model.name=%s -> %s", name, path)
        return path, name
    raise ValueError(
        "Neither `model.name` nor `model.model_name_or_path` is set. "
        "Pick one (e.g. model.name=Foundation-Sec-8B-Reasoning)."
    )


def sanitize_for_path(name: str) -> str:
    """Make a registry name safe to use as a directory component."""
    safe = name.strip().replace("/", "__").replace(" ", "_")
    return safe or "unknown_model"
