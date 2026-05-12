"""Model registry — resolves a short logical name (e.g. `Foundation-Sec-8B-Reasoning`)
into a filesystem path or HF repo id, by reading `model_path.json` at the repo root.

Schema (both forms supported; new entries should use the dict form):

    {
      "GeneralLLMs": {
        "<name>": "<path-or-hub-id>",                    # legacy string form
        "<name>": {"local": "<path>", "hf": "<repo>"}    # preferred dict form
      },
      ...
    }

Resolution for dict entries:
  1. If `local` is set and the path exists on disk, return `local`.
  2. Otherwise, if `hf` is set, return the HF repo id (transformers will
     download into the HF cache).
  3. Otherwise raise.

A value of `"downloading"` (or any other NOT_AVAILABLE marker) in either
form is treated as not yet available and raises a clear error.

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
from typing import Any, Dict, Iterator, Optional, Tuple, Union


LOGGER = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "model_path.json"
NOT_AVAILABLE_MARKERS = {"downloading", "not_available", "tbd", "todo"}

# An entry value is either a string (legacy) or a dict with optional
# `local`/`hf` keys (new). The registry is `{name: Entry}`.
Entry = Union[str, Dict[str, Any]]


def _iter_entries(data: Dict[str, Any]) -> Iterator[Tuple[str, str, Entry]]:
    """Yield (category, name, entry) triples.

    Recognizes two shapes:
      - {category: {name: entry}}                                 (canonical)
      - {name: entry}                                             (flat root)

    An "entry" is either a string path/repo OR a dict like
    {"local": "...", "hf": "..."} — both are passed through to the resolver.
    """
    for category, models in data.items():
        if isinstance(models, dict) and not {"local", "hf"} & set(models.keys()):
            # Category dict: descend one level.
            for name, entry in models.items():
                yield category, name, entry
        else:
            # Flat root: the value itself is an entry (string or rich dict).
            yield "_root", category, models


def load_registry(path: Optional[Path] = None) -> Dict[str, Entry]:
    """Return a flat `{name: entry}` dict from `model_path.json`."""
    p = Path(path) if path else DEFAULT_REGISTRY_PATH
    if not p.exists():
        raise FileNotFoundError(f"Model registry not found at {p}")
    with p.open("r", encoding="utf-8") as h:
        data = json.load(h)
    flat: Dict[str, Entry] = {}
    for _, name, entry in _iter_entries(data):
        flat[name] = entry
    return flat


def _is_marker(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in NOT_AVAILABLE_MARKERS


def _resolve_entry(name: str, entry: Entry) -> str:
    """Pick the best target for an entry (local-if-present, else HF)."""
    if isinstance(entry, str):
        if _is_marker(entry):
            raise FileNotFoundError(
                f"Model '{name}' is marked '{entry}' in registry — not yet downloaded."
            )
        return entry
    if isinstance(entry, dict):
        local = entry.get("local")
        hf = entry.get("hf")
        if local and not _is_marker(local) and Path(local).expanduser().exists():
            LOGGER.info("registry: '%s' -> local %s", name, local)
            return str(Path(local).expanduser())
        if hf and not _is_marker(hf):
            LOGGER.info(
                "registry: '%s' local missing; falling back to HF '%s'", name, hf,
            )
            return hf
        if local and not _is_marker(local):
            # Local was specified but path doesn't exist, and no HF fallback.
            raise FileNotFoundError(
                f"Model '{name}': local path '{local}' does not exist and no 'hf' fallback set."
            )
        raise FileNotFoundError(
            f"Model '{name}' has no usable target (entry={entry})."
        )
    raise TypeError(f"Model '{name}' has unsupported entry type {type(entry)}: {entry!r}")


def resolve(name: str, registry_path: Optional[Path] = None) -> str:
    """Resolve `name` -> path/HF-repo via the registry.

    For dict entries, prefer `local` if the path exists on disk; otherwise
    fall back to `hf`. Raises KeyError if name is absent, FileNotFoundError
    if neither target is usable.
    """
    registry = load_registry(registry_path)
    if name not in registry:
        raise KeyError(
            f"Model '{name}' not found in registry. Available: {sorted(registry)}"
        )
    return _resolve_entry(name, registry[name])


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
