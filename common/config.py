"""YAML config loader with env-var ${VAR:-default} interpolation and
`section.key=value` CLI overrides.

Same env-interp convention used by agent-skills-collection / -scanning /
-preparation: cloners can keep machine-specific paths in env vars and
leave config.example.yaml in git untouched.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import yaml


_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _interpolate_env(value: Any) -> Any:
    """Recursively expand ${VAR} / ${VAR:-default} in any string value
    nested inside a dict/list. Non-string leaves pass through."""
    if isinstance(value, str):
        def repl(m: "re.Match[str]") -> str:
            name, default = m.group(1), m.group(2)
            return os.environ.get(name, default if default is not None else "")
        return _ENV_VAR_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _parse_scalar(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _apply_cli_overrides(config: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be in key=value form")
        key, value = item.split("=", 1)
        key = key.lstrip("-")
        parts = key.split(".")
        target = config
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = _parse_scalar(value)
    return config


def load_config(
    path: str,
    overrides: Sequence[str] = (),
    required_sections=("run", "data", "model", "training"),
) -> Dict[str, Any]:
    """Load a YAML config, interpolate ${VAR}, apply CLI key=value overrides,
    and ensure required top-level sections exist (as empty dicts if absent).

    `overrides` is the list of trailing positional args from a CLI like
    `--config foo.yaml training.max_steps=20 model.name=Qwen3-8B`.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config = _interpolate_env(config)
    for section in required_sections:
        config.setdefault(section, {})
    _apply_cli_overrides(config, list(overrides))
    return config
