from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if key == "base_config":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    base_path = cfg.get("base_config")
    if base_path:
        candidate = Path(base_path)
        if not candidate.is_absolute():
            candidate = (path.parent.parent / candidate).resolve()
        base = load_config(candidate)
        cfg = _deep_merge(base, cfg)
    cfg["_config_path"] = str(path)
    return cfg


def resolve_path(value: str | Path, config_path: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    root = Path(config_path).resolve().parent.parent
    return (root / path).resolve()
