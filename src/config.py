"""Load application settings from ``config/costsense.json``.

Optional ``config/costsense.local.json`` is deep-merged on top for local
overrides (gitignored). Environment variables still win for shell/CI overrides
when explicitly checked by callers (e.g. GitHub token fallback in ``gh_client``).
"""
from __future__ import annotations

import json
import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _config_dir() -> Path:
    override = os.environ.get("COSTSENSE_CONFIG_DIR")
    if override:
        return Path(override)
    return _repo_root() / "config"


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    config_dir = _config_dir()
    config = _read_json(config_dir / "costsense.json")
    local = _read_json(config_dir / "costsense.local.json")
    if local:
        config = _deep_merge(config, local)
    return config


def reload_config() -> dict[str, Any]:
    load_config.cache_clear()
    return load_config()


def get(path: str, default: Any = None) -> Any:
    """Return a nested value using dot notation, e.g. ``aws.region``."""
    node: Any = load_config()
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def get_str(path: str, default: str | None = None) -> str | None:
    value = get(path, default)
    if value is None:
        return default
    text = str(value).strip()
    return text or default
