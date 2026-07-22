"""Load local `.env` files into ``os.environ``.

Python does not read ``.env`` on its own — values only work after this runs.
Existing environment variables are never overwritten (shell exports win).

Searched paths (first match wins per key, all files are read in order):
  - ``<repo>/.env``
  - ``<repo>/src/.env``
  - ``./.env`` (current working directory)
"""
from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[7:].strip()
    if "=" not in line:
        return None
    key, _, value = line.partition("=")
    key = key.strip()
    if not key:
        return None
    value = value.strip()
    if value and value[0] in "\"'" and value[-1] == value[0]:
        value = value[1:-1]
    else:
        # Unquoted — drop inline comments.
        if " #" in value:
            value = value.split(" #", 1)[0].rstrip()
    return key, value


def _load_file(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        parsed = _parse_line(raw)
        if not parsed:
            continue
        key, value = parsed
        if key not in os.environ:
            os.environ[key] = value


def load_env() -> None:
    """Idempotent — safe to call from every entry point."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    root = _repo_root()
    for path in (root / ".env", root / "src" / ".env", Path.cwd() / ".env"):
        if path.is_file():
            _load_file(path)
