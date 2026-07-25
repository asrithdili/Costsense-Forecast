"""Disk-backed session state.

Streamlit's ``st.session_state`` is fine for state that lives within one
tab of one browser session. But if:

  * The user opens a second browser tab (each is a new session)
  * The user closes and reopens the tab
  * Streamlit's server restarts
  * A weird internal event clears the session

…then everything the user painstakingly ran (a PR verdict, an anomaly
report) vanishes and they have to click Run again. That's a bad demo
experience.

This module gives pages a two-tier cache:

  1. In-memory ``st.session_state`` — instant. Populated on Run.
  2. Disk-pickle fallback under ``data/ui_state/`` — restores state on
     new sessions.

Usage on a page::

    from src.dashboard.state_cache import cached_state

    verdict = cached_state.get("prp_verdict", (profile, url))
    if verdict is None:
        # nothing on disk either — user needs to click Run
        ...

    # After a successful Run:
    cached_state.set("prp_verdict", (profile, url), verdict)

Keys are (namespace, *identity) tuples. The disk file lives at
``data/ui_state/<namespace>__<sha1(identity)>.pkl``.
"""
from __future__ import annotations

import atexit
import hashlib
import os
import pickle
import threading
from pathlib import Path
from typing import Any, Optional, Tuple

import streamlit as st


_ROOT = Path(__file__).resolve().parents[2]
_UI_STATE_DIR = _ROOT / "data" / "ui_state"


def _wipe_ui_state_dir() -> None:
    """Delete every persisted state file. Called on module import (fresh
    Streamlit run) and on process exit (graceful shutdown).

    User expectation: stopping Streamlit with Ctrl+C should reset the
    app's caches so the next run doesn't restore stale values. This
    trades cross-restart durability for a cleaner start each session —
    the deliberate trade-off requested for this app."""
    if not _UI_STATE_DIR.exists():
        return
    for path in _UI_STATE_DIR.glob("*.pkl"):
        try:
            path.unlink()
        except OSError:
            # Nothing we can meaningfully do if a file is locked — the
            # next startup will retry.
            pass


# Wipe on import — every `streamlit run` starts clean. This catches
# any shutdown mode (Ctrl+C, kill, crash, power loss) because it
# runs at the next start rather than depending on a shutdown hook
# firing.
#
# EXCEPTION: containerised deployments (ECS Fargate) restart the
# container on health-check failures, task-def updates, or Fargate
# platform patching — every restart is a fresh filesystem. The wipe
# behaviour that's helpful locally becomes destructive there: every
# cache set() by one Streamlit page is gone by the time another
# page tries to read it (because a task restart happened between,
# or the ALB routed a subsequent request to a warm task instance
# that was never told about the write).
#
# Set COSTSENSE_PRESERVE_CACHE=1 in the container to skip both the
# on-import wipe and the atexit wipe. In that mode the ONLY thing
# that clears the cache is a task restart (which is already a full
# filesystem reset). Locally, leave the env var unset and behaviour
# is unchanged.
_UI_STATE_DIR.mkdir(parents=True, exist_ok=True)
if os.environ.get("COSTSENSE_PRESERVE_CACHE") != "1":
    _wipe_ui_state_dir()
    # atexit is belt-and-suspenders for SIGINT (Ctrl+C). It won't fire
    # on SIGKILL / power loss, which is why the on-import wipe is the
    # real safety net when the flag is off.
    atexit.register(_wipe_ui_state_dir)

_FILE_LOCK = threading.Lock()


def _identity_hash(identity: Tuple[Any, ...]) -> str:
    """Short SHA-1 of the identity tuple. Same identity across processes
    yields the same file name."""
    raw = "\x1f".join(str(x) for x in identity).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def _path_for(namespace: str, identity: Tuple[Any, ...]) -> Path:
    safe_ns = "".join(c if c.isalnum() or c in "-_" else "_"
                       for c in namespace)
    return _UI_STATE_DIR / f"{safe_ns}__{_identity_hash(identity)}.pkl"


class _CachedState:
    """Two-tier state cache. Reads prefer session_state; writes hit both."""

    def _key(self, namespace: str, identity: Tuple[Any, ...]) -> str:
        return f"csstate::{namespace}::{_identity_hash(identity)}"

    def get(self, namespace: str, identity: Tuple[Any, ...],
            default: Any = None) -> Any:
        """Return the cached value, or ``default`` if not found.

        Checks session_state first (hot cache), then disk (cold cache).
        On disk hit, promotes back into session_state so subsequent
        reads are fast.
        """
        ss_key = self._key(namespace, identity)
        if ss_key in st.session_state:
            return st.session_state[ss_key]

        path = _path_for(namespace, identity)
        if not path.exists():
            return default
        try:
            with _FILE_LOCK, path.open("rb") as f:
                value = pickle.load(f)
        except Exception:  # noqa: BLE001
            # Corrupt / incompatible pickle — nuke and treat as miss.
            try:
                path.unlink()
            except OSError:
                pass
            return default

        st.session_state[ss_key] = value
        return value

    def set(self, namespace: str, identity: Tuple[Any, ...],
            value: Any) -> None:
        """Write to both session_state and disk. Safe to call from
        multiple threads (rare in Streamlit but possible via
        ThreadPoolExecutor)."""
        ss_key = self._key(namespace, identity)
        st.session_state[ss_key] = value

        path = _path_for(namespace, identity)
        try:
            with _FILE_LOCK, path.open("wb") as f:
                pickle.dump(value, f)
        except Exception:  # noqa: BLE001
            # Disk-write failure shouldn't take the page down — the
            # in-memory copy still works for this session.
            pass

    def clear(self, namespace: str,
              identity: Optional[Tuple[Any, ...]] = None) -> None:
        """Clear a specific entry (identity given) or a whole namespace
        (identity = None). Useful when the user hits a Refresh button
        that should invalidate the cache."""
        if identity is not None:
            ss_key = self._key(namespace, identity)
            st.session_state.pop(ss_key, None)
            path = _path_for(namespace, identity)
            try:
                path.unlink()
            except OSError:
                pass
            return

        # Whole namespace: iterate the disk dir + session_state.
        prefix = f"csstate::{namespace}::"
        for key in list(st.session_state.keys()):
            if isinstance(key, str) and key.startswith(prefix):
                st.session_state.pop(key, None)
        safe_ns = "".join(c if c.isalnum() or c in "-_" else "_"
                          for c in namespace)
        for path in _UI_STATE_DIR.glob(f"{safe_ns}__*.pkl"):
            try:
                path.unlink()
            except OSError:
                pass


# Module-level singleton — pages import this directly.
cached_state = _CachedState()
