"""Entry point for Streamlit Community Cloud.

Streamlit Cloud looks for a top-level ``streamlit_app.py`` by default.
This file just delegates to the real app under ``src/dashboard/app.py`` —
we don't move the app itself because every internal import (and every
page under ``src/dashboard/pages/``) uses ``from src.dashboard...``.

Doing the delegation this way keeps the entry point discoverable for
Streamlit Cloud without touching the internal package layout.

For local development, keep running:
    streamlit run src/dashboard/app.py

Both paths land on the same code — this file just ``exec``s the real
entry point so the shared page navigation, session state, and
``src`` imports resolve identically.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_APP_PATH = _HERE / "src" / "dashboard" / "app.py"

# Add repo root to sys.path so `from src...` imports resolve.
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _promote_streamlit_secrets_to_env() -> None:
    """Streamlit Cloud stores runtime secrets in ``st.secrets``, but our
    app (and boto3 / PyGithub under the hood) read from ``os.environ``.
    Promote every top-level ``st.secrets`` key into the process
    environment so downstream code doesn't need to know it's running on
    Streamlit Cloud.

    On local runs (no secrets.toml present), ``st.secrets`` is empty
    and this becomes a no-op.
    """
    try:
        import streamlit as st
    except Exception:  # noqa: BLE001 — Streamlit may not be importable during hot-reload edges
        return
    try:
        secrets = st.secrets
    except Exception:  # noqa: BLE001 — thrown when secrets.toml is missing (local dev)
        return
    for key in secrets:
        value = secrets[key]
        if isinstance(value, (str, int, float, bool)):
            os.environ.setdefault(key, str(value))


_promote_streamlit_secrets_to_env()

# Run the real app as if `streamlit run src/dashboard/app.py` had been
# called. `run_name="__main__"` matches what Streamlit expects.
runpy.run_path(str(_APP_PATH), run_name="__main__")
