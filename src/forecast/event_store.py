"""Streamlit session-state persistence for forecast events."""
from __future__ import annotations

import streamlit as st

from src.forecast.events import CostEvent


def events_session_key(account_id: str) -> str:
    return f"fc_events::{account_id}"


def import_flash_key(account_id: str) -> str:
    return f"fc_import_flash::{account_id}"


def get_stored_events(account_id: str) -> list[CostEvent]:
    raw = st.session_state.get(events_session_key(account_id), [])
    if not raw:
        return []
    if isinstance(raw[0], dict):
        return [CostEvent.from_dict(row) for row in raw]
    return list(raw)


def store_events(account_id: str, events: list[CostEvent]) -> None:
    st.session_state[events_session_key(account_id)] = [
        event.to_dict() for event in events
    ]


def set_import_flash(account_id: str, message: str, tone: str = "info") -> None:
    st.session_state[import_flash_key(account_id)] = {
        "message": message,
        "tone": tone,
    }


def pop_import_flash(account_id: str) -> dict | None:
    return st.session_state.pop(import_flash_key(account_id), None)
