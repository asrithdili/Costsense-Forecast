"""Streamlit session-state persistence for forecast events."""
from __future__ import annotations

import streamlit as st

from src.forecast.events import CostEvent


def events_session_key(profile: str) -> str:
    """Per AWS profile — multiple profiles can share one account_id."""
    return f"fc_events::{profile}"


def import_flash_key(profile: str) -> str:
    return f"fc_import_flash::{profile}"


def get_stored_events(profile: str) -> list[CostEvent]:
    raw = st.session_state.get(events_session_key(profile), [])
    if not raw:
        return []
    if isinstance(raw[0], dict):
        return [CostEvent.from_dict(row) for row in raw]
    return list(raw)


def store_events(profile: str, events: list[CostEvent]) -> None:
    st.session_state[events_session_key(profile)] = [
        event.to_dict() for event in events
    ]


def set_import_flash(profile: str, message: str, tone: str = "info") -> None:
    st.session_state[import_flash_key(profile)] = {
        "message": message,
        "tone": tone,
    }


def pop_import_flash(profile: str) -> dict | None:
    return st.session_state.pop(import_flash_key(profile), None)
