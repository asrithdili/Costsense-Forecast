"""Shared top-of-page control bar.

Renders a single-row "Controls" strip above the page title with
labeled chips: AWS Account + (optional) Bedrock Model. Matches the
inline-controls pattern seen in dashboarding tools (Datadog, CloudZero
etc.) where every control is a compact "Label value" pair on one line.

The 5-page navigation stays in Streamlit's default sidebar list.

Call `render(...)` early on each page. It returns the selected profile
(and optionally model) so the rest of the page can use them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import streamlit as st

from src.aws.profiles import ProfileInfo, resolve_all


DEFAULT_MODELS = (
    ("us.anthropic.claude-sonnet-4-6",              "Claude Sonnet 4.6"),
    ("anthropic.claude-3-haiku-20240307-v1:0",      "Claude 3 Haiku"),
)


_CSS = """
<style>
/* --- collapse the empty top header strip --- */
[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stToolbar"] { right: 8px; top: 8px; }

/* --- app polish --- */
.stApp { background: #0e1015; }
section[data-testid="stSidebar"] > div {
  background: #14161d;
  border-right: 1px solid #23262f;
}
h1, h2, h3, h4 { letter-spacing: -0.01em; color: #f5f6fa; }
.main .block-container {
  padding-top: 0.5rem !important;
  padding-bottom: 3rem;
  max-width: 1400px;
}

/* --- sidebar page nav — polish the default --- */
[data-testid="stSidebarNav"] { padding-top: 0.5rem; }
[data-testid="stSidebarNav"] ul { padding-left: 0; }
[data-testid="stSidebarNav"] li a {
  border-radius: 8px;
  padding: 6px 12px !important;
  margin: 2px 6px !important;
  color: #c7cbd6 !important;
  font-weight: 500 !important;
}
[data-testid="stSidebarNav"] li a:hover {
  background: #1c1f28 !important;
  color: #ffffff !important;
}
[data-testid="stSidebarNav"] li a[aria-current="page"] {
  background: #232838 !important;
  color: #ffffff !important;
  box-shadow: inset 0 0 0 1px #3b4152;
}

/* --- top control expander — compact header --- */
[data-testid="stExpander"] {
  border: 1px solid #1c2a38 !important;
  border-radius: 8px !important;
  background: #0f1620 !important;
  margin-bottom: 16px !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] details > summary {
  padding: 6px 14px !important;
  min-height: 34px !important;
  font-size: 12.5px !important;
  color: #b4b9c5 !important;
  font-weight: 500 !important;
  letter-spacing: 0.01em;
}
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] details > summary:hover {
  color: #ffffff !important;
  background: #131b26 !important;
}
[data-testid="stExpander"] p {
  font-size: 12.5px !important;
  margin: 0 !important;
  color: inherit !important;
}
[data-testid="stExpander"] svg { color: #6f7686 !important; }

/* --- top control bar wrapper (inside the expander) --- */
.cs-controls-wrap {
  background: transparent;
  border: none;
  border-radius: 0;
  margin-bottom: 4px;
  padding: 0;
}
/* Inner grid so labels + selectboxes sit inline like tabs. */
.cs-controls-wrap [data-testid="column"] {
  padding: 0 !important;
}
.cs-controls-title {
  display: flex;
  align-items: center;
  height: 100%;
  padding: 0 18px;
  color: #d3d7df;
  font-weight: 600;
  font-size: 13px;
  letter-spacing: 0.02em;
  border-right: 1px solid #1c2a38;
  white-space: nowrap;
  user-select: none;
}

/* Selectbox layout — put the label inline to the LEFT of the value,
   the way the "Controls" reference bar does. */
.cs-controls-wrap [data-testid="stSelectbox"] {
  padding: 6px 16px !important;
  border-right: 1px solid #1c2a38;
  min-width: 0;
}
.cs-controls-wrap [data-testid="stSelectbox"] > label {
  display: inline-block !important;
  color: #d3d7df !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  text-transform: none !important;
  letter-spacing: 0.01em !important;
  margin: 0 !important;
  padding: 0 !important;
  min-height: 0 !important;
}
.cs-controls-wrap [data-testid="stSelectbox"] > label p {
  color: #d3d7df !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  margin: 0 !important;
}
.cs-controls-wrap [data-testid="stSelectbox"] div[data-baseweb="select"] {
  margin-top: 2px !important;
}
.cs-controls-wrap [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  background: transparent !important;
  border: none !important;
  min-height: 26px !important;
  padding: 0 !important;
  color: #8fb9ff !important;
  font-size: 13px !important;
  font-weight: 400 !important;
  box-shadow: none !important;
}
.cs-controls-wrap [data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
  background: transparent !important;
}
/* the little chevron */
.cs-controls-wrap [data-testid="stSelectbox"] div[data-baseweb="select"] svg {
  color: #8fb9ff !important;
  fill: #8fb9ff !important;
}
/* Remove the border on the last column so the bar ends cleanly. */
.cs-controls-wrap > div > div > div:last-child [data-testid="stSelectbox"] {
  border-right: none;
}

/* --- metric cards --- */
[data-testid="stMetric"] {
  background: #181b23;
  padding: 14px 16px;
  border-radius: 10px;
  border: 1px solid #23262f;
}
[data-testid="stMetricLabel"] { color: #8a90a0; font-size: 12px; letter-spacing: 0.02em; }
[data-testid="stMetricValue"] { color: #f5f6fa; font-weight: 600; }

/* --- primary buttons --- */
.stButton > button[data-testid="baseButton-primary"] {
  background: linear-gradient(180deg, #6b4dff, #5a3fe6);
  border: 1px solid #7c5cff;
  color: #ffffff;
  font-weight: 600;
}

/* --- chat bubbles --- */
[data-testid="stChatMessage"] {
  background: #181b23;
  border: 1px solid #23262f;
  border-radius: 10px;
}
</style>
"""


@dataclass
class TopBarSelection:
    profile: ProfileInfo
    model_id: str | None


def inject_css() -> None:
    """Inject the shared CSS. Safe to call multiple times per page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def top_bar(header: str):
    """Context manager that returns a collapsed top-bar expander for a
    page to fill with its own controls.

        with top_bar("Controls · Dashboard"):
            col1, col2 = st.columns(2)
            with col1: st.selectbox(...)
            with col2: st.slider(...)

    Injects the shared CSS and dark-theme polish automatically.
    """
    inject_css()
    return st.expander(header, expanded=False)


def render(
    *,
    include_model: bool = False,
    model_options: Iterable[tuple[str, str]] = DEFAULT_MODELS,
    default_model_index: int = 0,
    key_prefix: str = "topbar",
) -> TopBarSelection:
    """Render the shared top control bar and return the selection.

    Controls live inside a collapsed expander so they don't consume
    vertical space when the user isn't changing them. The expander
    header always shows the current Account (and Model, if enabled) so
    the current context is visible at a glance without expanding.

    Calls `st.stop()` if no AWS profiles resolve.
    """
    st.markdown(_CSS, unsafe_allow_html=True)

    profiles = [p for p in resolve_all() if p.account_id]
    if not profiles:
        st.error("No AWS profiles reachable.")
        st.stop()

    # Read the current selection from session_state (so the expander
    # header reflects it BEFORE expanding). Fallback to first option.
    labels = [p.label for p in profiles]
    picked_label = st.session_state.get(f"{key_prefix}_profile", labels[0])
    if picked_label not in labels:
        picked_label = labels[0]

    model_ids = [mid for mid, _ in model_options]
    model_labels = [display for _, display in model_options]
    picked_model_idx = st.session_state.get(
        f"{key_prefix}_model", default_model_index,
    )
    if not (0 <= picked_model_idx < len(model_ids)):
        picked_model_idx = default_model_index

    if include_model:
        header = (f"Controls  ·  Account: {picked_label}  ·  "
                  f"Model: {model_labels[picked_model_idx]}")
    else:
        header = f"Controls  ·  Account: {picked_label}"

    with st.expander(header, expanded=False):
        st.markdown('<div class="cs-controls-wrap">', unsafe_allow_html=True)
        if include_model:
            col_title, col_account, col_model, _spacer = st.columns(
                [0.9, 3.5, 3.5, 4.1], gap="small",
            )
        else:
            col_title, col_account, _spacer = st.columns(
                [0.9, 3.5, 7.6], gap="small",
            )

        with col_title:
            st.markdown(
                '<div class="cs-controls-title">Controls</div>',
                unsafe_allow_html=True,
            )

        with col_account:
            picked_label = st.selectbox(
                "Account",
                labels,
                index=labels.index(picked_label),
                key=f"{key_prefix}_profile",
                label_visibility="visible",
            )

        if include_model:
            with col_model:
                picked_model_idx = st.selectbox(
                    "Model",
                    range(len(model_labels)),
                    index=picked_model_idx,
                    format_func=lambda i: model_labels[i],
                    key=f"{key_prefix}_model",
                    label_visibility="visible",
                )
        st.markdown('</div>', unsafe_allow_html=True)

    active = profiles[labels.index(picked_label)]
    model_id = model_ids[picked_model_idx] if include_model else None
    return TopBarSelection(profile=active, model_id=model_id)
