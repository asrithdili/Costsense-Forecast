"""Shared top-of-page control bar + sidebar chrome for CostSense.

Rules:
- **Theme + shell.** ``inject_css()`` composes the shared design system
  (``costsense_theme.inject_css()``) with structural shell CSS in ``_CSS``
  below. Shell rules win on layout conflicts (sidebar flex footer, wider
  max-width) so navigation and footer pinning stay stable.
- **Sidebar order.** ``render_sidebar_header()`` renders the Diligent brand
  card first, then custom ``st.page_link`` nav. Native multipage nav is
  disabled via ``showSidebarNavigation = false`` in ``.streamlit/config.toml``
  (with a CSS fallback). Page-specific sidebar widgets and
  ``render_sidebar_footer()`` follow below.
- The optional footer status card (`render_sidebar_footer()`) pins to
  the bottom of the sidebar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import streamlit as st

from src.aws.profiles import ProfileInfo, resolve_all
from src.dashboard.costsense_theme import callout, inject_css as inject_theme_css


DEFAULT_MODELS = (
    ("us.anthropic.claude-sonnet-4-6",              "Claude Sonnet 4.6"),
    ("anthropic.claude-3-haiku-20240307-v1:0",      "Claude 3 Haiku"),
)


# Page-nav definition — path is relative to the entry file
# `src/dashboard/app.py`. Streamlit's `st.page_link` picks up each page's
# `page_icon` from its own `set_page_config` automatically, so we don't
# duplicate it here.
_PAGES = [
    ("app.py",                          "Ask CostSense"),
    ("pages/2_Dashboard.py",            "Dashboard"),
    ("pages/3_PR_Predictor.py",         "PR Predictor"),
    ("pages/4_Anomalies.py",            "Anomalies"),
    ("pages/5_Org_Level_Impact.py",     "Org Level Impact"),
    ("pages/6_Future_Forecast.py",      "Future Forecast"),
    ("pages/7_Close_The_Loop.py",       "Close the Loop"),
]


# ---------------------------------------------------------------------------
# CSS — kept structural. No `.stApp { background }` etc, so Streamlit's
# theme (System / Light / Dark) drives the base colors.
# ---------------------------------------------------------------------------
_CSS = """
<style>
/* Slim the empty header strip a bit but leave its theme-derived color. */
[data-testid="stHeader"] { min-height: 42px; }
[data-testid="stToolbar"] { right: 12px; top: 8px; }

.main .block-container {
  padding-top: 1rem !important;
  padding-bottom: 3rem;
  max-width: 1400px;
}

/* Footer pinned to bottom of the sidebar (still works — plain flex). */
section[data-testid="stSidebar"] > div {
  display: flex;
  flex-direction: column;
}
.cs-sidebar-footer { margin-top: auto; }

/* Hide Streamlit's auto page nav — custom nav renders below the brand card. */
[data-testid="stSidebarNav"] { display: none !important; }

/* --- "Pages" section label between the brand card and the links --- */
.cs-pages-label {
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  padding: 4px 4px 2px 4px;
  margin: 4px 0 4px 0;
  color: var(--muted);
}

/* Separator between page links and the rest of the sidebar content. */
.cs-nav-sep {
  height: 1px;
  margin: 12px 4px 8px 4px;
  background: var(--line);
}

/* Style st.page_link entries in the sidebar as compact nav pills. */
section[data-testid="stSidebar"] [data-testid="stPageLink"] {
  padding: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a {
  padding: 6px 12px !important;
  margin: 1px 4px !important;
  border-radius: 8px !important;
  font-size: 14px !important;
  font-weight: 500 !important;
  color: var(--ink) !important;
}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {
  background: var(--brand-soft) !important;
  color: var(--brand) !important;
}
section[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {
  background: var(--brand-soft) !important;
  color: var(--brand) !important;
  font-weight: 600 !important;
}

/* --- Diligent brand card — teal-aligned, calm product chrome --- */
.cs-brandbar {
  padding: 14px 16px;
  margin: 0 0 10px 0;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: linear-gradient(180deg, var(--brand-soft) 0%, var(--card) 100%);
  text-align: center;
  box-shadow: 0 1px 2px rgba(20, 24, 31, 0.04);
}
.cs-brandbar .cs-bb-row1 {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 9px;
  margin-bottom: 6px;
}
.cs-brandbar .cs-bb-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--brand);
  box-shadow: 0 0 8px rgba(12, 124, 116, 0.35);
  flex-shrink: 0;
}
.cs-brandbar .cs-bb-name {
  font-weight: 800;
  font-size: 16px;
  letter-spacing: -0.015em;
  line-height: 1;
  color: var(--ink);
}
.cs-brandbar .cs-bb-sub {
  display: block;
  font-weight: 600;
  font-size: 9.5px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  line-height: 1;
}

/* --- Controls-bar expander polish --- */
[data-testid="stExpander"] {
  border: 1px solid var(--line) !important;
  border-radius: 10px !important;
  margin-bottom: 16px !important;
  box-shadow: 0 1px 2px rgba(20, 24, 31, 0.04);
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] details > summary {
  padding: 6px 14px !important;
  min-height: 34px !important;
  font-size: 12.5px !important;
  font-weight: 500 !important;
  color: var(--ink) !important;
}
.cs-controls-title {
  display: flex; align-items: center; height: 100%;
  padding: 0 12px;
  font-weight: 600; font-size: 13px;
  color: var(--muted);
  white-space: nowrap; user-select: none;
}

/* --- Sidebar footer status card --- */
.cs-sidebar-footer {
  margin-top: auto;
  padding: 14px 12px 16px 12px;
  border-top: 1px solid var(--line);
  font-size: 11.5px;
  line-height: 1.45;
  color: var(--muted);
}
.cs-sidebar-footer .cs-sf-title {
  display: flex; align-items: center; gap: 8px;
  font-weight: 700;
  font-size: 12.5px;
  letter-spacing: 0.02em;
  margin-bottom: 6px;
  color: var(--ink);
}
.cs-sidebar-footer .cs-sf-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--brand);
  box-shadow: 0 0 8px rgba(12, 124, 116, 0.35);
}
.cs-sidebar-footer .cs-sf-row {
  display: flex; justify-content: space-between; gap: 8px;
  padding: 2px 0;
}
.cs-sidebar-footer .cs-sf-row span:last-child {
  text-align: right; word-break: break-all;
  color: var(--ink);
}
.cs-sidebar-footer .cs-sf-pill {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 2px 7px;
  border-radius: 999px;
  background: var(--brand-soft);
  border: 1px solid var(--brand);
  color: var(--brand);
  font-size: 10.5px;
  font-weight: 600;
  letter-spacing: 0.02em;
}
.cs-sidebar-footer .cs-sf-pill::before {
  content: ""; width: 6px; height: 6px; border-radius: 50%;
  background: var(--brand);
  box-shadow: 0 0 6px rgba(12, 124, 116, 0.4);
}
.cs-sidebar-footer .cs-sf-caption {
  font-size: 10.5px; margin-top: 8px;
  padding-top: 8px;
  border-top: 1px dashed var(--line);
  color: var(--faint);
}
</style>
"""


_TOP_BRAND_HTML = (
    '<div class="cs-brandbar">'
    '<div class="cs-bb-row1">'
    '<span class="cs-bb-dot"></span>'
    '<span class="cs-bb-name">Diligent</span>'
    '</div>'
    '<span class="cs-bb-sub">CostSense · AI FinOps</span>'
    '</div>'
)

# Fallback if config.toml showSidebarNavigation is ever re-enabled.
_SIDEBAR_HIDE_CSS = (
    '<style>[data-testid="stSidebarNav"]{display:none!important;}</style>'
)


def _inject_sidebar_hide_css() -> None:
    """Inject nav-hide CSS into the sidebar stream (earlier than main-pane CSS)."""
    st.sidebar.markdown(_SIDEBAR_HIDE_CSS, unsafe_allow_html=True)


@dataclass
class TopBarSelection:
    profile: ProfileInfo
    model_id: str | None


def inject_css() -> None:
    """Inject shared theme + shell CSS. Safe to call multiple times per page."""
    inject_theme_css()
    _inject_sidebar_hide_css()
    st.markdown(_CSS, unsafe_allow_html=True)


def render_sidebar_header() -> None:
    """Render brand card + page nav at the top of the sidebar.

    Native multipage nav is disabled in config; custom links render here.
    """
    _inject_sidebar_hide_css()
    with st.sidebar:
        st.markdown(_TOP_BRAND_HTML, unsafe_allow_html=True)
        st.markdown('<div class="cs-pages-label">Pages</div>', unsafe_allow_html=True)
        for path, label in _PAGES:
            st.page_link(path, label=label, use_container_width=True)
        st.markdown('<div class="cs-nav-sep"></div>', unsafe_allow_html=True)


def render_top_brand() -> None:
    """Backwards-compat alias."""
    render_sidebar_header()


def top_bar(header: str):
    """Context manager that returns a collapsed top-bar expander.

        with top_bar("Controls · Dashboard"):
            ...

    Injects the shared CSS automatically.
    """
    inject_css()
    return st.expander(header, expanded=False)


def render_sidebar_footer(
    *,
    active_profile: str | None = None,
    account_id: str | None = None,
    extra_rows: list[tuple[str, str]] | None = None,
    caption: str = "Read-only access · Secrets auto-scrubbed",
) -> None:
    """Pin a compact status card to the bottom of the sidebar.

    Call INSIDE a `with st.sidebar:` block at the very end of the
    sidebar rendering.
    """
    rows: list[tuple[str, str]] = []
    if account_id:
        rows.append(("Account", account_id))
    if active_profile:
        rows.append(("Profile", active_profile))
    if extra_rows:
        rows.extend(extra_rows)
    row_html = "".join(
        f'<div class="cs-sf-row"><span>{label}</span>'
        f'<span>{value}</span></div>'
        for label, value in rows
    )
    st.markdown(
        f"""
        <div class="cs-sidebar-footer">
          <div class="cs-sf-title">
            <span class="cs-sf-dot"></span>
            <span>CostSense</span>
            <span style="flex:1"></span>
            <span class="cs-sf-pill">Live</span>
          </div>
          {row_html}
          <div class="cs-sf-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render(
    *,
    include_model: bool = False,
    model_options: Iterable[tuple[str, str]] = DEFAULT_MODELS,
    default_model_index: int = 0,
    key_prefix: str = "topbar",
) -> TopBarSelection:
    """Compact top bar for pages that only need Account + optional Model.

    Renders inside `top_bar()` — a collapsed expander whose header shows
    the current selections. Returns the selected profile + model.
    """
    inject_css()

    profiles = [p for p in resolve_all() if p.account_id]
    if not profiles:
        callout("No AWS profiles reachable.", tone="error")
        st.stop()

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
        if include_model:
            col_account, col_model = st.columns([1, 1], gap="small")
        else:
            col_account, _spacer = st.columns([1, 1], gap="small")

        with col_account:
            picked_label = st.selectbox(
                "Account", labels,
                index=labels.index(picked_label),
                key=f"{key_prefix}_profile",
            )

        if include_model:
            with col_model:
                picked_model_idx = st.selectbox(
                    "Model",
                    range(len(model_labels)),
                    index=picked_model_idx,
                    format_func=lambda i: model_labels[i],
                    key=f"{key_prefix}_model",
                )

    active = profiles[labels.index(picked_label)]
    model_id = model_ids[picked_model_idx] if include_model else None
    return TopBarSelection(profile=active, model_id=model_id)
