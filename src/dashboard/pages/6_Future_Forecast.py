"""Future forecast — baseline plus explicit dated events.

Projects spend as a chosen baseline method plus toggleable future events,
with explainable best/expected/worst bands for budget planning.
"""
from __future__ import annotations

import sys
import traceback
from datetime import date, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st

from src.aws.cost_explorer import fetch_daily_totals
from src.aws.profiles import ProfileInfo, resolve_all
from src.dashboard.costsense_theme import callout, section
from src.dashboard.forecast_events_ui import (
    render_answer_band,
    render_anomaly_import_section,
    render_chart,
    render_controls,
    render_ledger,
    render_pr_import_section,
    render_waterfall,
)
from src.dashboard.nav import (
    inject_css,
    render_sidebar_footer,
    render_sidebar_header,
    top_bar,
)
from src.forecast import baselines as B
from src.forecast.adapters import drain_pending_events
from src.forecast.event_store import get_stored_events, pop_import_flash
from src.forecast.scenario import project


st.set_page_config(page_title="CostSense · Future Forecast", layout="wide")
inject_css()
render_sidebar_header()

section(
    "Future forecast",
    "Project spend as a baseline plus dated events. Toggle any assumption "
    "to see how the forecast changes — the conversation that improves "
    "budget reviews.",
    kicker="Looking ahead",
)


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_history(
    profile: str, cutoff_iso: str, days: int, service: str | None = None,
) -> pd.DataFrame:
    cutoff = date.fromisoformat(cutoff_iso)
    totals = fetch_daily_totals(
        cutoff - timedelta(days=days), cutoff, profile=profile, service=service,
    )
    if not totals:
        return pd.DataFrame(columns=["day", "actual_usd"])
    return pd.DataFrame(
        [{"day": d.isoformat(), "actual_usd": float(a)} for d, a in totals],
    ).sort_values("day").reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_by_service(
    profile: str, cutoff_iso: str, days: int,
) -> dict[str, list[tuple[str, float]]]:
    from src.aws.cost_explorer import fetch_daily_by_service

    cutoff = date.fromisoformat(cutoff_iso)
    raw = fetch_daily_by_service(
        cutoff - timedelta(days=days), cutoff, profile=profile,
    )
    return {
        s: [(d.isoformat(), amt) for d, amt in pts]
        for s, pts in raw.items()
    }


with st.spinner("Resolving profiles…"):
    profiles: list[ProfileInfo] = resolve_all()
reachable = [p for p in profiles if p.account_id]
if not reachable:
    callout(
        "No reachable AWS profiles. Run `aws sso login` or launch via "
        "`aws-vault exec <profile> --` first, then reload.",
        tone="error",
    )
    st.stop()

labels = [p.label for p in reachable]
picked_label = st.session_state.get("fc_profile", labels[0])
if picked_label not in labels:
    picked_label = labels[0]

_last_fc_profile = st.session_state.get("fc_last_profile")
if _last_fc_profile != picked_label:
    st.session_state["fc_last_profile"] = picked_label
    st.session_state["fc_widget_ver"] = (
        st.session_state.get("fc_widget_ver", 0) + 1
    )
_widget_ver = st.session_state.get("fc_widget_ver", 0)

_cutoff = st.session_state.get("fc_cutoff", date.today())
_history_days = st.session_state.get("fc_history_days", 90)
_svc_pick = st.session_state.get("fc_service", "(all services — total spend)")

try:
    svc_map = _fetch_by_service(
        reachable[labels.index(picked_label)].profile,
        _cutoff.isoformat(),
        _history_days,
    )
except Exception:  # noqa: BLE001
    svc_map = {}

if svc_map:
    svc_totals = sorted(
        ((s, sum(a for _, a in v)) for s, v in svc_map.items()),
        key=lambda x: -x[1],
    )
    svc_options = ["(all services — total spend)"] + [
        f"{s}  —  ${t:,.0f}" for s, t in svc_totals
    ]
else:
    svc_options = ["(all services — total spend)"]
if _svc_pick not in svc_options:
    _svc_pick = svc_options[0]
_selected_service = (
    None if _svc_pick.startswith("(all") else _svc_pick.split("  —  ")[0]
)

header = (
    f"Controls  ·  Account: {picked_label}  ·  "
    f"Cutoff: {_cutoff.isoformat()}  ·  History: {_history_days}d  ·  "
    f"Scope: {_selected_service or 'All services'}"
)
with top_bar(header):
    r1c1, r1c2, r1c3, r1c4 = st.columns(
        [3, 2, 2, 2], gap="medium", vertical_alignment="bottom",
    )
    with r1c1:
        picked_label = st.selectbox(
            "Account", labels, index=labels.index(picked_label), key="fc_profile",
        )
    with r1c2:
        cutoff = st.date_input(
            "Cutoff",
            value=_cutoff,
            key="fc_cutoff",
            help="Forecast starts the day after this date.",
        )
    with r1c3:
        history_days = st.slider(
            "History (days)", 30, 180, _history_days, step=15, key="fc_history_days",
        )
    with r1c4:
        pick_svc = st.selectbox(
            "Service filter", svc_options,
            index=svc_options.index(_svc_pick), key="fc_service",
        )
    selected_service = (
        None if pick_svc.startswith("(all") else pick_svc.split("  —  ")[0]
    )
    if st.button("Refresh cache", key="fc_refresh_cache"):
        st.cache_data.clear()
        st.rerun()

chosen = reachable[labels.index(picked_label)]
active_profile = chosen.profile
account_id = chosen.account_id

events, _pending_added = drain_pending_events(account_id)
flash = pop_import_flash(account_id)
if flash:
    callout(flash["message"], tone=flash.get("tone", "info"))
elif _pending_added:
    callout(
        f"Imported {_pending_added} queued event(s) from Anomalies or PR Predictor.",
        tone="success",
    )

with st.sidebar:
    render_sidebar_footer(
        active_profile=active_profile,
        account_id=account_id,
        extra_rows=[
            ("Cutoff", cutoff.isoformat()),
            ("History", f"{history_days}d"),
            ("Scope", selected_service or "All services"),
            ("Events", str(len(events))),
        ],
    )

with st.spinner(f"Fetching cost history for `{active_profile}`…"):
    try:
        hist_df = _fetch_history(
            active_profile, cutoff.isoformat(), history_days,
            service=selected_service,
        )
    except Exception as e:  # noqa: BLE001
        callout(f"Cost Explorer fetch failed: {e}", tone="error")
        st.code(traceback.format_exc())
        st.stop()

if hist_df.empty:
    callout(
        "No cost history for this account and window. Try a different "
        "cutoff, history length, or service filter.",
        tone="warning",
    )
    st.stop()

hist_dates = [date.fromisoformat(d) for d in hist_df["day"]]
hist_values = hist_df["actual_usd"].tolist()
start_day = hist_dates[-1] + timedelta(days=1)

# Import handlers run before projection so events land in the same render.
render_pr_import_section(
    account_id=account_id,
    active_profile=active_profile,
    widget_ver=_widget_ver,
)
render_anomaly_import_section(
    account_id=account_id,
    active_profile=active_profile,
)
events = get_stored_events(account_id)

trailing_avg = sum(hist_values[-7:]) / min(7, len(hist_values))
default_cpu = round(trailing_avg / 750, 2) if trailing_avg else 4.53

horizon, method, scenario, budget, units = render_controls(
    budget_default=640_000,
    cost_per_unit_day=default_cpu,
    unit_label="orgs",
)

baseline = B.build_baseline(
    method,
    hist_values,
    hist_dates,
    horizon,
    unit_counts=units,
    cost_per_unit_day=default_cpu,
)
proj = project(
    baseline.values,
    method=baseline.method,
    explanation=baseline.explanation,
    residual_std=baseline.residual_std,
    start_day=start_day,
    events=events,
)

render_answer_band(proj, budget, scenario)
st.caption(baseline.explanation)
st.divider()
render_chart(hist_dates, hist_values, proj, scenario, budget)
st.divider()
render_waterfall(proj)
st.divider()
render_ledger(proj, account_id, start_day)
