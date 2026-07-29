"""CostSense forecast dashboard — live AWS Cost Explorer only.

Flow when you pick a profile in the sidebar:
  1. STS resolves the profile → account id.
  2. Cost Explorer is called for `history_days` back from the cutoff date.
  3. Prophet forecasts the next 7 days on data strictly before the cutoff.
  4. The chart shows past actuals (Cost Explorer) + future prediction with band.
  5. Backtest scores auto-compute for target dates where a 7-day-old forecast
     exists on disk AND the actual has landed.

Nothing is stubbed or seeded — if the account has no cost data, the panels
say so instead of drawing fake numbers.
"""
from __future__ import annotations

import json
import sys
import traceback
from datetime import date, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.aws.cost_explorer import fetch_daily_by_service, fetch_daily_totals
from src.aws.profiles import ProfileInfo, resolve_all
from src.backtest.scorer import backfill_scores_from_disk
from src.forecast.backtest_replay import training_fit_replay
from src.pipeline.paths import actuals_dir, backtest_dir, predictions_dir
from src.pipeline.run_daily import run as run_pipeline


FORECAST_MODEL_OPTIONS: dict[str, str] = {
    "lightgbm": "LightGBM",
    "ewm": "EWM (auto-tuned)",
    "prophet": "Prophet",
    "aws": "AWS native (GetCostForecast)",
}
from src.pr_scanner.repos import (
    gh_login,
    gh_orgs,
    recent_base_branches,
    repo_default_branch,
    repos_with_user_prs,
)
from src.dashboard.costsense_theme import (
    C, callout, metric, money, pill, plotly_layout, section,
)
from src.dashboard.live_cost_meter import render_live_cost_meter
from src.dashboard.notifications_ui import NotificationDraft, render_notification_button
from src.dashboard.nav import (
    inject_css, render_sidebar_footer, render_sidebar_header, top_bar,
)


st.set_page_config(page_title="CostSense · forecast", layout="wide")
inject_css()
# Render the Diligent brand card FIRST — before any AWS calls — so it
# appears instantly instead of waiting for STS profile resolution.
render_sidebar_header()


# ---------- cached AWS fetchers ----------

@st.cache_data(ttl=600, show_spinner=False)
def _fetch_cutoff_day_actual(
    profile: str,
    cutoff_iso: str,
    service: str | None = None,
) -> float | None:
    """Return Cost Explorer actual for cutoff day, or None when CE has no row."""
    cutoff = date.fromisoformat(cutoff_iso)
    totals = fetch_daily_totals(
        cutoff, cutoff + timedelta(days=1), profile=profile, service=service,
    )
    for day, amount in totals:
        if day == cutoff:
            return float(amount)
    return None


def _hist_past_for_chart(
    hist_df: pd.DataFrame,
    cutoff: date,
    profile: str,
    service: str | None,
) -> pd.DataFrame:
    """Past actuals through cutoff. Skip today when CE has not landed yet."""
    cutoff_iso = cutoff.isoformat()
    today = date.today()

    def _append_cutoff_row(df: pd.DataFrame) -> pd.DataFrame:
        day_actual = _fetch_cutoff_day_actual(profile, cutoff_iso, service=service)
        if day_actual is not None:
            return pd.concat([df, pd.DataFrame([{
                "day": cutoff_iso,
                "actual_usd": day_actual,
            }])], ignore_index=True)
        if cutoff < today:
            return pd.concat([df, pd.DataFrame([{
                "day": cutoff_iso,
                "actual_usd": 0.0,
            }])], ignore_index=True)
        return df

    if hist_df.empty:
        day_actual = _fetch_cutoff_day_actual(profile, cutoff_iso, service=service)
        if day_actual is not None:
            return pd.DataFrame([{"day": cutoff_iso, "actual_usd": day_actual}])
        if cutoff < today:
            return pd.DataFrame([{"day": cutoff_iso, "actual_usd": 0.0}])
        return pd.DataFrame(columns=["day", "actual_usd"])

    past = hist_df.loc[_on_or_before(hist_df["day"], cutoff)].copy()
    if cutoff_iso not in past["day"].astype(str).tolist():
        past = _append_cutoff_row(past)
    return past.sort_values("day").reset_index(drop=True)


def _future_trace_xy(
    cutoff: date,
    anchor_y: float | None,
    future_x: list,
    future_y: list,
) -> tuple[list, list[float]]:
    """Connect cutoff anchor to future points for one continuous forecast trace."""
    ys = [float(v) for v in future_y]
    if anchor_y is None:
        return future_x, ys
    return [cutoff.isoformat()] + list(future_x), [anchor_y] + ys


_CHART_TICK_STEP_DAYS = 14


def _chart_day(value) -> date:
    return pd.Timestamp(value).date()


def _forecast_end_date(fc_future: pd.DataFrame) -> date | None:
    if fc_future.empty:
        return None
    return _chart_day(fc_future["target_date"].max())


def _chart_view_start(
    hist_past: pd.DataFrame,
    cutoff: date,
    history_days: int,
    *,
    bt_past: pd.DataFrame | None = None,
    day_col: str = "target_date",
) -> date:
    candidates: list[date] = []
    if not hist_past.empty:
        candidates.append(_chart_day(hist_past["day"].min()))
    if bt_past is not None and not bt_past.empty:
        candidates.append(_chart_day(bt_past[day_col].min()))
    if candidates:
        return min(candidates)
    return cutoff - timedelta(days=history_days)


def _chart_data_start(
    *series: tuple[pd.DataFrame, str],
    default: date,
) -> date:
    """First date with plotted data — avoids empty left padding on the axis."""
    candidates: list[date] = []
    for df, col in series:
        if df is not None and not df.empty:
            candidates.append(_chart_day(df[col].min()))
    return min(candidates) if candidates else default


def _fourteen_day_tick_dates(
    view_start: date,
    view_end: date,
    *,
    step: int = _CHART_TICK_STEP_DAYS,
) -> list[date]:
    """14-day ticks through the range; last label is exact forecast end day."""
    ticks: list[date] = []
    d = view_start
    while d <= view_end:
        ticks.append(d)
        d += timedelta(days=step)
    ticks = [t for t in ticks if t <= view_end]
    if not ticks or ticks[-1] != view_end:
        ticks.append(view_end)
    return ticks


def _forecast_chart_xaxis(view_start: date, view_end: date) -> dict:
    ticks = _fourteen_day_tick_dates(view_start, view_end)
    return dict(
        gridcolor=C.HAIRLINE,
        zerolinecolor=C.HAIRLINE,
        range=[view_start.isoformat(), view_end.isoformat()],
        tickmode="array",
        tickvals=[t.isoformat() for t in ticks],
        tickformat="%d %b",
    )


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_history(
    profile: str, cutoff_iso: str, days: int, service: str | None = None,
) -> pd.DataFrame:
    cutoff = date.fromisoformat(cutoff_iso)
    totals = fetch_daily_totals(
        cutoff - timedelta(days=days), cutoff + timedelta(days=1),
        profile=profile, service=service,
    )
    if not totals:
        return pd.DataFrame(columns=["day", "actual_usd"])
    df = pd.DataFrame([{"day": d.isoformat(), "actual_usd": float(a)} for d, a in totals])
    return df.sort_values("day").reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_by_service(
    profile: str, cutoff_iso: str, days: int,
) -> dict[str, list[tuple[str, float]]]:
    """Returns {service: [(day_iso, amount), ...]} sorted by total spend desc."""
    cutoff = date.fromisoformat(cutoff_iso)
    raw = fetch_daily_by_service(
        cutoff - timedelta(days=days), cutoff + timedelta(days=1), profile=profile,
    )
    # normalize to iso strings for st.cache_data hashability
    return {
        s: [(d.isoformat(), amt) for d, amt in pts]
        for s, pts in raw.items()
    }


def _latest_forecast_file(account_id: str, service: str | None = None) -> Path | None:
    """Return the newest forecast JSON that matches the current service
    filter. Prevents mixing an all-services forecast onto a service-filtered
    history chart (which would make the future line jump discontinuously)."""
    all_fs = sorted(predictions_dir(account_id).glob("forecast_*.json"))
    if not all_fs:
        return None
    if service:
        svc_suffix = f"__{service.replace(' ', '_')}.json"
        matches = [f for f in all_fs if f.name.endswith(svc_suffix)]
        return matches[-1] if matches else None
    # No service selected → prefer files without any __service suffix
    plain = [f for f in all_fs if "__" not in f.name.replace("forecast_", "", 1)]
    return plain[-1] if plain else None


def _load_latest_forecast(account_id: str,
                          service: str | None = None) -> dict | None:
    f = _latest_forecast_file(account_id, service=service)
    return json.loads(f.read_text()) if f else None


def _load_backtest(account_id: str) -> pd.DataFrame:
    rows = [json.loads(f.read_text())
            for f in sorted(backtest_dir(account_id).glob("score_*.json"))]
    return pd.DataFrame(rows)


def _rgba(hex_color: str, alpha: float) -> str:
    """Hex token → rgba() string for Plotly fill colors."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _maybe_bump_cutoff_to_today() -> None:
    """Advance the default cutoff when the calendar day changes so charts
    don't stay pinned to yesterday's run."""
    today = date.today()
    anchor = st.session_state.get("_dash_today_anchor")
    if anchor == today.isoformat():
        return
    if anchor:
        prev_day = date.fromisoformat(anchor)
        if st.session_state.get("dash_cutoff") == prev_day:
            st.session_state["dash_cutoff"] = today
    if "dash_cutoff" not in st.session_state:
        st.session_state["dash_cutoff"] = today
    st.session_state["_dash_today_anchor"] = today.isoformat()


def _on_or_before(series: pd.Series, cutoff: date) -> pd.Series:
    return pd.to_datetime(series) <= pd.Timestamp(cutoff)


def _after(series: pd.Series, cutoff: date) -> pd.Series:
    return pd.to_datetime(series) > pd.Timestamp(cutoff)


def _ensure_row_at_cutoff(
    df: pd.DataFrame,
    cutoff: date,
    day_col: str,
    defaults: dict[str, float],
) -> pd.DataFrame:
    """Pad a missing cutoff row when CE End is exclusive (used for fitted series)."""
    if df.empty:
        row = {day_col: cutoff.isoformat(), **defaults}
        return pd.DataFrame([row])
    out = df.copy()
    cutoff_iso = cutoff.isoformat()
    if cutoff_iso not in out[day_col].astype(str).tolist():
        row = {day_col: cutoff_iso, **defaults}
        out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
    return out.sort_values(day_col).reset_index(drop=True)


def _past_through_cutoff(
    df: pd.DataFrame,
    cutoff: date,
    day_col: str,
    pad_defaults: dict[str, float],
) -> pd.DataFrame:
    if df.empty:
        return _ensure_row_at_cutoff(df, cutoff, day_col, pad_defaults)
    past = df.loc[_on_or_before(df[day_col], cutoff)].copy()
    return _ensure_row_at_cutoff(past, cutoff, day_col, pad_defaults)


def _value_at_cutoff(
    df: pd.DataFrame,
    cutoff: date,
    day_col: str,
    y_col: str,
) -> float | None:
    if df.empty:
        return None
    match = df.loc[pd.to_datetime(df[day_col]) == pd.Timestamp(cutoff)]
    if not match.empty:
        return float(match.iloc[-1][y_col])
    return float(df.iloc[-1][y_col])


def _backtest_from_training_fit(replay_df: pd.DataFrame) -> pd.DataFrame:
    """Map in-sample training fit rows to backtest chart schema."""
    out = replay_df.rename(columns={
        "predicted_usd": "predicted_usd",
        "actual_usd": "actual_usd",
    }).copy()
    out["abs_error_usd"] = out["abs_err"]
    return out.dropna(subset=["actual_usd"])


def _fit_past_through_cutoff(
    df: pd.DataFrame,
    cutoff: date,
    day_col: str = "target_date",
) -> pd.DataFrame:
    """Extend fitted/backtest series to cutoff (forward-fill last fit)."""
    if df.empty:
        return df
    past = df.loc[_on_or_before(df[day_col], cutoff)].copy()
    cutoff_iso = cutoff.isoformat()
    if cutoff_iso not in past[day_col].astype(str).tolist():
        last = past.iloc[-1].copy()
        last[day_col] = cutoff_iso
        past = pd.concat([past, pd.DataFrame([last])], ignore_index=True)
    return past.sort_values(day_col).reset_index(drop=True)


def _past_anchor_y(
    hist_past: pd.DataFrame,
    fit_past: pd.DataFrame,
    cutoff: date,
    *,
    y_fit: str = "predicted_usd",
    y_actual: str = "actual_usd",
    day_col_fit: str = "target_date",
    day_col_actual: str = "day",
) -> float | None:
    """Y value at cutoff for bridging past → future lines."""
    if not fit_past.empty:
        return _value_at_cutoff(fit_past, cutoff, day_col_fit, y_fit)
    if not hist_past.empty:
        return _value_at_cutoff(hist_past, cutoff, day_col_actual, y_actual)
    return None


# ---------- page title (rendered first so the page never looks blank) ----------

section(
    "CostSense — daily cost forecast",
    "Read-only cost forecast for the selected AWS account. "
    "Open **Controls** below — enable **Include PR impact** only when "
    "you want merged/open PRs layered on top of the billing forecast.",
    kicker="Forecast",
)


# ---------- top control bar ----------

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
picked_label = st.session_state.get("dash_profile", labels[0])
if picked_label not in labels:
    picked_label = labels[0]
chosen = reachable[labels.index(picked_label)]
active_profile = chosen.profile
account_id = chosen.account_id

# Detect account change EARLY (before any Cost Explorer / GitHub calls
# run, so we don't waste a slow API round-trip on the old profile).
# When the profile changes we:
#   1. Bump a version counter — used to build fresh widget keys so
#      Streamlit re-instantiates the multiselect with new defaults
#      instead of restoring the OLD account's saved value.
#   2. Reset the "last profile" pointer and skip the wipe-cascade so
#      no fetches happen on the old profile.
_last_profile = st.session_state.get("dash_last_profile")
if _last_profile != active_profile:
    st.session_state["dash_last_profile"] = active_profile
    st.session_state["dash_widget_ver"] = (
        st.session_state.get("dash_widget_ver", 0) + 1
    )
    # Purge any downstream widget state that was tied to the old profile.
    for k in ("dash_service", "dash_base_branch", "dash_base_branch_text"):
        st.session_state.pop(k, None)
# Used to salt widget keys — the multiselect gets a NEW key on each
# account change, guaranteeing Streamlit uses the fresh `default=` list.
_widget_ver = st.session_state.get("dash_widget_ver", 0)

# Read defaults for other controls up front so the header can reflect them.
_maybe_bump_cutoff_to_today()
_cutoff = st.session_state.get("dash_cutoff", date.today())
_history_days = st.session_state.get("dash_history_days", 90)
_model_choice = st.session_state.get("dash_model", "lightgbm")

# Service list (used both for dropdown + header).
try:
    svc_map = _fetch_by_service(active_profile, _cutoff.isoformat(), _history_days)
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
_svc_pick = st.session_state.get("dash_service", svc_options[0])
if _svc_pick not in svc_options:
    _svc_pick = svc_options[0]
_selected_service = (None if _svc_pick.startswith("(all")
                     else _svc_pick.split("  —  ")[0])

_include_pr = st.session_state.get("dash_include_pr", False)

# GitHub org + repos — only when PR impact is enabled (avoids gh API calls).
orgs: list[str] = []
short_names: list[str] = []
default_repos: list[str] = []
default_org_idx = 0
gh_org_default = "DiligentCorp"
gh_org = st.session_state.get("dash_gh_org", gh_org_default)
if _include_pr:
    try:
        gh_login()
        orgs = list(gh_orgs())
    except Exception:  # noqa: BLE001
        orgs = []

    if orgs:
        for _i, o in enumerate(orgs):
            if o == "DiligentCorp":
                default_org_idx = _i
                break
        gh_org_default = orgs[default_org_idx]
    gh_org = st.session_state.get("dash_gh_org", gh_org_default)

    try:
        suggested_full = list(repos_with_user_prs(gh_org)) if gh_org else []
    except Exception:  # noqa: BLE001
        suggested_full = []
    short_names = [r.split("/", 1)[-1] for r in suggested_full]

    from src.pr_scanner.profile_repo_match import match_repos as _match
    default_repos = _match(active_profile, short_names) or short_names

svc_hdr = _selected_service or "All services"
_pr_hdr = "PR on" if _include_pr else "billing only"
header = (f"Controls  ·  Account: {picked_label}  ·  "
          f"Cutoff: {_cutoff.isoformat()}  ·  "
          f"History: {_history_days}d  ·  Scope: {svc_hdr}  ·  "
          f"Model: {_model_choice}  ·  {_pr_hdr}")

with top_bar(header):
    # Row 1 — AWS
    r1c1, r1c2, r1c3, r1c4 = st.columns(
        [3, 2, 2, 2], gap="medium", vertical_alignment="bottom",
    )
    with r1c1:
        picked_label = st.selectbox(
            "Account", labels,
            index=labels.index(picked_label),
            key="dash_profile",
        )
    with r1c2:
        cutoff = st.date_input(
            "Cutoff", value=_cutoff, key="dash_cutoff",
            help="Forecast starts the day AFTER this. History trains "
                 "only on days STRICTLY before.",
        )
    with r1c3:
        history_days = st.slider(
            "History (days)", 30, 180, _history_days, step=15,
            key="dash_history_days",
        )
    with r1c4:
        _model_ids = list(FORECAST_MODEL_OPTIONS)
        _model_idx = (_model_ids.index(_model_choice)
                      if _model_choice in FORECAST_MODEL_OPTIONS else 0)
        model_choice = st.selectbox(
            "Forecast model", options=_model_ids,
            format_func=lambda m: FORECAST_MODEL_OPTIONS[m],
            index=_model_idx,
            key="dash_model",
            help="LightGBM learns lag + calendar features (default). EWM "
                 "adapts to level shifts fast. Prophet handles weekly "
                 "seasonality. AWS native calls Cost Explorer "
                 "GetCostForecast — billing history only, no PR awareness "
                 "in the model itself.",
        )

    # Row 2 — service filter + optional PR layer
    r2c1, r2c2 = st.columns([4, 2], gap="medium", vertical_alignment="bottom")
    with r2c1:
        pick_svc = st.selectbox(
            "Service filter", svc_options,
            index=svc_options.index(_svc_pick),
            key="dash_service",
            help="Forecast one service at a time for cleaner attribution.",
        )
    with r2c2:
        include_pr = st.checkbox(
            "Include PR impact",
            value=False,
            key="dash_include_pr",
            help="When off, forecast uses Cost Explorer history only — no "
                 "GitHub or Bedrock calls. Turn on to layer merged/open "
                 "PR cost deltas on top.",
        )
    selected_service = (None if pick_svc.startswith("(all")
                        else pick_svc.split("  —  ")[0])

    selected_repos: list[str] = []
    base_branch: str | None = None
    pr_lookback = 14
    analyzer_choice = "regex"
    llm_model_choice = "us.anthropic.claude-sonnet-4-6"

    if include_pr:
        # Row 3 — GitHub
        r3c1, r3c2, r3c3, r3c4 = st.columns(
            [2, 4, 2, 2], gap="medium", vertical_alignment="bottom",
        )
        with r3c1:
            if orgs:
                gh_org = st.selectbox(
                    "GitHub org", orgs,
                    index=orgs.index(gh_org) if gh_org in orgs else default_org_idx,
                    key="dash_gh_org",
                )
            else:
                gh_org = st.text_input(
                    "GitHub org", value=gh_org, key="dash_gh_org",
                )
        with r3c2:
            # Widget key includes the version counter — bumped on account
            # change so Streamlit sees this as a NEW widget and honors
            # `default=` (instead of restoring the previous multiselect
            # state, which would keep showing the old profile's repo).
            picked_short = st.multiselect(
                "Repos", options=short_names, default=default_repos,
                key=f"dash_repos_v{_widget_ver}",
            )
        selected_repos = [f"{gh_org}/{n}" for n in picked_short] if gh_org else []

        # Base branch dropdown
        branch_choices: list[str] = []
        if selected_repos:
            for r in selected_repos:
                try:
                    for b in recent_base_branches(r):
                        if b not in branch_choices:
                            branch_choices.append(b)
                    default = repo_default_branch(r)
                    if default not in branch_choices:
                        branch_choices.append(default)
                except Exception:  # noqa: BLE001
                    continue
        with r3c3:
            if branch_choices:
                base_branch = st.selectbox(
                    "Base branch", branch_choices, key="dash_base_branch",
                )
            else:
                base_branch = st.text_input(
                    "Base branch", placeholder="e.g. main",
                    key="dash_base_branch_text",
                ) or None
        with r3c4:
            pr_lookback = st.slider(
                "PR lookback (d)", 3, 30, 14, step=1, key="dash_pr_lookback",
            )

        # Row 4 — PR analyzer
        r4c1, r4c2, _, _ = st.columns(
            [2, 3, 2, 2], gap="medium", vertical_alignment="bottom",
        )
        with r4c1:
            analyzer_choice = st.selectbox(
                "PR analyzer", options=["hybrid", "llm", "regex"],
                index=0, key="dash_analyzer",
                help="hybrid = LLM + regex fallback. llm = Bedrock only. "
                     "regex = fast, misses config tweaks.",
            )
        with r4c2:
            llm_model_choice = st.selectbox(
                "Bedrock model",
                options=[
                    "us.anthropic.claude-sonnet-4-6",
                    "anthropic.claude-3-haiku-20240307-v1:0",
                ],
                index=0,
                key="dash_pr_llm",
                disabled=(analyzer_choice == "regex"),
            )

    # Row 5 — backtest controls
    r5c1, r5c2 = st.columns([2, 2], gap="medium", vertical_alignment="bottom")
    with r5c1:
        show_replay = st.checkbox(
            "Show training fit", value=True, key="dash_show_backtest",
            disabled=(model_choice == "aws"),
            help="Train once at the cutoff and overlay in-sample fit on "
                 "recent history — did the model track actual spend? "
                 "Unavailable for AWS native.",
        )
    with r5c2:
        fit_lookback_days = st.slider(
            "Fit window (d)", 7, 60, 30, step=1,
            disabled=not show_replay, key="dash_fit_lookback",
            help="How many days of training history to score the fit on.",
        )

    _forecast_help = (
        "Fetch Cost Explorer, scan PRs, fit the model, and save the "
        "next-7-day forecast to disk."
        if include_pr else
        "Fetch Cost Explorer, fit the model, and save the next-7-day "
        "forecast to disk (billing history only)."
    )
    r6c1, r6c2, _ = st.columns(
        [2, 2, 8], gap="medium", vertical_alignment="bottom",
    )
    with r6c1:
        do_forecast = st.button(
            "Run forecast", type="primary", key="dash_run_forecast",
            help=_forecast_help,
        )
    with r6c2:
        do_refresh = st.button("Refresh cache", key="dash_refresh_cache")
    if do_refresh:
        st.cache_data.clear()
        st.rerun()


# ---------- sidebar footer ----------

with st.sidebar:
    render_sidebar_footer(
        active_profile=active_profile,
        account_id=account_id,
        extra_rows=[
            ("Cutoff",   cutoff.isoformat()),
            ("History",  f"{history_days}d"),
            ("Model",    model_choice),
            ("Scope",    selected_service or "All services"),
            ("PR layer", "on" if include_pr else "off"),
        ],
    )


# ---------- main pane ----------

scope_label = selected_service or "All services"
with st.container(border=True):
    section(
        "Active scope",
        f"Account **{account_id}** via `{active_profile}` · scope: {scope_label} · "
        f"cutoff {cutoff.isoformat()} · history {history_days}d",
        kicker="Live",
    )


# History — always fetched live. Show a spinner in the gap so the
# page doesn't look blank while Cost Explorer responds.
with st.spinner(f"Fetching cost history for `{active_profile}`… "
                f"({history_days}d, service: {selected_service or 'all'})"):
    try:
        hist_df = _fetch_history(
            active_profile, cutoff.isoformat(), history_days,
            service=selected_service,
        )
    except Exception as e:  # noqa: BLE001
        callout(f"Cost Explorer fetch failed: {e}", tone="error")
        st.code(traceback.format_exc())
        st.stop()


# On-demand forecast run
#
# Cap open-PR deep-analysis at the top 3 most merge-ready PRs. Each PR
# takes ~30-90s in the Bedrock agent (10-20 tool calls per PR). At 3
# PRs with 4-worker concurrency, the whole open-PR pass fits in one
# parallel batch and finishes in ~60-90s. Larger caps (8+) push
# wall-clock past 5 minutes on a busy repo, which the demo needs to
# stay under.
_MAX_OPEN_PRS_DEEP_ANALYSIS = 3

if do_forecast:
    _run_msg = (
        # Open PRs are analyzed by the SAME deep agent the PR Predictor
        # page uses (precedent lookup + Cost Explorer + CloudWatch tools).
        # We cap it at the top-N most merge-ready PRs and run them
        # in parallel so wall-clock stays ~1-2 minutes on a busy repo.
        f"Deep-analyzing merged PRs + top {_MAX_OPEN_PRS_DEEP_ANALYSIS} "
        f"most merge-ready open PRs, then fitting {model_choice}…"
        if include_pr else
        (f"Calling GetCostForecast for next 7 days…"
         if model_choice == "aws" else
         f"Fetching Cost Explorer, fitting {model_choice}…")
    )
    with st.spinner(_run_msg):
        try:
            out = run_pipeline(
                cutoff=cutoff, profile=active_profile,
                history_days=history_days,
                repos=(selected_repos or None) if include_pr else None,
                base_branch=base_branch if include_pr else None,
                pr_lookback_days=pr_lookback,
                analyzer=analyzer_choice,
                llm_model=llm_model_choice if include_pr and analyzer_choice != "regex" else None,
                service=selected_service,
                model=model_choice,
                include_open_prs=include_pr,
                max_open_prs=_MAX_OPEN_PRS_DEEP_ANALYSIS,
            )
            callout(f"Wrote {Path(out).name}", tone="success")
            st.cache_data.clear()
        except Exception as e:  # noqa: BLE001
            callout(f"Pipeline failed: {e}", tone="error")
            st.code(traceback.format_exc())


latest = _load_latest_forecast(account_id, service=selected_service)

# ---- PR layer status banner ---------------------------------------------
# Makes it obvious whether the "Include PR impact" toggle actually did
# anything on the last run: how many merged PRs were priced, how many open
# PRs the LLM code-reviewed, and the total expected $/day pressure they
# add to the future forecast. When the toggle was off (or no PRs found),
# the banner explains that instead of leaving the reader guessing.
if latest:
    _pr_scan = latest.get("pr_scan") or {}
    _open_scan = latest.get("open_pr_scan") or {}
    _merged_impacts = _pr_scan.get("impacts") or []
    _merged_count = sum(1 for i in _merged_impacts
                        if i.get("est_daily_delta_usd"))
    _open_count = int(_open_scan.get("count", 0))
    _open_delta = float(
        _open_scan.get("total_expected_daily_delta_usd", 0.0)
    )
    _merged_delta = float(latest.get("pr_delta_daily_usd_at_cutoff", 0.0))
    _repos_scanned = _pr_scan.get("repos") or []
    _pr_layer_ran = bool(_repos_scanned or _merged_impacts or _open_count)

    with st.container(border=True):
        if _pr_layer_ran:
            section(
                "PR layer status",
                (f"Scanned {len(_repos_scanned)} repo(s). Open PRs are "
                 "ranked by merge readiness (approved + passing CI + "
                 "recent activity) and the top 3 go through the same "
                 "deep agent the PR Predictor uses — precedent lookup + "
                 "full AWS tool loop. Merged PRs shape the past baseline; "
                 "open PRs bump the future forecast weighted by merge "
                 "probability."),
                kicker="Included",
            )
            pcols = st.columns(3, gap="medium")
            with pcols[0]:
                metric(
                    "Merged PRs priced",
                    _merged_count,
                    delta=(f"${_merged_delta:+,.2f}/day at cutoff"
                           if _merged_delta else None),
                    good=(True if _merged_delta < 0
                          else False if _merged_delta > 0 else None),
                )
            with pcols[1]:
                metric(
                    "Open PRs code-reviewed",
                    _open_count,
                    delta=("scanned by Bedrock" if _open_count else
                           "none found" if _repos_scanned else None),
                )
            with pcols[2]:
                metric(
                    "Expected Δ from open PRs",
                    f"${_open_delta:+,.2f}/day",
                    delta=("weighted by merge probability"
                           if _open_count else None),
                    good=(True if _open_delta < 0
                          else False if _open_delta > 0 else None),
                )
        else:
            section(
                "PR layer status",
                "Last run was billing-only. Enable **Include PR impact** "
                "in Controls and pick at least one repo to layer merged + "
                "open PR deltas on top of the forecast.",
                kicker="Not included",
            )

# Backfill score_*.json from every forecast_*.json on disk
try:
    backfill_scores_from_disk(account_id, active_profile)
except Exception:  # noqa: BLE001
    pass

bt = _load_backtest(account_id)
bt_saved_count = len(bt)
fc_df = pd.DataFrame(latest["forecast"]) if latest else pd.DataFrame()

# PR step series (history + future) persisted by the pipeline.
# Only show it when NO service filter is selected — PR deltas are account-wide,
# so overlaying them on a per-service chart would be misleading.
pr_series_df = pd.DataFrame()
if selected_service is None and latest and latest.get("pr_daily_series"):
    pr_series_df = pd.DataFrame(latest["pr_daily_series"])

# Reconstruct PrSteps from the saved impacts (account-wide PR layer).
# IMPORTANT: PR deltas are estimated for the WHOLE account. Applying them
# to a service-filtered backtest would inflate predictions relative to the
# filtered actuals (e.g. a NAT-gateway PR worth $32/day would be added to
# an EC2-Other chart showing ~$3/day). Skip when service filter is on.
saved_pr_steps: list = []
if selected_service is None and latest and latest.get("pr_scan", {}).get("impacts"):
    from src.forecast.timeseries import PrStep as _PrStep
    for imp in latest["pr_scan"]["impacts"]:
        if not imp.get("est_daily_delta_usd"):
            continue
        try:
            merge_day = date.fromisoformat(imp["merged_at"][:10])
        except (ValueError, KeyError):
            continue
        saved_pr_steps.append(_PrStep(
            from_day=merge_day,
            delta_usd=imp["est_daily_delta_usd"],
            pr_id=f"{imp['repo']}#{imp['pr_number']}",
        ))

# In-sample training fit overlay + trust metric (single train at cutoff)
replay_points: list = []
replay_df = pd.DataFrame()
if show_replay and model_choice != "aws" and not hist_df.empty:
    with st.spinner(f"Fitting {model_choice} on training data…"):
        try:
            hist_for_replay = pd.DataFrame({
                "day": pd.to_datetime(hist_df["day"]),
                "amount_usd": hist_df["actual_usd"],
            })
            replay_points = training_fit_replay(
                hist_for_replay,
                cutoff=cutoff,
                lookback_days=fit_lookback_days,
                pr_steps=saved_pr_steps or None,
                model=model_choice,
            )
            if replay_points:
                replay_df = pd.DataFrame([{
                    "origin": p.origin_date.isoformat(),
                    "target_date": p.target_date.isoformat(),
                    "horizon": p.horizon,
                    "predicted_usd": p.predicted_usd,
                    "lower_usd": p.lower_usd,
                    "upper_usd": p.upper_usd,
                    "actual_usd": p.actual_usd,
                } for p in replay_points])
                replay_df["abs_err"] = (replay_df["predicted_usd"]
                                        - replay_df["actual_usd"]).abs()
                replay_df["ape"] = replay_df.apply(
                    lambda r: (r["abs_err"] / r["actual_usd"])
                    if r["actual_usd"] else None, axis=1)
        except Exception as e:  # noqa: BLE001
            callout(f"Training fit overlay failed: {e}", tone="warning")

# Chart boundary: past (≤ cutoff) vs future (> cutoff, next 7d from saved run).
_chart_cutoff = cutoff
hist_past = _hist_past_for_chart(
    hist_df, _chart_cutoff, active_profile, selected_service,
)
fc_future = (
    fc_df.loc[_after(fc_df["target_date"], _chart_cutoff)].copy()
    if not fc_df.empty else fc_df
)
replay_past = _fit_past_through_cutoff(replay_df, _chart_cutoff, "target_date")
_forecast_end = _forecast_end_date(fc_future)
_chart_x_end = _forecast_end if _forecast_end else _chart_cutoff
_chart_x_start = _chart_view_start(
    hist_past, _chart_cutoff, history_days,
)
_forecast_stale = (
    latest
    and latest.get("run_cutoff")
    and latest["run_cutoff"] != _chart_cutoff.isoformat()
)

# KPI row
kpis = st.columns(4, gap="medium")
next_7_total = fc_future["adjusted_usd"].sum() if not fc_future.empty else None
last_7_actual = hist_past.tail(7)["actual_usd"].sum() if not hist_past.empty else None
_replay_valid = replay_df.dropna(subset=["actual_usd"]) if not replay_df.empty else replay_df
_replay_total_actual = _replay_valid["actual_usd"].abs().sum() if not _replay_valid.empty else 0.0
replay_wape = ((_replay_valid["abs_err"].sum() / _replay_total_actual * 100)
               if _replay_total_actual else None)
_bt_valid = bt.dropna(subset=["actual_usd"]).tail(30) if not bt.empty else bt
_bt_total_actual = _bt_valid["actual_usd"].abs().sum() if not _bt_valid.empty else 0.0
wape_val = ((_bt_valid["abs_error_usd"].sum() / _bt_total_actual * 100)
            if _bt_total_actual else None)

with kpis[0]:
    last_display = money(last_7_actual) if last_7_actual is not None and last_7_actual >= 1_000 else (
        f"${last_7_actual:,.0f}" if last_7_actual is not None else "—"
    )
    metric("Last 7d actual", last_display)
with kpis[1]:
    next_display = money(next_7_total) if next_7_total is not None and next_7_total >= 1_000 else (
        f"${next_7_total:,.0f}" if next_7_total is not None else "—"
    )
    metric("Next 7d forecast", next_display)
delta = (next_7_total - last_7_actual
         if next_7_total is not None and last_7_actual is not None else None)
delta_pct_kpi = (delta / last_7_actual * 100) if delta is not None and last_7_actual else None
with kpis[2]:
    delta_display = f"${delta:+,.0f}" if delta is not None else "—"
    delta_label = f"{delta_pct_kpi:+.1f}%" if delta_pct_kpi is not None else None
    delta_good = None if delta is None or delta == 0 else delta < 0
    metric(
        "Forecast vs last 7d",
        delta_display,
        delta=delta_label,
        good=delta_good,
    )
with kpis[3]:
    wape_label = (
        "Trust check (training-fit WAPE)"
        if replay_wape is not None else "Rolling 30d WAPE"
    )
    wape_display = (
        f"{replay_wape:.1f}%" if replay_wape is not None
        else (f"{wape_val:.1f}%" if wape_val is not None else "—")
    )
    metric(wape_label, wape_display)

_last_7_avg = (last_7_actual / 7) if last_7_actual else None
_next_7_avg = (next_7_total / 7) if next_7_total is not None else None
render_live_cost_meter(
    daily_burn_usd=_last_7_avg,
    forecast_daily_usd=_next_7_avg,
    delta_pct=delta_pct_kpi,
    account_label=account_id,
)

_FORECAST_SPIKE_PCT = 20.0
if (
    delta_pct_kpi is not None
    and delta_pct_kpi > _FORECAST_SPIKE_PCT
    and next_7_total is not None
    and last_7_actual is not None
    and last_7_actual > 0
):
    _next_avg_txt = f"${_next_7_avg:,.0f}/day" if _next_7_avg is not None else "—"
    _last_avg_txt = f"${_last_7_avg:,.0f}/day" if _last_7_avg is not None else "—"
    render_notification_button(
        button_label="Notify forecast spike",
        state_key=f"dashboard::{account_id}",
        draft=NotificationDraft(
            title="Forecast spend spike",
            severity="High",
            reason=(
                f"Next 7-day forecast is {delta_pct_kpi:+.1f}% above the "
                f"recent 7-day baseline for account {account_id}."
            ),
            recipient="finops-team@example.com",
            subject=(
                f"[CostSense] Forecast spike — {account_id} "
                f"(+{delta_pct_kpi:.1f}% vs baseline)"
            ),
            body=(
                f"CostSense flagged a material forecast increase on the Dashboard.\n\n"
                f"Account: {account_id} ({active_profile})\n"
                f"Last 7d actual (avg): {_last_avg_txt}\n"
                f"Next 7d forecast (avg): {_next_avg_txt}\n"
                f"Change vs baseline: {delta_pct_kpi:+.1f}%\n\n"
                f"Please review the Dashboard forecast and recent PR / usage drivers."
            ),
            source_page="Dashboard",
            source_type="forecast_spike",
        ),
    )

st.divider()

section(
    "Cost forecast",
    "Past actuals from Cost Explorer plus the saved future forecast band.",
    kicker="Overview",
)

if _forecast_stale:
    callout(
        f"Saved forecast was run at cutoff **{latest['run_cutoff']}** but "
        f"controls are set to **{_chart_cutoff.isoformat()}**. "
        "Click **Run forecast** to refresh future dates.",
        tone="info",
    )

# Unified chart: past actuals + future band (split at cutoff)
if hist_past.empty and fc_future.empty:
    callout(
        f"No Cost Explorer data for account **{account_id}** in the last "
        f"{history_days} days through cutoff, and no future forecast on disk "
        f"after **{_chart_cutoff.isoformat()}**. If this is a fresh sandbox, "
        "spend may simply be $0 — otherwise click **Run forecast**.",
        tone="info",
    )
else:
    fig = go.Figure()
    _past_anchor = _past_anchor_y(
        hist_past, replay_past, _chart_cutoff, y_fit="predicted_usd",
    )

    if not fc_future.empty:
        future_x = fc_future["target_date"].tolist()
        upper_y = fc_future["upper_usd"].tolist()
        lower_y = fc_future["lower_usd"].tolist()
        adjusted_y = fc_future["adjusted_usd"].tolist()
        band_x, upper_plot = _future_trace_xy(
            _chart_cutoff, float(upper_y[0]), future_x, upper_y,
        )
        _, lower_plot = _future_trace_xy(
            _chart_cutoff, float(lower_y[0]), future_x, lower_y,
        )

        fig.add_trace(go.Scatter(
            x=band_x, y=upper_plot,
            mode="lines",
            line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=band_x, y=lower_plot,
            mode="lines", fill="tonexty",
            line=dict(width=0),
            fillcolor=_rgba(C.INFO, 0.15),
            showlegend=False, hoverinfo="skip",
        ))
        # Bridge cutoff → first future day (no hover — keeps past tooltip clean).
        if _past_anchor is not None and future_x:
            fig.add_trace(go.Scatter(
                x=[_chart_cutoff.isoformat(), future_x[0]],
                y=[_past_anchor, float(adjusted_y[0])],
                mode="lines",
                line=dict(color=C.INFO, width=2.5, dash="dot", shape="linear"),
                showlegend=False, hoverinfo="skip",
                legendgroup="future",
            ))
        fig.add_trace(go.Scatter(
            x=future_x,
            y=[float(v) for v in adjusted_y],
            mode="lines+markers", name="future prediction (next 7d)",
            line=dict(color=C.INFO, width=2.5, dash="dot", shape="linear"),
            legendgroup="future",
            legendrank=1,
            hovertemplate="future prediction: %{y:,.2f}<extra></extra>",
        ))
        fig.add_vline(
            x=_chart_cutoff.isoformat(), line_dash="dash", line_color=C.FAINT,
            annotation_text="cutoff", annotation_position="top",
        )
    elif not hist_past.empty:
        callout(
            "No saved future forecast after cutoff — click **Run forecast** "
            "in **Controls** above.",
            tone="info",
        )

    if not hist_past.empty:
        fig.add_trace(go.Scatter(
            x=hist_past["day"], y=hist_past["actual_usd"],
            mode="lines+markers", name="actual",
            line=dict(color=C.GOOD, width=2.5, shape="spline", smoothing=1.0),
            legendrank=3,
            hovertemplate="actual: %{y:,.2f}<extra></extra>",
        ))
    if not replay_past.empty:
        fig.add_trace(go.Scatter(
            x=replay_past["target_date"], y=replay_past["predicted_usd"],
            mode="lines+markers",
            name="fitted (training)",
            line=dict(color=C.INFO, width=2.5, shape="spline", smoothing=1.0),
            marker=dict(color=C.INFO, size=6),
            legendrank=2,
            hovertemplate="fitted (training): %{y:,.2f}<extra></extra>",
        ))
    _fig_layout = plotly_layout(height=440)
    _overview_x_start = _chart_data_start(
        (hist_past, "day"),
        (replay_past, "target_date"),
        default=cutoff - timedelta(days=history_days),
    )
    _fig_layout["xaxis"] = _forecast_chart_xaxis(_overview_x_start, _chart_x_end)
    fig.update_layout(
        **_fig_layout,
        yaxis_title="USD / day",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    if not replay_past.empty:
        with st.expander(
            f"Training fit detail  ·  {len(replay_df)} in-sample days"
            + (f"  ·  WAPE {replay_wape:.1f}%" if replay_wape is not None else ""),
            expanded=False,
        ):
            display = replay_df.copy()
            display["ape_pct"] = display["ape"].apply(
                lambda v: f"{v * 100:.1f}%" if v is not None else "—"
            )
            st.dataframe(
                display[["target_date", "predicted_usd", "actual_usd",
                         "abs_err", "ape_pct"]]
                .rename(columns={
                    "target_date": "Day",
                    "predicted_usd": "Fitted",
                    "actual_usd": "Actual",
                    "abs_err": "Abs error",
                    "ape_pct": "APE",
                }),
                use_container_width=True, hide_index=True,
            )

    if latest and not fc_future.empty:
        section(
            "Forecast detail",
            "Daily baseline, PR delta, and confidence band.",
            kicker="Breakdown",
        )
        st.dataframe(
            fc_future[["target_date", "baseline_usd", "pr_delta_usd",
                       "adjusted_usd", "lower_usd", "upper_usd"]]
            .rename(columns={
                "target_date": "Date",
                "baseline_usd": "Baseline",
                "pr_delta_usd": "PR Δ",
                "adjusted_usd": "Adjusted",
                "lower_usd": "Lower",
                "upper_usd": "Upper",
            }),
            use_container_width=True, hide_index=True,
        )



# ---------- Open PRs about to land ----------
if latest and latest.get("open_pr_scan", {}).get("count"):
    ops = latest["open_pr_scan"]
    st.divider()
    total = ops.get("total_expected_daily_delta_usd", 0.0)
    impact_pill = (
        "High" if total > 5 else "Low" if total < -5 else "Medium"
    )
    with st.container(border=True):
        st.markdown(pill(impact_pill), unsafe_allow_html=True)
        section(
            "PRs about to land — future cost pressure",
            f"**{ops['count']}** open PR(s) analyzed · probability-weighted expected "
            f"impact **${total:+,.2f}/day** once they merge. Each open PR uses the "
            "same deep AWS-tool pipeline as merged PRs; impact is weighted by merge "
            "likelihood (review state × CI status × draft/age).",
            kicker="Upcoming",
        )
    open_rows = []
    for p in ops.get("prs", []):
        arrow = "↗" if p["direction"] == "increase" else \
                "↘" if p["direction"] == "decrease" else "→"
        open_rows.append({
            "Repo": p["repo"],
            "PR": f"#{p['pr_number']}",
            "Title": p["pr_title"][:70],
            "Direction": f"{arrow} {p['direction']}",
            "If merged ($/d)": p["est_daily_delta_usd"],
            "Merge prob": p["merge_probability"],
            "Expected ($/d)": p["expected_daily_delta_usd"],
            "Expected merge": p["expected_merge_day"],
            "State": (
                ("draft " if p["is_draft"] else "")
                + (p["review_state"].lower() if p["review_state"] else "")
                + (f" · ci={p['checks_state'].lower()}"
                   if p["checks_state"] else "")
            ).strip(),
            "Link": p["pr_url"],
        })
    if open_rows:
        st.dataframe(
            pd.DataFrame(open_rows),
            use_container_width=True, hide_index=True,
            column_config={"Link": st.column_config.LinkColumn("Link")},
        )
        with st.expander("How the future forecast uses these"):
            st.markdown(
                "Each PR's `Expected ($/d)` = `If merged` × `Merge prob`.  \n"
                "The **future forecast line** on the chart has been bumped "
                "by the sum of these expected deltas starting on each PR's "
                "**Expected merge** date.  \n"
                "If a PR is approved + passing CI, its merge probability is "
                "high (~0.90) and its full impact lands on the forecast.  \n"
                "Draft or stale PRs contribute little to the forecast."
            )


# ---------- PRs driving this forecast ----------
if latest and latest.get("pr_scan"):
    pr_scan = latest["pr_scan"]
    st.divider()
    at_cutoff = latest.get("pr_delta_daily_usd_at_cutoff", 0.0)
    section(
        "PRs driving this forecast",
        f"Scanned {len(pr_scan.get('repos', []))} repo(s) · base "
        f"`{pr_scan.get('base_branch', '—')}` · "
        f"last {pr_scan.get('lookback_days', 0)}d · "
        f"cumulative PR delta at cutoff **${at_cutoff:+,.2f}/day** "
        "(list prices; upper bound)",
        kicker="Merged PRs",
    )
    impacts = pr_scan.get("impacts", [])
    if not impacts:
        callout(
            f"No IaC-touching PRs merged to `{pr_scan.get('base_branch')}` "
            f"in the last {pr_scan.get('lookback_days')} days. "
            "Forecast is baseline only.",
            tone="info",
        )
    else:
        analyzer_used = pr_scan.get("analyzer", "regex")
        model_used = pr_scan.get("llm_model") or ""
        st.caption(f"Analyzer: **{analyzer_used}**"
                   + (f" · model `{model_used}`" if model_used else ""))

        pr_rows = []
        change_rows = []
        for imp in impacts:
            pr_rows.append({
                "Repo": imp["repo"],
                "PR": f"#{imp['pr_number']}",
                "Title": imp["pr_title"],
                "Author": imp["author"],
                "Merged": imp["merged_at"][:10],
                "$/day Δ": imp["est_daily_delta_usd"],
                "LLM summary": imp.get("llm_summary", ""),
                "Link": imp["pr_url"],
            })
            for c in imp["changes"]:
                change_rows.append({
                    "Repo": imp["repo"],
                    "PR": f"#{imp['pr_number']}",
                    "Resource type": c["resource_type"],
                    "Name": c["resource_name"],
                    "Action": c["action"],
                    "Instance": c["instance_hint"] or "",
                    "$/day Δ": c["est_daily_delta_usd"],
                    "Source": c["price_source"],
                    "Rationale": c.get("rationale", ""),
                })

        st.dataframe(
            pd.DataFrame(pr_rows),
            use_container_width=True, hide_index=True,
            column_config={"Link": st.column_config.LinkColumn("Link")},
        )
        with st.expander(f"Per-resource breakdown  ·  {len(change_rows)} rows"):
            st.dataframe(
                pd.DataFrame(change_rows),
                use_container_width=True, hide_index=True,
            )
        llm_errors = [i for i in impacts if i.get("llm_error")]
        if llm_errors:
            with st.expander(f"LLM errors  ·  {len(llm_errors)}"):
                for i in llm_errors:
                    st.text(f"{i['repo']}#{i['pr_number']}: {i['llm_error']}")

# ---------- cost driver breakdown ----------
if svc_map and not selected_service:
    st.divider()
    section(
        "Cost drivers — what moved recently",
        "Compare last 7 days vs the prior 7 days. Big movers here that "
        "aren't in your PR list are likely non-code (console changes, "
        "trials expiring, RIs). No model can predict those from git.",
        kicker="Drivers",
    )

    rows = []
    for svc, points in svc_map.items():
        pts = sorted(points)  # (day_iso, amt)
        if len(pts) < 14:
            continue
        recent = sum(a for _, a in pts[-7:])
        prior = sum(a for _, a in pts[-14:-7])
        rows.append({
            "Service": svc,
            "Prior 7d ($)": round(prior, 2),
            "Last 7d ($)": round(recent, 2),
            "Δ ($)": round(recent - prior, 2),
            "Δ (%)": round((recent - prior) / prior * 100, 1) if prior else None,
        })
    if rows:
        drivers = pd.DataFrame(rows)
        drivers["_abs"] = drivers["Δ ($)"].abs()
        drivers = drivers.sort_values("_abs", ascending=False).drop(columns="_abs")
        st.dataframe(
            drivers.head(10),
            use_container_width=True, hide_index=True,
        )

st.divider()
section(
    "Backtest — predicted vs actual",
    "How past predictions tracked against realized spend.",
    kicker="Accuracy",
)

# Backtest chart source: persisted score_*.json; training fit for non-AWS models.
_aws_native = model_choice == "aws"
_BT_MIN_TRUST = 7
_bt_min_days = fit_lookback_days
bt_persisted = bt.copy()
bt_chart = bt_persisted.copy()
bt_source = "saved"
if not _aws_native and (bt_chart.empty or len(bt_chart) < _bt_min_days) and not replay_df.empty:
    bt_chart = _backtest_from_training_fit(replay_df)
    bt_source = "training fit (in-memory)"

_has_bt_view = not bt_chart.empty or (_aws_native and not hist_past.empty)

if not _has_bt_view:
    callout(
        "No backtest data yet — enable **Show training fit** in "
        "the controls (non-AWS models), or run **Run forecast** daily and "
        "wait for saved forecasts to age past 7 days.",
        tone="info",
    )
else:
    if _aws_native:
        st.caption(
            "AWS native has no in-sample training fit. Chart shows Cost Explorer "
            f"actuals plus **{bt_saved_count}** persisted score(s) from saved "
            "forecast JSON on disk."
        )
        if bt_saved_count < _BT_MIN_TRUST:
            callout(
                f"Only {bt_saved_count} scored day(s) on disk — run **Run forecast** "
                f"daily to build history. MAE/WAPE need at least {_BT_MIN_TRUST} scores.",
                tone="info",
            )
    elif bt_source != "saved":
        _src_note = (
            f"Source: {bt_source} — computed from live history, not persisted."
        )
        if bt_saved_count:
            _src_note += (
                f" Only {bt_saved_count} persisted score(s) on disk "
                f"(need {_bt_min_days} for saved-only chart)."
            )
        st.caption(_src_note)

    bt_metrics = (
        bt_persisted.sort_values("target_date").reset_index(drop=True)
        if _aws_native and not bt_persisted.empty
        else bt_chart.sort_values("target_date").reset_index(drop=True)
    )
    _metrics_ok = (not _aws_native) or bt_saved_count >= _BT_MIN_TRUST
    if not bt_metrics.empty and _metrics_ok:
        total_abs_err = bt_metrics["abs_error_usd"].sum()
        total_actual = bt_metrics["actual_usd"].abs().sum()
        wape_val_bt = (total_abs_err / total_actual * 100) if total_actual else None
    else:
        wape_val_bt = None

    bcols = st.columns(3, gap="medium")
    with bcols[0]:
        metric(
            "Days scored",
            bt_saved_count if _aws_native else len(bt_chart),
        )
    with bcols[1]:
        if bt_metrics.empty or not _metrics_ok:
            metric("MAE", "—")
        else:
            mae_val = bt_metrics["abs_error_usd"].mean()
            mae_display = money(mae_val) if mae_val >= 1_000 else f"${mae_val:.2f}"
            metric("MAE", mae_display)
    with bcols[2]:
        metric(
            "WAPE",
            f"{wape_val_bt:.1f}%" if wape_val_bt is not None else "—",
        )

    fig2 = go.Figure()
    bt_past = pd.DataFrame()
    bt_slice = pd.DataFrame()

    if _aws_native:
        if not hist_past.empty:
            fig2.add_trace(go.Scatter(
                x=hist_past["day"], y=hist_past["actual_usd"],
                mode="lines+markers", name="actual",
                line=dict(color=C.GOOD, width=2.5, shape="spline", smoothing=1.0),
                legendrank=4,
                hovertemplate="actual: %{y:,.2f}<extra></extra>",
            ))
        if not bt_persisted.empty:
            bt_dates = pd.to_datetime(bt_persisted["target_date"])
            bt_slice = bt_persisted.loc[
                bt_dates <= pd.Timestamp(_chart_cutoff)
            ].copy()
            pred_mode = "lines+markers" if len(bt_slice) > 1 else "markers"
            fig2.add_trace(go.Scatter(
                x=bt_slice["target_date"], y=bt_slice["predicted_usd"],
                mode=pred_mode, name="saved prediction",
                line=dict(color=C.INFO, width=2.5, shape="linear"),
                marker=dict(color=C.INFO, size=8),
                legendrank=3,
                hovertemplate="saved prediction: %{y:,.2f}<extra></extra>",
            ))
        bt_past = bt_slice
    else:
        bt_dates = pd.to_datetime(bt_chart["target_date"])
        bt_slice = bt_chart.loc[bt_dates <= pd.Timestamp(_chart_cutoff)].copy()
        bt_past = (
            _fit_past_through_cutoff(bt_slice, _chart_cutoff, "target_date")
            if not bt_slice.empty else bt_slice
        )
        fig2.add_trace(go.Scatter(
            x=bt_past["target_date"], y=bt_past["predicted_usd"],
            mode="lines+markers", name="fitted (training)",
            line=dict(color=C.INFO, width=2.5, shape="spline", smoothing=1.0),
            legendrank=3,
            hovertemplate="fitted (training): %{y:,.2f}<extra></extra>",
        ))
        fig2.add_trace(go.Scatter(
            x=bt_past["target_date"], y=bt_past["actual_usd"],
            mode="lines+markers", name="actual",
            line=dict(color=C.GOOD, width=2.5, shape="spline", smoothing=1.0),
            legendrank=4,
            hovertemplate="actual: %{y:,.2f}<extra></extra>",
        ))

    # Future forecast: only dates after cutoff; bridge closes any calendar gap.
    if not fc_future.empty:
        future_x = fc_future["target_date"].tolist()
        future_y = [float(v) for v in fc_future["adjusted_usd"].tolist()]
        upper_y = [float(v) for v in fc_future["upper_usd"].tolist()]
        lower_y = [float(v) for v in fc_future["lower_usd"].tolist()]
        if _aws_native:
            past_anchor = _past_anchor_y(
                hist_past, bt_past, _chart_cutoff,
                y_fit="predicted_usd", y_actual="actual_usd",
                day_col_fit="target_date", day_col_actual="day",
            )
        else:
            past_anchor = _value_at_cutoff(
                bt_past, _chart_cutoff, "target_date", "predicted_usd",
            )
        band_x, upper_plot = _future_trace_xy(
            _chart_cutoff, upper_y[0], future_x, upper_y,
        )
        _, lower_plot = _future_trace_xy(
            _chart_cutoff, lower_y[0], future_x, lower_y,
        )
        future_plot_x, future_plot_y = _future_trace_xy(
            _chart_cutoff, past_anchor, future_x, future_y,
        )

        fig2.add_trace(go.Scatter(
            x=band_x, y=upper_plot, mode="lines",
            line=dict(width=0),
            showlegend=False, hoverinfo="skip",
            legendrank=2,
        ))
        fig2.add_trace(go.Scatter(
            x=band_x, y=lower_plot, mode="lines", fill="tonexty",
            name="future forecast band",
            line=dict(width=0),
            fillcolor=_rgba(C.INFO, 0.15), hoverinfo="skip",
            legendrank=2,
        ))
        fig2.add_trace(go.Scatter(
            x=future_plot_x, y=future_plot_y,
            mode="lines+markers", name="future prediction (next 7d)",
            line=dict(color=C.INFO, width=2.5, dash="dot", shape="linear"),
            legendrank=1,
            hovertemplate="future prediction: %{y:,.2f}<extra></extra>",
        ))
        fig2.add_vline(
            x=_chart_cutoff.isoformat(),
            line_dash="dash", line_color=C.FAINT,
            annotation_text="now", annotation_position="top",
        )
    _bt_start_series: list = [(bt_past, "target_date")]
    if _aws_native and not hist_past.empty:
        _bt_start_series.insert(0, (hist_past, "day"))
    _backtest_x_start = _chart_data_start(
        *_bt_start_series,
        default=cutoff - timedelta(days=history_days),
    )
    _backtest_xaxis = _forecast_chart_xaxis(_backtest_x_start, _chart_x_end)
    _fig2_layout = plotly_layout(height=340)
    _fig2_layout["xaxis"] = _backtest_xaxis
    fig2.update_layout(
        **_fig2_layout,
        yaxis_title="USD / day",
        hovermode="x unified",
    )
    st.plotly_chart(fig2, use_container_width=True)

    if not bt_metrics.empty and _metrics_ok:
        err_roll = bt_metrics["abs_error_usd"].rolling(7, min_periods=1).sum()
        act_roll = bt_metrics["actual_usd"].abs().rolling(7, min_periods=1).sum()
        bt_metrics = bt_metrics.copy()
        bt_metrics["wape_pct"] = (err_roll / act_roll * 100).where(act_roll > 0)
        bt_wape_plot = bt_metrics.loc[
            pd.to_datetime(bt_metrics["target_date"]) <= pd.Timestamp(_chart_cutoff)
        ]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=bt_wape_plot["target_date"],
            y=bt_wape_plot["wape_pct"],
            mode="lines+markers", name="rolling 7-day WAPE",
            line=dict(color=C.BAD, width=2.5, shape="spline", smoothing=1.0),
        ))
        _fig3_layout = plotly_layout(height=260)
        _fig3_layout["xaxis"] = _backtest_xaxis
        fig3.update_layout(
            **_fig3_layout,
            yaxis_title="WAPE %",
            hovermode="x unified",
        )
        st.plotly_chart(fig3, use_container_width=True)


    st.plotly_chart(fig3, use_container_width=True)


# ---------- Future forecast summary (no LLM, computed from data) ----------
st.divider()
section(
    "What will happen next — and why",
    "Direction and drivers behind the next 7-day forecast.",
    kicker="Outlook",
)

if latest and latest.get("forecast"):
    fc_rows = latest["forecast"]
    future_total = sum(float(r.get("adjusted_usd", 0)) for r in fc_rows)
    future_avg = future_total / max(1, len(fc_rows))

    # Compare with the last 7 days of actuals
    last7_actual = hist_df.tail(7)["actual_usd"].sum() if not hist_df.empty else 0.0
    last7_avg = last7_actual / 7 if last7_actual else 0.0
    delta_pct = ((future_avg - last7_avg) / last7_avg * 100) if last7_avg else None

    # Direction summary
    if delta_pct is None:
        direction_pill = "Medium"
    elif delta_pct > 5:
        direction_pill = "High"
    elif delta_pct < -5:
        direction_pill = "Low"
    else:
        direction_pill = "Medium"

    future_avg_display = (
        money(future_avg) if future_avg >= 1_000 else f"${future_avg:,.0f}"
    )
    future_total_display = (
        money(future_total) if future_total >= 1_000 else f"${future_total:,.0f}"
    )
    with st.container(border=True):
        st.markdown(pill(direction_pill), unsafe_allow_html=True)
        st.markdown(
            f"### Next 7 days — est. **{future_avg_display}/day** "
            f"({future_total_display} total)"
        )

    # Concrete reasons pulled from the forecast payload
    tuned = latest.get("tuned_params") or {}
    dow_ratios = tuned.get("dow_ratios") or {}
    pr_impacts = (latest.get("pr_scan") or {}).get("impacts") or []
    nonzero_prs = [p for p in pr_impacts if p.get("est_daily_delta_usd", 0)]
    pr_delta_at_cutoff = float(latest.get("pr_delta_daily_usd_at_cutoff", 0.0))
    open_pr_scan = latest.get("open_pr_scan") or {}
    open_pr_expected = float(open_pr_scan.get(
        "total_expected_daily_delta_usd", 0.0
    ))

    st.markdown("**Why the graph looks the way it does:**")

    reasons: list[str] = []

    if latest.get("model") == "aws":
        reasons.append(
            "**AWS native forecast.** The future line comes from Cost "
            "Explorer `GetCostForecast` — AWS's statistical model on your "
            "billing history (as-of this API call). Enable **Include PR "
            "impact** to layer code-change deltas on top."
        )

    # Recent trajectory
    if last7_avg > 0 and delta_pct is not None:
        if abs(delta_pct) < 5:
            reasons.append(
                f"**Steady level.** Last 7 days averaged "
                f"\\${last7_avg:,.0f}/day; the forecast holds close to that "
                f"({delta_pct:+.1f}%) because no strong up- or down-trend is "
                "visible in recent history."
            )
        else:
            reasons.append(
                f"**Trend continues.** Last 7 days averaged "
                f"\\${last7_avg:,.0f}/day; the model projects "
                f"\\${future_avg:,.0f}/day next week ({delta_pct:+.1f}%) "
                "because the trajectory of recent days points in that direction."
            )

    # Day-of-week pattern
    if dow_ratios:
        highest = max(dow_ratios.items(), key=lambda kv: float(kv[1]))
        lowest = min(dow_ratios.items(), key=lambda kv: float(kv[1]))
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        hi_name = dow_names[int(highest[0])]
        lo_name = dow_names[int(lowest[0])]
        hi_pct = (float(highest[1]) - 1) * 100
        lo_pct = (float(lowest[1]) - 1) * 100
        if abs(hi_pct) > 5 or abs(lo_pct) > 5:
            reasons.append(
                f"**Weekly rhythm.** The model applies a day-of-week pattern "
                f"learned from the last 4 weeks: **{hi_name}** typically runs "
                f"{hi_pct:+.0f}% vs the account average, **{lo_name}** runs "
                f"{lo_pct:+.0f}%. That's why the forecast line wiggles rather "
                f"than being flat."
            )

    # Merged PRs already in effect
    if pr_delta_at_cutoff and abs(pr_delta_at_cutoff) > 0.5:
        n = len(nonzero_prs)
        reasons.append(
            f"**Merged PR impact ({n} PR(s)).** The forecast is offset by "
            f"**\\${pr_delta_at_cutoff:+,.2f}/day** because recent merged PRs "
            "changed AWS resources (Lambda memory, log volume, etc.). "
            "This offset carries forward into every day of the future forecast."
        )

    # Open PRs about to merge
    if open_pr_expected and abs(open_pr_expected) > 0.5:
        n_open = int(open_pr_scan.get("count", 0))
        reasons.append(
            f"**Upcoming PRs ({n_open} open).** Additional "
            f"**\\${open_pr_expected:+,.2f}/day** expected once open PRs merge "
            "(probability-weighted). The future line rises/falls slightly on "
            "each expected merge date."
        )

    # Confidence band
    band_widths = [float(r.get("upper_usd", 0)) - float(r.get("lower_usd", 0))
                   for r in fc_rows]
    avg_band = sum(band_widths) / max(1, len(band_widths))
    if future_avg > 0:
        band_pct = avg_band / future_avg * 100
        if band_pct > 40:
            reasons.append(
                f"**Wide uncertainty band (±\\${avg_band/2:,.0f}/day, "
                f"~{band_pct/2:.0f}%).** History was volatile, so the model "
                "isn't confident. Actual cost could land anywhere inside "
                "the shaded region — treat the number as a rough range, "
                "not a promise."
            )
        else:
            reasons.append(
                f"**Tight uncertainty band (±\\${avg_band/2:,.0f}/day, "
                f"~{band_pct/2:.0f}%).** History was stable, so the model "
                "is reasonably confident in this number."
            )

    if reasons:
        for r in reasons:
            st.markdown(f"- {r}")
    else:
        callout(
            "Not enough data yet to explain the forecast. Click "
            "**Run forecast** in **Controls** first.",
            tone="info",
        )
else:
    callout(
        "No forecast on disk yet. Click **Run forecast** in **Controls** "
        "to generate one.",
        tone="info",
    )
