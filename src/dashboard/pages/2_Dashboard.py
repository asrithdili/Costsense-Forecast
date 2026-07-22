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
from src.backtest.scorer import score_for_target
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
def _fetch_history(
    profile: str, cutoff_iso: str, days: int, service: str | None = None,
) -> pd.DataFrame:
    cutoff = date.fromisoformat(cutoff_iso)
    totals = fetch_daily_totals(
        cutoff - timedelta(days=days), cutoff, profile=profile, service=service,
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
        cutoff - timedelta(days=days), cutoff, profile=profile,
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


def _auto_score(account_id: str, profile: str) -> int:
    """For every forecast on disk older than 7 days, score any target dates
    that don't yet have a backtest file. Returns count of newly scored rows."""
    n = 0
    today = date.today()
    for f in sorted(predictions_dir(account_id).glob("forecast_*.json")):
        payload = json.loads(f.read_text())
        for row in payload.get("forecast", []):
            target = date.fromisoformat(row["target_date"])
            if target > today - timedelta(days=1):
                continue  # actual not yet available
            score_file = backtest_dir(account_id) / f"score_{target.isoformat()}.json"
            if score_file.exists():
                continue
            try:
                score_for_target(target, profile=profile)
                n += 1
            except Exception:  # noqa: BLE001
                pass
    return n


def _rgba(hex_color: str, alpha: float) -> str:
    """Hex token → rgba() string for Plotly fill colors."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


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
if do_forecast:
    _run_msg = (
        f"Fetching Cost Explorer, scanning PRs, fitting {model_choice}…"
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
                (f"Scanned {len(_repos_scanned)} repo(s). "
                 "Merged PRs shape the past baseline; open PRs bump the "
                 "future forecast weighted by merge probability."),
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

# opportunistically score any old forecasts whose targets have landed
try:
    _auto_score(account_id, active_profile)
except Exception:  # noqa: BLE001
    pass

bt = _load_backtest(account_id)
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
if show_replay and not hist_df.empty:
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


# KPI row
kpis = st.columns(4, gap="medium")
next_7_total = fc_df["adjusted_usd"].sum() if not fc_df.empty else None
last_7_actual = hist_df.tail(7)["actual_usd"].sum() if not hist_df.empty else None
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

st.divider()

section(
    "Cost forecast",
    "Past actuals from Cost Explorer plus the saved future forecast band.",
    kicker="Overview",
)

# Unified chart: past actuals + future band
if hist_df.empty and fc_df.empty:
    callout(
        f"No Cost Explorer data for account **{account_id}** in the last "
        f"{history_days} days, and no forecast on disk. If this is a fresh "
        f"sandbox, spend may simply be $0.",
        tone="info",
    )
else:
    fig = go.Figure()
    if not hist_df.empty:
        fig.add_trace(go.Scatter(
            x=hist_df["day"], y=hist_df["actual_usd"],
            mode="lines+markers", name="actual (Cost Explorer)",
            line=dict(color=C.BRAND, width=2.5, shape="spline",
                      smoothing=1.0),
        ))
    if not fc_df.empty:
        fig.add_trace(go.Scatter(
            x=fc_df["target_date"], y=fc_df["upper_usd"],
            mode="lines",
            line=dict(width=0, shape="spline", smoothing=1.0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=fc_df["target_date"], y=fc_df["lower_usd"],
            mode="lines", fill="tonexty", name="forecast interval",
            line=dict(width=0, shape="spline", smoothing=1.0),
            fillcolor=_rgba(C.BRAND, 0.15),
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=fc_df["target_date"], y=fc_df["baseline_usd"],
            mode="lines+markers", name="baseline forecast",
            line=dict(color=C.INFO, width=2, dash="dot",
                      shape="spline", smoothing=1.0),
        ))
        fig.add_trace(go.Scatter(
            x=fc_df["target_date"], y=fc_df["adjusted_usd"],
            mode="lines+markers", name="adjusted (baseline + PR delta)",
            line=dict(color=C.BRAND_DARK, width=2.5, shape="spline",
                      smoothing=1.0),
        ))
        fig.add_vline(
            x=latest["run_cutoff"], line_dash="dash", line_color=C.FAINT,
            annotation_text="cutoff", annotation_position="top",
        )
    else:
        callout(
            "No saved future forecast yet — click **Run forecast** in "
            "**Controls** above. Training fit below still shows how well "
            "the model tracks history.",
            tone="info",
        )

    if not pr_series_df.empty:
        fig.add_trace(go.Scatter(
            x=pr_series_df["day"], y=pr_series_df["pr_cum_usd"],
            mode="lines", name="PR-attributable ($/day)",
            line=dict(color=C.SEV["High"], width=1.5, dash="dot"),
            hovertemplate="%{x}<br>PR delta $%{y:,.2f}<extra></extra>",
        ))
    if not replay_df.empty:
        fig.add_trace(go.Scatter(
            x=replay_df["target_date"], y=replay_df["predicted_usd"],
            mode="markers",
            name="training fit (in-sample)",
            marker=dict(color=C.BAD, size=8, symbol="diamond",
                        line=dict(color="white", width=1)),
            customdata=replay_df[["origin", "horizon", "actual_usd",
                                  "abs_err"]].values,
            hovertemplate=("Day %{x}<br>"
                           "Fitted $%{y:,.2f}<br>"
                           "Actual $%{customdata[2]:,.2f}<br>"
                           "Trained at %{customdata[0]}<br>"
                           "Abs err $%{customdata[3]:,.2f}"
                           "<extra></extra>"),
        ))
    fig.update_layout(
        **plotly_layout(height=440),
        yaxis_title="USD / day",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    if not replay_df.empty:
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

    if latest:
        section(
            "Forecast detail",
            "Daily baseline, PR delta, and confidence band.",
            kicker="Breakdown",
        )
        st.dataframe(
            fc_df[["target_date", "baseline_usd", "pr_delta_usd",
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

# If we have saved daily backtest scores, use them. Otherwise fall back to the
# in-sample training fit we just computed.
bt_source = "saved"
if bt.empty and not replay_df.empty:
    bt = replay_df.rename(columns={
        "predicted_usd": "predicted_usd",
        "actual_usd": "actual_usd",
    }).copy()
    bt["abs_error_usd"] = bt["abs_err"]
    bt = bt.dropna(subset=["actual_usd"])
    bt_source = "training fit (in-memory)"

if bt.empty:
    callout(
        "No backtest data yet — enable **Show training fit** in "
        "the controls, or wait for saved forecasts to age past 7 days.",
        tone="info",
    )
else:
    if bt_source != "saved":
        st.caption(f"Source: {bt_source} — computed from live history, "
                   "not persisted.")
    bt = bt.sort_values("target_date").reset_index(drop=True)
    total_abs_err = bt["abs_error_usd"].sum()
    total_actual = bt["actual_usd"].abs().sum()
    wape_val_bt = (total_abs_err / total_actual * 100) if total_actual else None

    bcols = st.columns(3, gap="medium")
    with bcols[0]:
        metric("Days scored", len(bt))
    with bcols[1]:
        mae_val = bt["abs_error_usd"].mean()
        mae_display = money(mae_val) if mae_val >= 1_000 else f"${mae_val:.2f}"
        metric("MAE", mae_display)
    with bcols[2]:
        metric(
            "WAPE",
            f"{wape_val_bt:.1f}%" if wape_val_bt is not None else "—",
        )

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=bt["target_date"], y=bt["predicted_usd"],
        mode="lines+markers", name="fitted (training)",
        line=dict(color=C.BRAND_DARK, width=2.5, shape="spline", smoothing=1.0),
    ))
    fig2.add_trace(go.Scatter(
        x=bt["target_date"], y=bt["actual_usd"],
        mode="lines+markers", name="actual",
        line=dict(color=C.BRAND, width=2.5, shape="spline", smoothing=1.0),
    ))
    # Extend chart with the future 7-day forecast so past + future live on
    # one axis. Future comes from `latest["forecast"]` (already persisted by
    # the last "Run forecast" click), and we connect it visually by
    # prepending the last past-prediction point.
    if latest and latest.get("forecast"):
        fut_df = pd.DataFrame(latest["forecast"])
        if not fut_df.empty:
            last_past = bt.iloc[-1]
            future_x = [last_past["target_date"]] + fut_df["target_date"].tolist()
            future_y = [float(last_past["predicted_usd"])] + fut_df["adjusted_usd"].tolist()
            upper_y = [float(last_past["predicted_usd"])] + fut_df["upper_usd"].tolist()
            lower_y = [float(last_past["predicted_usd"])] + fut_df["lower_usd"].tolist()

            fig2.add_trace(go.Scatter(
                x=future_x, y=upper_y, mode="lines",
                line=dict(width=0, shape="spline", smoothing=1.0),
                showlegend=False, hoverinfo="skip",
            ))
            fig2.add_trace(go.Scatter(
                x=future_x, y=lower_y, mode="lines", fill="tonexty",
                name="future forecast band",
                line=dict(width=0, shape="spline", smoothing=1.0),
                fillcolor=_rgba(C.BRAND, 0.15), hoverinfo="skip",
            ))
            fig2.add_trace(go.Scatter(
                x=future_x, y=future_y,
                mode="lines+markers", name="future prediction (next 7d)",
                line=dict(color=C.BRAND_DARK, width=2.5, dash="dot",
                          shape="spline", smoothing=1.0),
            ))
            fig2.add_vline(
                x=str(last_past["target_date"]),
                line_dash="dash", line_color=C.FAINT,
                annotation_text="now", annotation_position="top",
            )
    fig2.update_layout(
        **plotly_layout(height=340),
        yaxis_title="USD / day",
        hovermode="x unified",
    )
    st.plotly_chart(fig2, use_container_width=True)

    err_roll = bt["abs_error_usd"].rolling(7, min_periods=1).sum()
    act_roll = bt["actual_usd"].abs().rolling(7, min_periods=1).sum()
    bt["wape_pct"] = (err_roll / act_roll * 100).where(act_roll > 0)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=bt["target_date"],
        y=bt["wape_pct"],
        mode="lines+markers", name="rolling 7-day WAPE",
        line=dict(color=C.BAD, width=2.5, shape="spline", smoothing=1.0),
    ))
    fig3.update_layout(
        **plotly_layout(height=260),
        yaxis_title="WAPE %",
        hovermode="x unified",
    )
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
