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
from src.forecast.backtest_replay import walk_forward
from src.pipeline.paths import actuals_dir, backtest_dir, predictions_dir
from src.pipeline.run_daily import run as run_pipeline
from src.pr_scanner.repos import (
    gh_login,
    gh_orgs,
    recent_base_branches,
    repo_default_branch,
    repos_with_user_prs,
)
from src.dashboard.nav import inject_css, top_bar


st.set_page_config(page_title="CostSense · forecast", layout="wide", page_icon="💸")
inject_css()


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


# ---------- sidebar ----------

# ---------- top control bar ----------

with st.spinner("Resolving profiles…"):
    profiles: list[ProfileInfo] = resolve_all()
reachable = [p for p in profiles if p.account_id]
if not reachable:
    st.error("No reachable AWS profiles. Run `aws sso login` first, then reload.")
    st.stop()

labels = [p.label for p in reachable]
picked_label = st.session_state.get("dash_profile", labels[0])
if picked_label not in labels:
    picked_label = labels[0]
chosen = reachable[labels.index(picked_label)]
active_profile = chosen.profile
account_id = chosen.account_id

# Read defaults for other controls up front so the header can reflect them.
_cutoff = st.session_state.get("dash_cutoff", date.today())
_history_days = st.session_state.get("dash_history_days", 90)
_model_choice = st.session_state.get("dash_model", "ewm")

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

# GitHub org + repos
try:
    _gh_user = gh_login()
    orgs = list(gh_orgs())
except Exception:  # noqa: BLE001
    _gh_user = "?"
    orgs = []

default_org_idx = 0
if orgs:
    for _i, o in enumerate(orgs):
        if o == "DiligentCorp":
            default_org_idx = _i
            break
    gh_org_default = orgs[default_org_idx]
else:
    gh_org_default = "DiligentCorp"
gh_org = st.session_state.get("dash_gh_org", gh_org_default)

try:
    suggested_full = list(repos_with_user_prs(gh_org)) if gh_org else []
except Exception:  # noqa: BLE001
    suggested_full = []
short_names = [r.split("/", 1)[-1] for r in suggested_full]

from src.pr_scanner.profile_repo_match import match_repos as _match
default_repos = _match(active_profile, short_names) or short_names

svc_hdr = _selected_service or "All services"
header = (f"Controls  ·  Account: {picked_label}  ·  "
          f"Cutoff: {_cutoff.isoformat()}  ·  "
          f"History: {_history_days}d  ·  Scope: {svc_hdr}  ·  "
          f"Model: {_model_choice}")

with top_bar(header):
    # Row 1 — AWS
    r1c1, r1c2, r1c3, r1c4 = st.columns([3, 2, 2, 2])
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
        model_choice = st.selectbox(
            "Forecast model", options=["ewm", "prophet"],
            index=0 if _model_choice == "ewm" else 1,
            key="dash_model",
            help="EWM adapts to level shifts fast (default). Prophet "
                 "handles weekly seasonality when history is stable.",
        )

    # Row 2 — service filter
    r2c1, _ = st.columns([6, 6])
    with r2c1:
        pick_svc = st.selectbox(
            "Service filter", svc_options,
            index=svc_options.index(_svc_pick),
            key="dash_service",
            help="Forecast one service at a time for cleaner attribution.",
        )
    selected_service = (None if pick_svc.startswith("(all")
                        else pick_svc.split("  —  ")[0])

    # Row 3 — GitHub
    r3c1, r3c2, r3c3, r3c4 = st.columns([2, 4, 2, 2])
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
        picked_short = st.multiselect(
            "Repos", options=short_names, default=default_repos,
            key="dash_repos",
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
            )
    with r3c4:
        pr_lookback = st.slider(
            "PR lookback (d)", 3, 30, 14, step=1, key="dash_pr_lookback",
        )

    # Row 4 — PR analyzer / model + backtest controls
    r4c1, r4c2, r4c3, r4c4 = st.columns([2, 3, 2, 2])
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
    with r4c3:
        show_replay = st.checkbox(
            "Show backtest", value=True, key="dash_show_backtest",
            help="Walk-forward past predictions on the chart.",
        )
    with r4c4:
        n_origins = st.slider(
            "Backtest origins", 2, 12, 6, step=1,
            disabled=not show_replay, key="dash_n_origins",
        )

    r5c1, _ = st.columns([2, 10])
    with r5c1:
        stride_days = st.slider(
            "Stride (d)", 3, 14, 7, step=1,
            disabled=not show_replay, key="dash_stride",
        )

do_forecast = False


# ---------- main pane ----------

st.title("CostSense — daily cost forecast")
svc_label = f"**{selected_service}**" if selected_service else "**All services**"
st.info(f"Live account: **{account_id}** via profile `{active_profile}`  ·  "
        f"scope: {svc_label}  ·  cutoff {cutoff.isoformat()}  ·  "
        f"history {history_days}d")


# History — always fetched live
try:
    hist_df = _fetch_history(
        active_profile, cutoff.isoformat(), history_days,
        service=selected_service,
    )
except Exception as e:  # noqa: BLE001
    st.error(f"Cost Explorer fetch failed: {e}")
    st.code(traceback.format_exc())
    st.stop()


# On-demand forecast run
if do_forecast:
    with st.spinner(f"Fetching Cost Explorer, scanning PRs, fitting {model_choice}…"):
        try:
            out = run_pipeline(
                cutoff=cutoff, profile=active_profile,
                history_days=history_days,
                repos=selected_repos or None,
                base_branch=base_branch,
                pr_lookback_days=pr_lookback,
                analyzer=analyzer_choice,
                llm_model=llm_model_choice if analyzer_choice != "regex" else None,
                service=selected_service,
                model=model_choice,
            )
            st.success(f"Wrote {Path(out).name}")
            st.cache_data.clear()
        except Exception as e:  # noqa: BLE001
            st.error(f"Pipeline failed: {e}")
            st.code(traceback.format_exc())


latest = _load_latest_forecast(account_id, service=selected_service)

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

# Reconstruct PrSteps from the saved impacts so walk-forward can use them.
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

# Walk-forward replay for "past predictions" overlay + trust metric
replay_points: list = []
replay_df = pd.DataFrame()
if show_replay and not hist_df.empty:
    with st.spinner(f"Retraining {model_choice} at {n_origins} past origins…"):
        try:
            hist_for_replay = pd.DataFrame({
                "day": pd.to_datetime(hist_df["day"]),
                "amount_usd": hist_df["actual_usd"],
            })
            replay_points = walk_forward(
                hist_for_replay,
                end=cutoff,
                n_origins=n_origins,
                stride_days=stride_days,
                horizon_days=7,
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
            st.warning(f"Walk-forward replay failed: {e}")


# KPI row
kpis = st.columns(4)
next_7_total = fc_df["adjusted_usd"].sum() if not fc_df.empty else None
last_7_actual = hist_df.tail(7)["actual_usd"].sum() if not hist_df.empty else None
mape_val = (bt["ape"].dropna().tail(30).mean() * 100
            if not bt.empty and bt["ape"].notna().any() else None)
replay_mape = (replay_df["ape"].dropna().mean() * 100
               if not replay_df.empty and replay_df["ape"].notna().any() else None)
# WAPE is the trustworthy version when actuals include near-zero days.
_replay_valid = replay_df.dropna(subset=["actual_usd"]) if not replay_df.empty else replay_df
_replay_total_actual = _replay_valid["actual_usd"].abs().sum() if not _replay_valid.empty else 0.0
replay_wape = ((_replay_valid["abs_err"].sum() / _replay_total_actual * 100)
               if _replay_total_actual else None)

kpis[0].metric("Last 7d actual",
               f"${last_7_actual:,.0f}" if last_7_actual is not None else "—")
kpis[1].metric("Next 7d forecast",
               f"${next_7_total:,.0f}" if next_7_total is not None else "—")
delta = (next_7_total - last_7_actual
         if next_7_total is not None and last_7_actual is not None else None)
kpis[2].metric(
    "Forecast vs last 7d",
    f"${delta:+,.0f}" if delta is not None else "—",
    delta=f"{(delta / last_7_actual * 100):+.1f}%"
    if delta is not None and last_7_actual else None,
)
kpis[3].metric(
    "Trust check (walk-forward WAPE)"
    if replay_wape is not None else "Rolling 30d MAPE",
    f"{replay_wape:.1f}%" if replay_wape is not None
    else (f"{mape_val:.1f}%" if mape_val is not None else "—"),
    help="WAPE = Σ|err| / Σ|actual| across all past predictions. Robust "
         "when daily spend is near zero (unlike MAPE). Lower = better.",
)

st.divider()


# Unified chart: past actuals + future band
if hist_df.empty and fc_df.empty:
    st.info(f"No Cost Explorer data for account **{account_id}** in the last "
            f"{history_days} days, and no forecast on disk. If this is a fresh "
            f"sandbox, spend may simply be $0.")
else:
    fig = go.Figure()
    if not hist_df.empty:
        fig.add_trace(go.Scatter(
            x=hist_df["day"], y=hist_df["actual_usd"],
            mode="lines+markers", name="actual (Cost Explorer)",
            line=dict(color="#2E86AB", width=2.5, shape="spline",
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
            fillcolor="rgba(160,120,220,0.20)",
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=fc_df["target_date"], y=fc_df["baseline_usd"],
            mode="lines+markers", name="baseline forecast",
            line=dict(color="#A17DB5", width=2, dash="dot",
                      shape="spline", smoothing=1.0),
        ))
        fig.add_trace(go.Scatter(
            x=fc_df["target_date"], y=fc_df["adjusted_usd"],
            mode="lines+markers", name="adjusted (baseline + PR delta)",
            line=dict(color="#7B3F99", width=2.5, shape="spline",
                      smoothing=1.0),
        ))
        fig.add_vline(
            x=latest["run_cutoff"], line_dash="dash", line_color="#888",
            annotation_text="cutoff", annotation_position="top",
        )
    else:
        st.info("No saved future forecast yet — click **Run forecast** in the "
                "sidebar. Past predictions below still validate the model.")

    if not pr_series_df.empty:
        fig.add_trace(go.Scatter(
            x=pr_series_df["day"], y=pr_series_df["pr_cum_usd"],
            mode="lines", name="PR-attributable ($/day)",
            line=dict(color="#E27D60", width=1.5, dash="dot"),
            hovertemplate="%{x}<br>PR delta $%{y:,.2f}<extra></extra>",
        ))
    if not replay_df.empty:
        fig.add_trace(go.Scatter(
            x=replay_df["target_date"], y=replay_df["predicted_usd"],
            mode="markers",
            name="past prediction (walk-forward)",
            marker=dict(color="#C0504D", size=8, symbol="diamond",
                        line=dict(color="white", width=1)),
            customdata=replay_df[["origin", "horizon", "actual_usd",
                                  "abs_err"]].values,
            hovertemplate=("Target %{x}<br>"
                           "Predicted $%{y:,.2f}<br>"
                           "Actual $%{customdata[2]:,.2f}<br>"
                           "Origin %{customdata[0]} (h+%{customdata[1]})<br>"
                           "Abs err $%{customdata[3]:,.2f}"
                           "<extra></extra>"),
        ))
    fig.update_layout(
        height=440, margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="USD / day", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    if not replay_df.empty:
        with st.expander(
            f"Walk-forward detail  ·  {len(replay_df)} past predictions"
            + (f"  ·  MAPE {replay_mape:.1f}%" if replay_mape is not None else ""),
            expanded=False,
        ):
            display = replay_df.copy()
            display["ape_pct"] = display["ape"].apply(
                lambda v: f"{v * 100:.1f}%" if v is not None else "—"
            )
            st.dataframe(
                display[["origin", "target_date", "horizon",
                         "predicted_usd", "actual_usd", "abs_err", "ape_pct"]]
                .rename(columns={
                    "origin": "Origin (train cutoff)",
                    "target_date": "Target",
                    "horizon": "Days ahead",
                    "predicted_usd": "Predicted",
                    "actual_usd": "Actual",
                    "abs_err": "Abs error",
                    "ape_pct": "APE",
                }),
                use_container_width=True, hide_index=True,
            )

    if latest:
        st.subheader("Forecast detail")
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
    st.subheader("PRs about to land — future cost pressure")
    total = ops.get("total_expected_daily_delta_usd", 0.0)
    arrow_color = st.error if total > 5 else st.success if total < -5 else st.info
    arrow_color(
        f"**{ops['count']}** open PR(s) analyzed  ·  "
        f"probability-weighted expected impact **${total:+,.2f}/day** "
        "once they merge"
    )
    st.caption("Each open PR is analyzed with the same deep AWS-tool pipeline "
               "as merged PRs. Impact is weighted by merge likelihood "
               "(review state × CI status × draft/age).")
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
    st.subheader("PRs driving this forecast")
    at_cutoff = latest.get("pr_delta_daily_usd_at_cutoff", 0.0)
    st.caption(
        f"Scanned {len(pr_scan.get('repos', []))} repo(s) · base "
        f"`{pr_scan.get('base_branch', '—')}` · "
        f"last {pr_scan.get('lookback_days', 0)}d · "
        f"cumulative PR delta at cutoff **${at_cutoff:+,.2f}/day** "
        "(list prices; upper bound)"
    )
    impacts = pr_scan.get("impacts", [])
    if not impacts:
        st.info(f"No IaC-touching PRs merged to `{pr_scan.get('base_branch')}` "
                f"in the last {pr_scan.get('lookback_days')} days. "
                "Forecast is baseline only.")
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
    st.subheader("Cost drivers — what moved recently")
    st.caption("Compare last 7 days vs the prior 7 days. Big movers here that "
               "aren't in your PR list are likely non-code (console changes, "
               "trials expiring, RIs). No model can predict those from git.")

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
st.subheader("Backtest — predicted vs actual")

# If we have saved daily backtest scores, use them. Otherwise fall back to the
# walk-forward replay we just computed, which gives the same shape.
bt_source = "saved"
if bt.empty and not replay_df.empty:
    bt = replay_df.rename(columns={
        "predicted_usd": "predicted_usd",
        "actual_usd": "actual_usd",
    }).copy()
    bt["abs_error_usd"] = bt["abs_err"]
    bt = bt.dropna(subset=["actual_usd"])
    bt_source = "walk-forward (in-memory)"

if bt.empty:
    st.info("No backtest data yet — enable **Show walk-forward backtest** in "
            "the sidebar, or wait for saved forecasts to age past 7 days.")
else:
    if bt_source != "saved":
        st.caption(f"Source: {bt_source} — computed from live history, "
                   "not persisted.")
    bt = bt.sort_values("target_date").reset_index(drop=True)
    # WAPE = sum(|err|) / sum(actual). Robust to near-zero days where MAPE
    # explodes. This is the honest number when spend is sparse.
    total_abs_err = bt["abs_error_usd"].sum()
    total_actual = bt["actual_usd"].abs().sum()
    wape_val = (total_abs_err / total_actual * 100) if total_actual else None

    bcols = st.columns(4)
    bcols[0].metric("Days scored", len(bt))
    bcols[1].metric("MAE", f"${bt['abs_error_usd'].mean():.2f}")
    bcols[2].metric(
        "WAPE",
        f"{wape_val:.1f}%" if wape_val is not None else "—",
        help="Weighted APE = Σ|err| / Σ|actual|. Robust when daily spend "
             "is near zero — use this instead of MAPE for sandbox accounts.",
    )
    bcols[3].metric(
        "MAPE (raw)",
        f"{bt['ape'].dropna().mean() * 100:.1f}%"
        if bt["ape"].notna().any() else "—",
        help="Simple mean(|err| / actual). Blows up on near-zero days; "
             "WAPE is more trustworthy.",
    )

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=bt["target_date"], y=bt["predicted_usd"],
        mode="lines+markers", name="past prediction",
        line=dict(color="#7B3F99", width=2.5, shape="spline", smoothing=1.0),
    ))
    fig2.add_trace(go.Scatter(
        x=bt["target_date"], y=bt["actual_usd"],
        mode="lines+markers", name="actual",
        line=dict(color="#2E86AB", width=2.5, shape="spline", smoothing=1.0),
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
                fillcolor="rgba(160,120,220,0.20)", hoverinfo="skip",
            ))
            fig2.add_trace(go.Scatter(
                x=future_x, y=future_y,
                mode="lines+markers", name="future prediction (next 7d)",
                line=dict(color="#7B3F99", width=2.5, dash="dot",
                          shape="spline", smoothing=1.0),
            ))
            fig2.add_vline(
                x=str(last_past["target_date"]),
                line_dash="dash", line_color="#888",
                annotation_text="now", annotation_position="top",
            )
    fig2.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10),
                       yaxis_title="USD / day", hovermode="x unified",
                       legend=dict(orientation="h", yanchor="bottom",
                                   y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig2, use_container_width=True)

    bt["ape_pct"] = bt["ape"] * 100
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=bt["target_date"],
        y=bt["ape_pct"].rolling(7, min_periods=1).mean(),
        mode="lines+markers", name="rolling 7-day MAPE",
        line=dict(color="#C0504D", width=2.5, shape="spline", smoothing=1.0),
    ))
    fig3.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10),
                       yaxis_title="MAPE %", hovermode="x unified")
    st.plotly_chart(fig3, use_container_width=True)


# ---------- Future forecast summary (no LLM, computed from data) ----------
st.divider()
st.subheader("What will happen next — and why")

if latest and latest.get("forecast"):
    fc_rows = latest["forecast"]
    future_total = sum(float(r.get("adjusted_usd", 0)) for r in fc_rows)
    future_avg = future_total / max(1, len(fc_rows))

    # Compare with the last 7 days of actuals
    last7_actual = hist_df.tail(7)["actual_usd"].sum() if not hist_df.empty else 0.0
    last7_avg = last7_actual / 7 if last7_actual else 0.0
    delta_pct = ((future_avg - last7_avg) / last7_avg * 100) if last7_avg else None

    # Direction banner
    if delta_pct is None:
        direction, arrow, color = "flat", "→", st.info
    elif delta_pct > 5:
        direction, arrow, color = "up", "↗", st.warning
    elif delta_pct < -5:
        direction, arrow, color = "down", "↘", st.success
    else:
        direction, arrow, color = "flat", "→", st.info

    color(f"### {arrow} Next 7 days — est. **\\${future_avg:,.0f}/day** "
          f"(\\${future_total:,.0f} total)")

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
        st.info("Not enough data yet to explain the forecast. Click "
                "**Run forecast** in the sidebar first.")
else:
    st.info("No forecast on disk yet. Click **Run forecast** in the sidebar "
            "to generate one.")
