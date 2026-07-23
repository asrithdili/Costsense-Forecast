"""Org-Level Impact (v2).

Answers "where is the money, where is it moving, who owns it, and what do I
do" rather than just "what are the numbers." Key differences from v1:

  1. Loads on arrival — controls refine a result that is already on screen.
  2. Projection-led — the hero number is projected month-end, not a
     backward-looking window total.
  3. Ownership — group by team / OU / environment / account / service.
     Account IDs are not a language leadership speaks.
  4. Movers ranked by dollars with a materiality floor. Percent is displayed
     but never ranked on.
  5. Small-N mode — under 3 accounts with spend, the movers section is
     replaced by a service breakdown so the page never renders an empty
     column.
  6. Freshness — every render carries `data_through` (Cost Explorer lags ~24h).
  7. Routed — each account drills into service mix + two actions that carry
     context: Ask CostSense about this account and View anomalies.

Data source is auto-selected: the real Cost Explorer provider is tried first
and, on failure, we fall back to a deterministic demo provider so the page
still shows something on sandbox / non-payer profiles.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.aws.org_impact_data import (  # noqa: E402
    AccountSpend,
    OrgSpend,
    auto_provider,
)
from src.aws.profiles import resolve_all  # noqa: E402
from src.dashboard.costsense_theme import (  # noqa: E402
    C, callout, metric, money, plotly_layout, section,
)
from src.dashboard.nav import (  # noqa: E402
    inject_css, render_sidebar_footer, render_sidebar_header, top_bar,
)


GROUP_DIMENSIONS = {
    "Team": "team",
    "Account": "account",
    "OU": "ou",
    "Environment": "environment",
    "Service": "service",
}
WINDOWS = {"30 days": 30, "60 days": 60, "90 days": 90}
SMALL_N = 3  # below this, movers analysis is not meaningful


# ============================================================================
# PAGE SHELL
# ============================================================================
st.set_page_config(page_title="CostSense · Org Impact", layout="wide")
inject_css()
render_sidebar_header()

section(
    "Org-level impact",
    "Where org spend sits, where it is moving, and who owns it. Sourced "
    "from Cost Explorer's LINKED_ACCOUNT dimension on the management "
    "profile. Falls back to a demo view for non-payer profiles.",
    kicker="Organization",
)


with st.spinner("Resolving profiles…"):
    profiles = [p for p in resolve_all() if p.account_id]
if not profiles:
    callout("No AWS profiles reachable.", tone="error")
    st.stop()

# Prefer a payer/control-tower profile as the default if one exists.
default_idx = 0
for i, p in enumerate(profiles):
    if any(hint in p.profile.lower() for hint in ("control-tower", "tower", "master", "payer")):
        default_idx = i
        break
labels = [p.label for p in profiles]

# ---------- top control bar ----------
picked_label = st.session_state.get("orgv2_profile", labels[default_idx])
if picked_label not in labels:
    picked_label = labels[default_idx]
picked_window = st.session_state.get("orgv2_window", "30 days")
picked_floor = int(st.session_state.get("orgv2_floor", 250))

header = (f"Controls  ·  Profile: {picked_label}  ·  "
          f"Window: {picked_window}  ·  Floor: ${picked_floor}")
with top_bar(header):
    c1, c2, c3, c4 = st.columns(
        [3, 2, 2, 1], gap="medium", vertical_alignment="bottom",
    )
    with c1:
        picked_label = st.selectbox(
            "Management profile", labels,
            index=labels.index(picked_label),
            key="orgv2_profile",
            help="AWS Organizations management (payer) account. Non-payer "
                 "profiles will drop back to the demo view.",
        )
    with c2:
        picked_window = st.selectbox(
            "Window", list(WINDOWS.keys()),
            index=list(WINDOWS.keys()).index(picked_window),
            key="orgv2_window",
        )
    with c3:
        picked_floor = st.number_input(
            "Mover floor ($)", min_value=0, max_value=10_000,
            value=picked_floor, step=50, key="orgv2_floor",
            help="Week-over-week changes smaller than this are suppressed "
                 "as noise. Percent swings on tiny accounts are not signal.",
        )
    with c4:
        if st.button("Refresh", key="orgv2_refresh",
                     use_container_width=True):
            st.cache_data.clear()
            st.rerun()

active = profiles[labels.index(picked_label)]
window_days = WINDOWS[picked_window]
min_abs = float(picked_floor)


with st.sidebar:
    render_sidebar_footer(
        active_profile=active.profile,
        account_id=active.account_id,
        extra_rows=[
            ("Window", f"{window_days}d"),
            ("Mover floor", f"${int(min_abs):,}"),
        ],
    )


# ============================================================================
# DATA FETCH (cached)
# ============================================================================
_PROVIDER = auto_provider(budget_monthly=None)


@st.cache_data(ttl=900, show_spinner=False)
def _load(_provider_ref, provider_key: str,
          profile: str, window_days: int) -> OrgSpend:
    """Cost Explorer bills per request → cache for 15 minutes.

    The leading underscore keeps the (unhashable) provider object out of
    the hash key; `provider_key` carries its identity explicitly so
    swapping providers doesn't silently return the previous one's data.
    """
    return _provider_ref.fetch(profile, window_days)


with st.spinner(f"Fetching org spend via `{active.profile}` "
                f"({window_days}d)…"):
    try:
        org = _load(_PROVIDER, _PROVIDER.cache_key, active.profile,
                    window_days)
    except Exception as e:  # noqa: BLE001
        callout(f"Failed to load org spend: {e}", tone="error")
        st.stop()


# ============================================================================
# FRESHNESS PILL
# ============================================================================
_using_demo = "demo fallback" in (org.profile or "")
_pill_bg = C.BRAND_SOFT if not _using_demo else C.SEV_SOFT["Medium"]
_pill_fg = C.BRAND_DARK if not _using_demo else C.SEV["Medium"]
_pill_text = (
    f"Data through {org.data_through:%d %b %Y}"
    f"<span style='opacity:.7;font-weight:500;'> · Cost Explorer lags ~24h"
    f"</span>"
    if not _using_demo else
    f"Demo data · profile `{active.profile}` isn't a payer account"
)
st.markdown(
    f"<div style='display:inline-flex;align-items:center;gap:8px;"
    f"background:{_pill_bg};color:{_pill_fg};padding:5px 12px;"
    f"border-radius:999px;font-size:.8rem;font-weight:600;"
    f"margin-bottom:14px;'>{_pill_text}</div>",
    unsafe_allow_html=True,
)


# ============================================================================
# ANSWER BAND
# ============================================================================
hero_col, rest_col = st.columns([1.3, 2], gap="medium")

with hero_col:
    vs_prior = org.projection_vs_prior_month_pct
    budget_pct = org.budget_used_pct
    bits: List[str] = []
    if vs_prior is not None:
        bits.append(f"{vs_prior:+.1f}% vs last month")
    if budget_pct is not None:
        bits.append(f"{budget_pct:.0f}% of budget")
    over = ((vs_prior is not None and vs_prior > 0)
            or (budget_pct is not None and budget_pct > 100))
    metric(
        "Projected month-end", money(org.projected_month_end),
        delta=" · ".join(bits) or None,
        good=None if not bits else not over,
    )
    st.caption(
        f"Run-rate from the last 7 days × "
        f"{org.days_remaining_in_month} days remaining. Reproducible on a "
        f"napkin — swap in ce:GetCostForecast for AWS's model instead."
    )

with rest_col:
    a, b, c = st.columns(3, gap="medium")
    with a:
        metric("Month to date", money(org.month_to_date))
    with b:
        metric(
            "Accounts with spend",
            f"{len(org.accounts_with_spend)} of {org.linked_accounts}",
        )
    with c:
        metric("Concentration", f"Top 3 = {org.concentration(3):.0f}%")

st.divider()


# ============================================================================
# DAILY SPEND TREND
# ============================================================================
def _render_trend(org: OrgSpend) -> None:
    st.markdown("##### Daily spend")
    st.caption(
        "Top accounts stacked, remainder rolled into Other. A trend number "
        "without a shape can't be explained to a reviewer."
    )
    series = org.daily_series(top_n=5)
    if not series or not org.dates:
        callout("No daily series available for this window.", tone="info")
        return

    palette = [C.BRAND, C.INFO, C.SEV["Medium"], C.SEV["High"],
               C.BRAND_DARK, C.FAINT]
    fig = go.Figure()
    for i, (label, values) in enumerate(series.items()):
        fig.add_trace(go.Scatter(
            x=org.dates, y=values, name=label, mode="lines",
            stackgroup="spend",
            line=dict(width=0.5, color=palette[i % len(palette)]),
            fillcolor=palette[i % len(palette)],
            hovertemplate="<b>%{fullData.name}</b><br>"
                          "%{x|%d %b}<br>$%{y:,.0f}<extra></extra>",
        ))
    lay = plotly_layout(height=280)
    lay["yaxis_title"] = "Daily spend (USD)"
    fig.update_layout(**lay)
    fig.update_traces(opacity=0.9)
    st.plotly_chart(fig, use_container_width=True)


_render_trend(org)
st.divider()


# ============================================================================
# OWNERSHIP GROUPING
# ============================================================================
def _render_ownership(org: OrgSpend) -> None:
    st.markdown("##### Where the money sits")
    dim_label = st.radio(
        "Group by", list(GROUP_DIMENSIONS.keys()),
        horizontal=True, key="orgv2_group_by",
    )
    grouped = org.group_by(GROUP_DIMENSIONS[dim_label])
    if not grouped:
        callout("Nothing to group for this window.", tone="info")
        return

    labels_ = list(grouped.keys())[:12]
    values = [grouped[k] for k in labels_]
    colors = [C.FAINT if k in ("Unallocated", "unknown") else C.BRAND
              for k in labels_]

    fig = go.Figure(go.Bar(
        y=labels_[::-1], x=values[::-1], orientation="h",
        marker=dict(color=colors[::-1]),
        text=[money(v) for v in values[::-1]], textposition="outside",
        hovertemplate="<b>%{y}</b><br>$%{x:,.0f}<extra></extra>",
    ))
    lay = plotly_layout(height=max(220, 40 * len(labels_)))
    lay["margin"] = dict(l=16, r=90, t=8, b=16)
    lay["showlegend"] = False
    lay["xaxis_title"] = f"Spend over {org.window_days} days (USD)"
    fig.update_layout(**lay)
    st.plotly_chart(fig, use_container_width=True)

    unalloc = grouped.get("Unallocated", 0.0)
    if unalloc > 0 and org.total > 0:
        st.caption(
            f"{money(unalloc)} ({unalloc / org.total * 100:.1f}%) sits in "
            f"accounts with no owner tag. That number is a tagging-hygiene "
            f"metric, not a rounding error — it caps how far cost "
            f"accountability can be pushed."
        )


_render_ownership(org)
st.divider()


# ============================================================================
# MOVERS  /  SMALL-N FALLBACK
# ============================================================================
def _render_service_mix(acct: AccountSpend, key_prefix: str) -> None:
    if not acct.services:
        st.caption("No service breakdown available.")
        return
    items = sorted(acct.services.items(), key=lambda kv: kv[1],
                   reverse=True)[:6]
    fig = go.Figure(go.Bar(
        y=[k for k, _ in items][::-1],
        x=[v for _, v in items][::-1],
        orientation="h", marker=dict(color=C.BRAND),
        text=[money(v) for _, v in items][::-1], textposition="outside",
        hovertemplate="<b>%{y}</b><br>$%{x:,.0f}<extra></extra>",
    ))
    lay = plotly_layout(height=max(160, 34 * len(items)))
    lay["margin"] = dict(l=16, r=90, t=8, b=16)
    lay["showlegend"] = False
    fig.update_layout(**lay)
    st.plotly_chart(
        fig, use_container_width=True,
        key=f"{key_prefix}_{acct.account_id}",
    )


def _render_actions(
    acct: AccountSpend,
    on_ask: Optional[Callable[[str], None]],
    on_anomalies: Optional[Callable[[str], None]],
) -> None:
    question = (
        f"Why did spend in account {acct.account_id} ({acct.name}) change "
        f"by {acct.delta_abs:+,.0f} dollars last week, and what is "
        f"driving {acct.top_service}?"
    )
    ac1, ac2 = st.columns(2, gap="medium")
    with ac1:
        if st.button(
            "Ask CostSense about this account",
            key=f"orgv2_ask_{acct.account_id}",
            use_container_width=True,
        ):
            if on_ask:
                on_ask(question)
            else:
                st.info(
                    "Wire `on_ask` to route this to the Ask CostSense page."
                )
                st.code(question, language="text")
    with ac2:
        # Only enabled when we have a local SSO profile for this account —
        # otherwise the Anomalies page can't actually analyze it.
        has_profile = _profile_label_for_account(acct.account_id) is not None
        anom_help = (
            None if has_profile
            else f"Log in to account {acct.account_id} with "
                 f"`aws sso login` to enable this."
        )
        if st.button(
            "View anomalies", key=f"orgv2_anom_{acct.account_id}",
            use_container_width=True, disabled=not has_profile,
            help=anom_help,
        ):
            if on_anomalies:
                on_anomalies(acct.account_id)
            else:
                st.info(
                    f"Wire `on_anomalies` to open Anomalies filtered to "
                    f"{acct.account_id}."
                )


def _render_mover(
    acct: AccountSpend, org: OrgSpend,
    on_ask: Optional[Callable], on_anomalies: Optional[Callable],
) -> None:
    up = acct.delta_abs > 0
    colour = C.BAD if up else C.GOOD
    sign = "+" if up else "−"
    pct = (f"{acct.delta_pct:+.1f}%" if acct.delta_pct is not None
           else "no prior baseline")

    with st.container(border=True):
        left, right = st.columns([4, 1], gap="medium")
        with left:
            st.markdown(
                f"<span style='font-weight:650;'>{acct.name}</span>"
                f"<span style='color:{C.MUTED};'> · {acct.team} · "
                f"{acct.environment}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<span style='color:{C.MUTED};font-size:.85rem;'>"
                f"<code>{acct.account_id}</code> · top service "
                f"<b>{acct.top_service}</b> "
                f"({acct.top_service_share:.0f}% of this account)</span>",
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                f"<div style='text-align:right;'>"
                f"<div class='cs-num' style='color:{colour};"
                f"font-weight:680;font-size:1.15rem;'>"
                f"{sign}{money(abs(acct.delta_abs))}</div>"
                f"<div class='cs-num' style='color:{C.MUTED};"
                f"font-size:.82rem;'>{pct}</div></div>",
                unsafe_allow_html=True,
            )

        with st.expander("Service mix · actions"):
            _render_service_mix(acct, "mix_mover")
            if acct.open_anomalies:
                st.markdown(
                    f"<span style='background:{C.SEV_SOFT['High']};"
                    f"color:{C.SEV['High']};padding:2px 10px;"
                    f"border-radius:999px;font-size:.8rem;font-weight:600;'>"
                    f"{acct.open_anomalies} open "
                    f"anomal{'y' if acct.open_anomalies == 1 else 'ies'}"
                    f"</span>",
                    unsafe_allow_html=True,
                )
            _render_actions(acct, on_ask, on_anomalies)


def _render_small_n(
    org: OrgSpend,
    on_ask: Optional[Callable], on_anomalies: Optional[Callable],
) -> None:
    n = len(org.accounts_with_spend)
    st.markdown("##### Service breakdown")
    st.caption(
        f"Only {n} account{'s' if n != 1 else ''} in this organization has "
        f"spend, so cross-account movers aren't meaningful yet. Showing "
        f"what the money is going to instead."
    )
    for acct in org.accounts_with_spend:
        with st.container(border=True):
            st.markdown(
                f"**{acct.name}** · `{acct.account_id}` · {acct.team}"
            )
            _render_service_mix(acct, "mix_small")
            _render_actions(acct, on_ask, on_anomalies)


# Navigation hooks — the host app owns switching pages; the page only asks.
# When these aren't wired, the buttons degrade to showing the composed
# question / instructions instead of crashing.
def _on_ask(question: str) -> None:
    st.session_state["pending_question"] = question
    try:
        st.switch_page("pages/1_Ask_CostSense.py")
    except Exception:  # noqa: BLE001
        # Page name differs / not found — surface it so the user isn't stuck.
        st.info("Open Ask CostSense manually and paste this:")
        st.code(question, language="text")


def _profile_label_for_account(account_id: str) -> Optional[str]:
    """Find the Anomalies-page label ('<profile> (<account>)') for this
    account. Anomalies keys its selectbox by *label*, not raw profile.
    Returns None when we don't have local SSO into that account — in
    which case the button shouldn't be clickable at all."""
    for p in profiles:
        if p.account_id == account_id:
            return p.label
    return None


def _on_anomalies(account_id: str) -> None:
    """Deep-link into the Anomalies page:
      1. Force its account selectbox to this account_id.
      2. Set an auto-run flag so it kicks off Analyze on landing.
    Anomalies reads both keys on load; if either is missing it renders
    its normal UX."""
    target_label = _profile_label_for_account(account_id)
    if not target_label:
        # Extra safety — this branch shouldn't be reachable because the
        # button is disabled when no matching profile exists.
        st.warning(
            f"No local AWS profile has account {account_id}. Log into "
            f"it with `aws sso login` first, then reload."
        )
        return
    st.session_state["anom_profile"] = target_label
    st.session_state["anom_autorun"] = True
    # Bump the widget-version counter so the repos multiselect and any
    # other account-sensitive widgets re-instantiate cleanly.
    st.session_state["anom_widget_ver"] = (
        st.session_state.get("anom_widget_ver", 0) + 1
    )
    st.session_state["anom_last_profile"] = None  # force account-change path
    try:
        st.switch_page("pages/4_Anomalies.py")
    except Exception:  # noqa: BLE001
        st.info(
            f"Open Anomalies manually — pre-selected profile: {target_label}"
        )


if len(org.accounts_with_spend) < SMALL_N:
    _render_small_n(org, _on_ask, _on_anomalies)
else:
    st.markdown("##### Biggest movers")
    st.caption(
        f"Last 7 days vs prior 7 days, ranked by dollars. "
        f"Changes under {money(min_abs)} are suppressed."
    )
    movers = org.movers(min_abs)
    if not movers:
        callout(
            f"No account moved more than {money(min_abs)} week over week. "
            f"Org spend is stable.",
            tone="success",
        )
    else:
        for acct in movers:
            _render_mover(acct, org, _on_ask, _on_anomalies)

    suppressed = org.suppressed_movers(min_abs)
    if suppressed:
        st.caption(
            f"{suppressed} account(s) moved less than {money(min_abs)} — "
            f"hidden as noise. Lower the floor in Controls to see them."
        )
