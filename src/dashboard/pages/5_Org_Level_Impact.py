"""Org-Level Impact — per-account 30-day spend rollup pulled from the AWS
Organizations management (control-tower) account via Cost Explorer's
LINKED_ACCOUNT dimension.

The management profile can see every linked account's spend even without
direct SSO into those accounts. Names come from Organizations if allowed;
otherwise the account id is shown.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st

from src.aws.org_spend import fetch_org_spend, top_service_by_account
from src.aws.profiles import resolve_all
from src.dashboard.costsense_theme import metric, money, section
from src.dashboard.nav import (
    inject_css, render_sidebar_footer, render_sidebar_header, top_bar,
)


st.set_page_config(page_title="CostSense · Org Impact", layout="wide",
                   page_icon="🏢")
inject_css()
render_sidebar_header()  # Diligent card renders before any AWS calls

section(
    "Org-Level Impact",
    "Per-account spend across every linked account in the AWS "
    "Organization. Data comes from Cost Explorer's `LINKED_ACCOUNT` "
    "dimension on the management/payer profile.",
    kicker="Organization",
)


# ---------- top control bar ----------

with st.spinner("Resolving profiles…"):
    profiles = [p for p in resolve_all() if p.account_id]
if not profiles:
    st.error("No AWS profiles reachable.")
    st.stop()
default_idx = 0
for i, p in enumerate(profiles):
    if "control-tower" in p.profile or "tower" in p.profile or "master" in p.profile:
        default_idx = i
        break
labels = [p.label for p in profiles]

# Header shows current picks; expander body has the widgets.
picked_label = st.session_state.get("orglvl_profile", labels[default_idx])
if picked_label not in labels:
    picked_label = labels[default_idx]
picked_days = st.session_state.get("orglvl_days", 30)
header = (f"Controls  ·  Account: {picked_label}  ·  "
          f"History: {picked_days}d")
with top_bar(header):
    c1, c2, c3, c4 = st.columns(
        [3, 2, 2, 2], gap="medium", vertical_alignment="bottom",
    )
    with c1:
        picked_label = st.selectbox(
            "Management profile", labels,
            index=labels.index(picked_label),
            key="orglvl_profile",
            help="AWS Organizations management (payer) account — "
                 "usually control-tower.",
        )
    with c2:
        picked_days = st.slider(
            "History (days)", 7, 90, picked_days, step=7,
            key="orglvl_days",
        )
    with c3:
        include_top_service = st.checkbox(
            "Fetch top service", value=False,
            help="One extra CE call per linked account. Slower but useful.",
            key="orglvl_top_service",
        )
    with c4:
        run = st.button("Fetch org spend", type="primary",
                        use_container_width=True)

active = profiles[labels.index(picked_label)]
days = picked_days

with st.sidebar:
    render_sidebar_footer(
        active_profile=active.profile,
        account_id=active.account_id,
        extra_rows=[
            ("Window", f"{days}d"),
            ("Top svc", "on" if include_top_service else "off"),
        ],
    )


# ---------- main ----------

cache_key = f"orgspend::{active.profile}::{days}::{include_top_service}"
data = st.session_state.get(cache_key)

if run:
    with st.spinner(f"Querying Cost Explorer via `{active.profile}` for the "
                    f"last {days} days across all linked accounts…"):
        try:
            data = fetch_org_spend(active.profile, days=days)
        except Exception as e:  # noqa: BLE001
            st.error(f"Cost Explorer failed: {e}")
            st.code(traceback.format_exc())
            st.stop()
        if include_top_service and data:
            # Only fetch top service for the top 30 by spend — the default
            # view shows those, and each extra account adds a Cost Explorer
            # call. If the user searches for a specific account outside the
            # top 30, top service will just show "—" for it; they can
            # follow up in the CostSense AI chat.
            from concurrent.futures import ThreadPoolExecutor
            targets = [a for a in data if a.total_usd > 0][:30]
            with st.spinner(f"Fetching top service for {len(targets)} "
                            "accounts (parallel)…"):
                def _fetch(acct):
                    try:
                        return acct.account_id, (top_service_by_account(
                            active.profile, acct.account_id, days=days,
                        ) or "—")
                    except Exception:  # noqa: BLE001
                        return acct.account_id, "—"
                with ThreadPoolExecutor(max_workers=10) as pool:
                    results = dict(pool.map(_fetch, targets))
            for a in data:
                a.__dict__["top_service"] = results.get(a.account_id, "—")
        st.session_state[cache_key] = data


if data is None:
    st.info("Open **Controls** above, pick the management/payer profile, "
            "and click **Fetch org spend**.")
    st.stop()

if not data:
    st.warning("No linked-account spend data found. Is this profile the "
               "AWS Organizations management account?")
    st.stop()

# KPI row
total = sum(a.total_usd for a in data)
active_count = sum(1 for a in data if a.total_usd > 0)
kpi = st.columns(3, gap="medium")
with kpi[0]:
    metric("Linked accounts", len(data))
with kpi[1]:
    metric("Accounts with spend", active_count)
with kpi[2]:
    spend_display = money(total) if total >= 1_000 else f"${total:,.0f}"
    metric(f"Total org spend ({days}d)", spend_display)

st.divider()

# Search / filter — default view shows top 30 by spend. Search jumps to any
# other account by ID substring, so users can look up accounts outside the
# top 30 without loading all 676+ rows.
TOP_N = 30
section(
    "Account spend",
    "Browse the top accounts by spend or search by account ID to jump "
    "outside the default view.",
    kicker="Results",
)
search = st.text_input(
    "Search account ID",
    placeholder="Enter part of an account id (e.g. 972575) to look up "
                "accounts outside the top 30",
    label_visibility="collapsed",
)

if search.strip():
    q = search.strip()
    filtered = [a for a in data if q in a.account_id]
    if not filtered:
        st.warning(f"No account id contains `{q}`. "
                   f"Showing top {TOP_N} by spend instead.")
        filtered = data[:TOP_N]
    else:
        st.caption(f"{len(filtered)} match(es) for `{q}`.")
else:
    filtered = data[:TOP_N]
    if len(data) > TOP_N:
        st.caption(f"Showing top **{TOP_N}** of {len(data)} accounts by "
                   f"spend. Use the search box above to look up a specific "
                   "account, or download the full CSV.")

rows = []
for a in filtered:
    row = {
        "Account ID": a.account_id,
        f"{days}d total ($)": round(a.total_usd, 2),
        "Last 7d ($)": a.spend_last_n_days(7),
        "Prior 7d ($)": (
            round(sum(v for _, v in a.daily[-14:-7]), 2)
            if len(a.daily) >= 14 else None
        ),
        "Trend (%)": a.trend_pct(),
    }
    if include_top_service:
        row["Top service"] = a.__dict__.get("top_service", "—")
    rows.append(row)

df = pd.DataFrame(rows)
st.dataframe(
    df,
    use_container_width=True, hide_index=True,
    column_config={
        f"{days}d total ($)": st.column_config.NumberColumn(format="$%.2f"),
        "Last 7d ($)": st.column_config.NumberColumn(format="$%.2f"),
        "Prior 7d ($)": st.column_config.NumberColumn(format="$%.2f"),
        "Trend (%)": st.column_config.NumberColumn(format="%+.1f%%"),
    },
)

st.divider()

# Top movers callout
if any(a.trend_pct() is not None for a in data):
    section(
        "Biggest movers",
        "Last 7 days vs prior 7 days — largest spend shifts across the org.",
        kicker="Trends",
    )
    with_trend = [a for a in data if a.trend_pct() is not None]
    biggest_up = sorted(with_trend, key=lambda a: -(a.trend_pct() or 0))[:3]
    biggest_down = sorted(with_trend, key=lambda a: (a.trend_pct() or 0))[:3]
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("**↗ Trending up**")
        for a in biggest_up:
            if (a.trend_pct() or 0) <= 0:
                continue
            last7 = a.spend_last_n_days(7)
            last7_txt = money(last7) if last7 >= 1_000 else f"${last7:,.0f}"
            st.markdown(
                f"- **{a.account_id}** — trend `{a.trend_pct():+.1f}%`, "
                f"last 7d {last7_txt}"
            )
    with c2:
        st.markdown("**↘ Trending down**")
        for a in biggest_down:
            if (a.trend_pct() or 0) >= 0:
                continue
            last7 = a.spend_last_n_days(7)
            last7_txt = money(last7) if last7 >= 1_000 else f"${last7:,.0f}"
            st.markdown(
                f"- **{a.account_id}** — trend `{a.trend_pct():+.1f}%`, "
                f"last 7d {last7_txt}"
            )
