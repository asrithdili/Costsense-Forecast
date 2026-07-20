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


st.set_page_config(page_title="CostSense · Org Impact", layout="wide",
                   page_icon="🏢")

st.title("Org-Level Impact")
st.caption("Per-account spend across every linked account in the AWS "
           "Organization. Data comes from Cost Explorer's `LINKED_ACCOUNT` "
           "dimension on the management/payer profile.")


# ---------- sidebar ----------

with st.sidebar:
    st.header("Management profile")
    profiles = [p for p in resolve_all() if p.account_id]
    if not profiles:
        st.error("No AWS profiles reachable.")
        st.stop()
    # Prefer control-tower-style names if present
    default_idx = 0
    for i, p in enumerate(profiles):
        if "control-tower" in p.profile or "tower" in p.profile or "master" in p.profile:
            default_idx = i
            break
    labels = [p.label for p in profiles]
    pick = st.selectbox("Profile", labels, index=default_idx,
                        help="This should be the AWS Organizations management "
                             "(payer) account — usually control-tower.")
    active = profiles[labels.index(pick)]

    st.divider()
    days = st.slider("History window (days)", 7, 90, 30, step=7)
    include_top_service = st.checkbox(
        "Fetch top service per account", value=False,
        help="One extra Cost Explorer call per linked account. Slower but "
             "more useful when spotting which account is driving cost.",
    )
    run = st.button("Fetch org spend", type="primary")


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
            with st.spinner("Fetching top service per account…"):
                for acct in data[:30]:
                    try:
                        acct.__dict__["top_service"] = top_service_by_account(
                            active.profile, acct.account_id, days=days,
                        ) or "—"
                    except Exception:  # noqa: BLE001
                        acct.__dict__["top_service"] = "—"
        st.session_state[cache_key] = data


if data is None:
    st.info("Pick a profile in the sidebar (ideally the management/payer "
            "account) and click **Fetch org spend**.")
    st.stop()

if not data:
    st.warning("No linked-account spend data found. Is this profile the "
               "AWS Organizations management account?")
    st.stop()

# KPI row
total = sum(a.total_usd for a in data)
active_count = sum(1 for a in data if a.total_usd > 0)
kpi = st.columns(3)
kpi[0].metric("Linked accounts", len(data))
kpi[1].metric("Accounts with spend", active_count)
kpi[2].metric(f"Total org spend ({days}d)", f"${total:,.0f}")

st.divider()

# Per-account table
rows = []
for a in data:
    row = {
        "Account ID": a.account_id,
        "Account name": a.account_name,
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

# CSV export
st.download_button(
    "Download as CSV", df.to_csv(index=False).encode("utf-8"),
    file_name="org_level_spend.csv", mime="text/csv",
)

st.divider()

# Top movers callout
if any(a.trend_pct() is not None for a in data):
    st.subheader("Biggest movers — last 7 days vs prior 7 days")
    with_trend = [a for a in data if a.trend_pct() is not None]
    biggest_up = sorted(with_trend, key=lambda a: -(a.trend_pct() or 0))[:3]
    biggest_down = sorted(with_trend, key=lambda a: (a.trend_pct() or 0))[:3]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**↗ Trending up**")
        for a in biggest_up:
            if (a.trend_pct() or 0) <= 0:
                continue
            st.markdown(
                f"- **{a.account_name}** — trend `{a.trend_pct():+.1f}%`, "
                f"last 7d ${a.spend_last_n_days(7):,.0f}"
            )
    with c2:
        st.markdown("**↘ Trending down**")
        for a in biggest_down:
            if (a.trend_pct() or 0) >= 0:
                continue
            st.markdown(
                f"- **{a.account_name}** — trend `{a.trend_pct():+.1f}%`, "
                f"last 7d ${a.spend_last_n_days(7):,.0f}"
            )
