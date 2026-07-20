"""Anomalies — full repo + full AWS sweep with ranked cost-cutting actions.

Flow:
  1. Sidebar: pick AWS profile + one or more GitHub repos.
  2. On "Scan": pre-fetches AWS-wide state (idle resources, rightsizing recs,
     top services, budgets) AND scans each repo (recent IaC changes, open
     PRs, scheduled rules).
  3. Feeds both sweeps to Bedrock Claude, which returns a ranked list of
     concrete actions: resource, action verb, $/day savings, confidence,
     category, rationale.
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

from src.ai_agent.anomaly_agent import analyze_anomalies
from src.ai_agent.aws_sweep import sweep_account
from src.ai_agent.aws_sweep import sweep_to_summary as aws_summary
from src.ai_agent.repo_sweep import sweep_repos
from src.ai_agent.repo_sweep import sweep_to_summary as repo_summary
from src.aws.profiles import resolve_all
from src.pr_scanner.repos import gh_login, gh_orgs, repos_with_user_prs


st.set_page_config(page_title="CostSense · Anomalies", layout="wide",
                   page_icon="🚨")

st.title("Anomalies & Recommendations")
st.caption("Full-repo + full-AWS sweep. Ranked list of concrete cost-cutting "
           "actions with $/day savings and confidence, grounded in real "
           "AWS + GitHub data.")


# ---------- sidebar ----------

with st.sidebar:
    st.header("AWS account")
    profiles = [p for p in resolve_all() if p.account_id]
    if not profiles:
        st.error("No AWS profiles reachable.")
        st.stop()
    labels = [p.label for p in profiles]
    pick = st.selectbox("Profile", labels)
    active = profiles[labels.index(pick)]

    st.divider()
    st.header("Repos to scan")
    try:
        orgs = list(gh_orgs())
        _gh_user = gh_login()
    except Exception:  # noqa: BLE001
        orgs = []
        _gh_user = "?"
    if orgs:
        gh_org = st.selectbox("GitHub org", orgs)
    else:
        gh_org = st.text_input("GitHub org")
    try:
        suggested = list(repos_with_user_prs(gh_org)) if gh_org else []
    except Exception:  # noqa: BLE001
        suggested = []
    selected_repos = st.multiselect(
        "Repos", options=suggested, default=suggested,
    )
    extra = st.text_input("Add repo (org/name)", placeholder="org/repo")
    if extra.strip() and extra.strip() not in selected_repos:
        selected_repos = selected_repos + [extra.strip()]

    st.divider()
    model_id = st.selectbox(
        "Bedrock model",
        index=1,
        options=[
            "anthropic.claude-3-haiku-20240307-v1:0",
            "us.anthropic.claude-sonnet-4-6",
        ],
    )

    st.divider()
    run_btn = st.button("Scan account + repos", type="primary")


# ---------- session cache ----------

report_key = f"anom::{active.profile}::{','.join(sorted(selected_repos))}"
report = st.session_state.get(report_key)

if run_btn:
    with st.spinner("Sweeping AWS (~30s) — Cost Explorer, Compute Optimizer, "
                    "resource inventory…"):
        try:
            aws_raw = sweep_account(active.profile)
            aws_sum = aws_summary(aws_raw)
        except Exception as e:  # noqa: BLE001
            st.error(f"AWS sweep failed: {e}")
            st.code(traceback.format_exc())
            st.stop()
    with st.spinner(f"Sweeping {len(selected_repos)} repo(s) via GitHub…"):
        try:
            repo_raw = sweep_repos(selected_repos) if selected_repos else []
            repo_sum = repo_summary(repo_raw)
        except Exception as e:  # noqa: BLE001
            st.error(f"Repo sweep failed: {e}")
            st.code(traceback.format_exc())
            st.stop()
    with st.spinner("Analyzing with Claude…"):
        try:
            report = analyze_anomalies(
                aws_summary=aws_sum, repo_summary=repo_sum,
                profile=active.profile, model_id=model_id,
            )
            st.session_state[report_key] = report
            st.session_state[report_key + "::aws"] = aws_sum
            st.session_state[report_key + "::repo"] = repo_sum
        except Exception as e:  # noqa: BLE001
            st.error(f"anomaly agent failed: {e}")
            st.code(traceback.format_exc())


# ---------- render ----------

if report is None:
    st.info("Pick an AWS profile and repos in the sidebar, then click "
            "**Scan account + repos**.")
    st.stop()

if report.error:
    st.error(f"Agent error: {report.error}")
    st.stop()

# Headline
st.subheader(report.summary or "Scan complete")
kpis = st.columns(3)
kpis[0].metric("Recommended actions", len(report.actions))
kpis[1].metric("Potential savings / day",
               f"${report.total_daily_savings_usd:,.2f}")
kpis[2].metric("AWS tool calls (drill-down)", report.tool_calls)

st.divider()

# Category filter
if report.actions:
    cats = sorted({a.category or "other" for a in report.actions})
    picked_cats = st.multiselect(
        "Filter by category", cats, default=cats,
    )
    filtered = [a for a in report.actions
                if (a.category or "other") in picked_cats]

    for i, a in enumerate(filtered, start=1):
        conf_color = {"high": "🟢", "medium": "🟡", "low": "⚪"}.get(
            a.confidence.lower(), "⚪"
        )
        with st.container(border=True):
            head = st.columns([3, 1, 1])
            head[0].markdown(
                f"#### {i}. `{a.category or 'other'}`"
            )
            head[1].metric("Save",
                           f"${a.est_daily_savings_usd:,.2f}/day",
                           label_visibility="visible")
            head[2].markdown(f"{conf_color} Confidence: **{a.confidence}**")
            st.markdown(
                f"- **Issue:** {a.issue or '_—_'}\n"
                f"- **Reason:** {a.reason or '_—_'}\n"
                f"- **Recommendation:** {a.recommendation or '_—_'}"
            )
            if a.source:
                st.caption(f"Source: `{a.source}`")

    st.divider()

    df = pd.DataFrame([
        {
            "issue": a.issue,
            "reason": a.reason,
            "recommendation": a.recommendation,
            "category": a.category,
            "est_daily_savings_usd": a.est_daily_savings_usd,
            "confidence": a.confidence,
            "source": a.source,
        } for a in filtered
    ])
    st.download_button(
        "Download as CSV", df.to_csv(index=False).encode("utf-8"),
        file_name="costsense_recommendations.csv", mime="text/csv",
    )

