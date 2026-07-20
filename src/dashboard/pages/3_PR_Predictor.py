"""PR Predictor — paste a GitHub PR URL, get predicted cost impact +
recommendations from the deep-AWS agent.
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

from src.ai_agent.agent import analyze_pr
from src.aws.profiles import resolve_all


st.set_page_config(page_title="CostSense · PR Predictor", layout="wide",
                   page_icon="🔮")

st.title("PR Predictor")
st.caption("Paste a GitHub PR URL. The agent reads the diff, queries the "
           "AWS account for real usage metrics, and predicts whether the PR "
           "will increase or decrease daily cost — plus recommendations for "
           "reducing it further.")


# ---------- sidebar ----------

with st.sidebar:
    st.header("AWS account")
    profiles = [p for p in resolve_all() if p.account_id]
    if not profiles:
        st.error("No AWS profiles reachable. `aws sso login` first.")
        st.stop()
    labels = [p.label for p in profiles]
    pick = st.selectbox("Profile", labels)
    active = profiles[labels.index(pick)]

    st.divider()
    st.subheader("Model")
    model_id = st.selectbox(
        "Bedrock model",
        index=1,   # default to Sonnet — Haiku often skips tools and returns neutral
        options=[
            "anthropic.claude-3-haiku-20240307-v1:0",
            "us.anthropic.claude-sonnet-4-6",
        ],
        help="Haiku is fast (~$0.001/PR). Sonnet is more accurate on complex "
             "diffs and better at tool use.",
    )


# ---------- main ----------

pr_url = st.text_input(
    "GitHub PR URL",
    placeholder="https://github.com/org/repo/pull/123",
)
run = st.button("Predict cost impact", type="primary",
                disabled=not pr_url.strip())

if run and pr_url.strip():
    with st.spinner("Fetching diff + querying AWS…"):
        try:
            verdict = analyze_pr(
                pr_url.strip(), profile=active.profile, model_id=model_id,
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"agent failed: {e}")
            st.code(traceback.format_exc())
            st.stop()

    if verdict.error:
        st.error(f"Agent error: {verdict.error}")
        st.stop()

    # verdict banner
    if verdict.direction == "increase":
        st.error(f"### ↗ {verdict.verdict}")
    elif verdict.direction == "decrease":
        st.success(f"### ↘ {verdict.verdict}")
    else:
        st.info(f"### → {verdict.verdict}")

    cols = st.columns(3)
    cols[0].metric("Est. daily impact",
                   f"${verdict.est_daily_delta_usd:+,.2f}")
    cols[1].metric("Est. monthly impact",
                   f"${verdict.est_daily_delta_usd * 30:+,.0f}")
    cols[2].metric("AWS tool calls", verdict.tool_calls)

    if verdict.detail:
        st.write(verdict.detail)

    st.divider()

    if verdict.findings:
        st.subheader("What this PR does to cost")
        st.dataframe(
            pd.DataFrame([{
                "Resource": f.resource,
                "Action": f.action,
                "$/day Δ": f.est_daily_delta_usd,
                "Confidence": f.confidence,
                "Rationale": f.rationale,
            } for f in verdict.findings]),
            use_container_width=True, hide_index=True,
        )

    if verdict.recommendations:
        st.subheader("Recommendations to reduce cost further")
        for i, r in enumerate(verdict.recommendations, start=1):
            with st.container(border=True):
                cc = st.columns([3, 1, 1])
                cc[0].markdown(f"**{i}. {r.resource}** — _{r.action}_")
                cc[1].metric("If applied", f"${r.est_daily_delta_usd:+,.2f}/d",
                             label_visibility="visible")
                cc[2].markdown(f"Confidence: **{r.confidence}**")
                st.markdown(r.rationale)
