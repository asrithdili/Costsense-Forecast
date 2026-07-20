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

# Streamlit's markdown renderer treats a "$...$" pair as inline LaTeX math,
# which is exactly what raw "$57.60/day" style text looks like — it mangles
# the font (serif/italic, wrong size) and garbles spacing. `_md` escapes the
# "$" before anything AI-generated goes through st.markdown/st.write so all
# body text renders in one consistent font. The CSS below is a second,
# belt-and-suspenders normalizer for font-size across every text element.
st.markdown("""
<style>
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    font-size: 1rem !important;
    line-height: 1.6 !important;
    font-style: normal !important;
}
</style>
""", unsafe_allow_html=True)


def _md(text: str | None) -> str:
    """Escape '$' so it never gets parsed as LaTeX math."""
    return (text or "").replace("$", "\\$")


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
        st.error(f"### ↗ {_md(verdict.verdict)}")
    elif verdict.direction == "decrease":
        st.success(f"### ↘ {_md(verdict.verdict)}")
    else:
        st.info(f"### → {_md(verdict.verdict)}")

    cols = st.columns(3)
    cols[0].metric("Est. daily impact",
                   f"${verdict.est_daily_delta_usd:+,.2f}")
    cols[1].metric("Est. monthly impact",
                   f"${verdict.est_daily_delta_usd * 30:+,.0f}")
    cols[2].metric("AWS tool calls", verdict.tool_calls)

    if verdict.detail:
        st.markdown(f"**In plain terms:** {_md(verdict.detail)}")

    st.divider()

    if verdict.findings:
        st.subheader("What this PR does to cost")
        st.dataframe(
            pd.DataFrame([{
                "Resource": f.resource,
                "Action": f.action,
                "$/day Δ": f.est_daily_delta_usd,
                "Rationale": f.rationale,
            } for f in verdict.findings]),
            use_container_width=True, hide_index=True,
        )

    if verdict.recommendations:
        st.subheader("Recommendations to reduce cost further")
        for i, r in enumerate(verdict.recommendations, start=1):
            with st.container(border=True):
                cc = st.columns([3, 1])
                cc[0].markdown(f"**{i}. {_md(r.resource)}** — _{r.action}_")
                cc[1].metric("If applied", f"${r.est_daily_delta_usd:+,.2f}/d",
                             label_visibility="visible")
                st.markdown(_md(r.rationale))

                if r.pros or r.cons:
                    pc = st.columns(2)
                    with pc[0]:
                        if r.pros:
                            st.markdown("**✅ Pros**\n" + "\n".join(
                                f"- {_md(p)}" for p in r.pros))
                    with pc[1]:
                        if r.cons:
                            st.markdown("**⚠️ Cons**\n" + "\n".join(
                                f"- {_md(c)}" for c in r.cons))

                if r.current_code or r.recommended_code:
                    with st.expander("📝 View suggested code change"):
                        code_cols = st.columns(2)
                        with code_cols[0]:
                            st.caption("Current")
                            st.code(r.current_code or "(not identified in diff)",
                                    language="text")
                        with code_cols[1]:
                            st.caption("Recommended")
                            st.code(r.recommended_code or "(no code change — see rationale above)",
                                    language="text")
