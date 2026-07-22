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
from src.dashboard.costsense_theme import metric, pill, section
from src.dashboard.nav import (
    inject_css, render_sidebar_footer, render_sidebar_header, top_bar,
)


st.set_page_config(page_title="CostSense · PR Predictor", layout="wide",
                   page_icon="🔮")
inject_css()
render_sidebar_header()  # Diligent card renders before any AWS calls

MODEL_OPTIONS = [
    ("us.anthropic.claude-sonnet-4-6",         "Claude Sonnet 4.6"),
    ("anthropic.claude-3-haiku-20240307-v1:0", "Claude 3 Haiku"),
]

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


_VERDICT_PILL = {
    "increase": "High",
    "decrease": "Low",
    "neutral": "Medium",
}


section(
    "PR Predictor",
    "Paste a GitHub PR URL. The agent reads the diff, queries the "
    "AWS account for real usage metrics, and predicts whether the PR "
    "will increase or decrease daily cost — plus recommendations for "
    "reducing it further.",
    kicker="Pull request",
)


# ---------- top control bar ----------

with st.spinner("Resolving profiles…"):
    profiles = [p for p in resolve_all() if p.account_id]
if not profiles:
    st.error("No AWS profiles reachable. Run `aws sso login` or launch via `aws-vault exec <profile> --` first.")
    st.stop()
labels = [p.label for p in profiles]
model_ids = [mid for mid, _ in MODEL_OPTIONS]
model_labels = [name for _, name in MODEL_OPTIONS]

picked_label = st.session_state.get("prp_profile", labels[0])
if picked_label not in labels:
    picked_label = labels[0]
picked_model_idx = st.session_state.get("prp_model_idx", 0)
if not (0 <= picked_model_idx < len(model_ids)):
    picked_model_idx = 0

header = (f"Controls  ·  Account: {picked_label}  ·  "
          f"Model: {model_labels[picked_model_idx]}")
with top_bar(header):
    c1, c2 = st.columns([3, 3], gap="medium", vertical_alignment="bottom")
    with c1:
        picked_label = st.selectbox(
            "Account", labels,
            index=labels.index(picked_label),
            key="prp_profile",
        )
    with c2:
        picked_model_idx = st.selectbox(
            "Model", range(len(model_labels)),
            index=picked_model_idx,
            format_func=lambda i: model_labels[i],
            key="prp_model_idx",
            help="Haiku is fast. Sonnet is more accurate on complex diffs.",
        )

active = profiles[labels.index(picked_label)]
model_id = model_ids[picked_model_idx]

with st.sidebar:
    render_sidebar_footer(
        active_profile=active.profile,
        account_id=active.account_id,
        extra_rows=[("Model", model_labels[picked_model_idx])],
    )


# ---------- main ----------

section(
    "GitHub PR URL",
    "The agent fetches the diff and queries live AWS metrics for the "
    "selected account.",
    kicker="Input",
)

# Persist the last predicted URL + verdict across tab switches so users
# don't have to re-predict when they come back. Keyed on (profile, url).
url_col, btn_col = st.columns([4, 1], gap="medium", vertical_alignment="bottom")
with url_col:
    pr_url = st.text_input(
        "GitHub PR URL",
        placeholder="https://github.com/org/repo/pull/123",
        key="prp_last_url",
        label_visibility="collapsed",
    )
with btn_col:
    run = st.button(
        "Predict cost impact", type="primary",
        disabled=not pr_url.strip(),
        use_container_width=True,
    )

verdict_key = f"prp_verdict::{active.profile}::{pr_url.strip()}"
verdict = st.session_state.get(verdict_key)

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
    st.session_state[verdict_key] = verdict

if verdict is not None:
    if verdict.error:
        st.error(f"Agent error: {verdict.error}")
        st.stop()

    st.divider()

    # verdict banner
    pill_level = _VERDICT_PILL.get(verdict.direction, "Medium")
    with st.container(border=True):
        st.markdown(pill(pill_level), unsafe_allow_html=True)
        st.markdown(f"### {_md(verdict.verdict)}")

    daily = verdict.est_daily_delta_usd
    monthly = daily * 30
    impact_good = daily < 0
    impact_bad = daily > 0
    cols = st.columns(3, gap="medium")
    with cols[0]:
        metric(
            "Est. daily impact",
            f"${daily:+,.2f}",
            delta="cost increase" if impact_bad else "savings" if impact_good else None,
            good=False if impact_bad else True if impact_good else None,
        )
    with cols[1]:
        metric(
            "Est. monthly impact",
            f"${monthly:+,.0f}",
            delta="projected" if daily != 0 else None,
            good=False if impact_bad else True if impact_good else None,
        )
    with cols[2]:
        metric("AWS tool calls", verdict.tool_calls)

    if verdict.detail:
        st.markdown(f"**In plain terms:** {_md(verdict.detail)}")

    st.divider()

    if verdict.findings:
        section(
            "What this PR does to cost",
            "Resource-level breakdown of estimated daily cost changes.",
            kicker="Findings",
        )
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
        section(
            "Recommendations to reduce cost further",
            "Ranked actions with trade-offs and optional code changes.",
            kicker="Actions",
        )
        for i, r in enumerate(verdict.recommendations, start=1):
            with st.container(border=True):
                head_l, head_r = st.columns([3, 1], gap="medium")
                with head_l:
                    st.markdown(f"**{i}. {_md(r.resource)}** — _{r.action}_")
                with head_r:
                    rec_delta = r.est_daily_delta_usd
                    rec_good = rec_delta < 0
                    rec_bad = rec_delta > 0
                    metric(
                        "If applied",
                        f"${rec_delta:+,.2f}/d",
                        delta="savings" if rec_good else "cost" if rec_bad else None,
                        good=True if rec_good else False if rec_bad else None,
                    )
                st.markdown(_md(r.rationale))

                if r.pros or r.cons:
                    pc = st.columns(2, gap="medium")
                    with pc[0]:
                        if r.pros:
                            st.markdown("**Pros**\n" + "\n".join(
                                f"- {_md(p)}" for p in r.pros))
                    with pc[1]:
                        if r.cons:
                            st.markdown("**Cons**\n" + "\n".join(
                                f"- {_md(c)}" for c in r.cons))

                if r.current_code or r.recommended_code:
                    with st.expander("View suggested code change"):
                        code_cols = st.columns(2, gap="medium")
                        with code_cols[0]:
                            st.caption("Current")
                            st.code(r.current_code or "(not identified in diff)",
                                    language="text")
                        with code_cols[1]:
                            st.caption("Recommended")
                            st.code(
                                r.recommended_code
                                or "(no code change — see rationale above)",
                                language="text",
                            )
