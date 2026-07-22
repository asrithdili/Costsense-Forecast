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

from datetime import date, timedelta

from src.ai_agent.agent import analyze_pr, narrate_pr_impact
from src.aws.cost_explorer import fetch_daily_totals
from src.aws.profiles import resolve_all
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


st.title("PR Predictor")
st.caption("Paste a GitHub PR URL. The agent reads the diff, queries the "
           "AWS account for real usage metrics, and predicts whether the PR "
           "will increase or decrease daily cost — plus recommendations for "
           "reducing it further.")


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
    c1, c2 = st.columns([3, 3])
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

# Persist the last predicted URL + verdict across tab switches so users
# don't have to re-predict when they come back. Keyed on (profile, url).
pr_url = st.text_input(
    "GitHub PR URL",
    placeholder="https://github.com/org/repo/pull/123",
    key="prp_last_url",
)
run = st.button("Predict cost impact", type="primary",
                disabled=not pr_url.strip())

verdict_key = f"prp_verdict::{active.profile}::{pr_url.strip()}"
verdict = st.session_state.get(verdict_key)

narrative_key = f"prp_narrative::{active.profile}::{pr_url.strip()}"
current_daily_key = f"prp_current_daily::{active.profile}"
narrative = st.session_state.get(narrative_key)
current_daily = st.session_state.get(current_daily_key)

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

    # Pull current account $/day (avg last 7 days from Cost Explorer)
    with st.spinner("Fetching current account spend…"):
        try:
            today = date.today()
            totals = fetch_daily_totals(
                today - timedelta(days=7), today, profile=active.profile,
            )
            current_daily = (sum(a for _, a in totals) / len(totals)
                             if totals else 0.0)
        except Exception as e:  # noqa: BLE001
            st.warning(f"Couldn't fetch current daily cost: {e}")
            current_daily = 0.0
    st.session_state[current_daily_key] = current_daily

    with st.spinner("Writing plain-English summary…"):
        try:
            narrative = narrate_pr_impact(
                current_daily_usd=current_daily,
                verdict=verdict,
                profile=active.profile,
                model_id=model_id,
            )
        except Exception as e:  # noqa: BLE001
            narrative = f"(narrative unavailable: {e})"
    st.session_state[narrative_key] = narrative

if verdict is not None:
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

    projected_daily = (float(current_daily or 0.0)
                       + verdict.est_daily_delta_usd)

    # Estimate metadata — shown as a badge/caption so it's clear whether
    # the number came from measured metrics or a fallback estimate.
    _basis = (verdict.estimation_basis or "measured").lower()
    _measured = bool(verdict.measured)
    _conf = (verdict.confidence or "medium").lower()
    _lo, _hi = verdict.est_daily_delta_low_usd, verdict.est_daily_delta_high_usd
    _has_range = (
        _lo is not None and _hi is not None
        and abs(_hi - _lo) > 0.01
    )
    if _basis == "measured" and _measured:
        _basis_label = "🟢 Measured (CloudWatch / Cost Explorer)"
    elif _basis == "sibling_account":
        _basis_label = ("🟡 Estimated from peer AWS account "
                        "(historical precedent)")
    elif _basis == "unknown":
        _basis_label = "⚪ Unquantifiable — no reachable AWS grounding"
    else:
        _basis_label = "⚪ Basis unknown"
    _conf_label = {"high": "🟢 High", "medium": "🟡 Medium",
                   "low": "🟠 Low"}.get(_conf, "⚪")

    cols = st.columns(5)
    cols[0].metric(
        "Current account $/day",
        f"${current_daily:,.2f}" if current_daily is not None else "—",
        help="7-day average from Cost Explorer for the selected account.",
    )
    _proj_delta_label = (
        f"{_lo:+,.2f} → {_hi:+,.2f}"
        if _has_range else f"{verdict.est_daily_delta_usd:+,.2f}"
    )
    cols[1].metric(
        "Projected $/day after merge",
        f"${projected_daily:,.2f}",
        delta=_proj_delta_label,
    )
    _daily_impact_label = (
        f"${_lo:+,.2f} to ${_hi:+,.2f}"
        if _has_range else f"${verdict.est_daily_delta_usd:+,.2f}"
    )
    cols[2].metric("Est. daily impact", _daily_impact_label)
    _monthly_low = (_lo * 30) if _has_range else verdict.est_daily_delta_usd * 30
    _monthly_high = (_hi * 30) if _has_range else verdict.est_daily_delta_usd * 30
    _monthly_label = (
        f"${_monthly_low:+,.0f} to ${_monthly_high:+,.0f}"
        if _has_range else f"${verdict.est_daily_delta_usd * 30:+,.0f}"
    )
    cols[3].metric("Est. monthly impact", _monthly_label)
    cols[4].metric("AWS tool calls", verdict.tool_calls)

    badge_cols = st.columns([3, 2])
    badge_cols[0].caption(f"**Basis:** {_basis_label}")
    badge_cols[1].caption(f"**Confidence:** {_conf_label}")
    if not _measured or _basis != "measured":
        if _basis == "sibling_account":
            st.caption(
                "⚠ This number is an estimate, not a measured cost. It "
                "comes from historical precedent: a prior scope-expansion "
                "PR in this repo, and the step change it caused in a "
                "sibling AWS account around its merge date."
            )
        else:
            st.caption(
                "⚠ No reachable AWS account runs the service this PR "
                "affects, and no historical precedent was found in this "
                "repo. The true cost delta is greater than $0/day but "
                "cannot be quantified from available data."
            )

    if narrative:
        st.markdown("**Cost summary:**")
        st.markdown(_md(narrative))

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
