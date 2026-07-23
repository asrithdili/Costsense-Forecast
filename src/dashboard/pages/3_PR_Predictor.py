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
from src.dashboard.costsense_theme import (
    callout, meta_row, metric, pill, section,
)
from src.dashboard.notifications_ui import NotificationDraft, render_notification_button
from src.dashboard.state_cache import cached_state
from src.dashboard.nav import (
    inject_css, render_sidebar_footer, render_sidebar_header, top_bar,
)
from src.forecast.adapters import event_from_pr_predictor, queue_pending_event


st.set_page_config(page_title="CostSense · PR Predictor", layout="wide")
inject_css()
render_sidebar_header()  # Diligent card renders before any AWS calls

MODEL_OPTIONS = [
    ("us.anthropic.claude-sonnet-4-6",         "Claude Sonnet 4.6"),
    ("anthropic.claude-3-haiku-20240307-v1:0", "Claude 3 Haiku"),
]

# Streamlit's markdown parser treats several characters as inline syntax
# that mangle plain LLM prose:
#   - "$…$" pair  → inline LaTeX (breaks font on prices like "$57.60/day")
#   - "~text~"    → strikethrough (crossed-out numbers like ~14,400 inv/day)
#   - "*text*"    → italic; "_text_" also italic
# `_md` escapes them before anything AI-generated hits st.markdown so all
# body text renders as plain, un-styled prose. The shared design system
# (costsense_theme.inject_css above) handles page-level typography.
def _md(text: str | None) -> str:
    """Escape markdown syntax that mangles plain prose from the LLM:
    "$" (LaTeX), "~" (strikethrough), "*" and "_" (emphasis)."""
    if not text:
        return ""
    return (text
            .replace("\\", "\\\\")
            .replace("$", "\\$")
            .replace("~", "\\~")
            .replace("*", "\\*")
            .replace("_", "\\_"))


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
    callout(
        "No AWS profiles reachable. Run `aws sso login` or launch via "
        "`aws-vault exec <profile> --` first.",
        tone="error",
    )
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

# Restore the last-predicted URL for this profile from disk BEFORE the
# text_input widget renders. Streamlit's session_state resets on new
# browser tabs / F5 reloads / server restarts; without this seeding
# step the URL field renders blank and the (profile, url) cache lookup
# below misses even though the verdict pickle IS on disk.
_last_url_for_profile = cached_state.get("prp_last_url", (active.profile,))
if _last_url_for_profile and "prp_last_url" not in st.session_state:
    st.session_state["prp_last_url"] = _last_url_for_profile

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

# Disk-backed cache. Restores the verdict / narrative / current_daily
# from the previous session so tab switches, browser reloads, or a
# Streamlit server restart don't wipe what the user just spent 30
# seconds computing. `cached_state` writes to both session_state (fast
# hot cache) AND a pickle under data/ui_state/ (survives everything).
_pr_url_key = pr_url.strip()
verdict = cached_state.get("prp_verdict", (active.profile, _pr_url_key))
narrative = cached_state.get("prp_narrative", (active.profile, _pr_url_key))
current_daily = cached_state.get("prp_current_daily", (active.profile,))

if run and pr_url.strip():
    with st.spinner("Fetching diff + querying AWS…"):
        try:
            verdict = analyze_pr(
                pr_url.strip(), profile=active.profile, model_id=model_id,
            )
        except Exception as e:  # noqa: BLE001
            callout(f"agent failed: {e}", tone="error")
            st.code(traceback.format_exc())
            st.stop()
    cached_state.set("prp_verdict", (active.profile, _pr_url_key), verdict)
    # Remember which URL this profile last predicted, so the next
    # session can re-seed the text box + re-lookup the verdict.
    cached_state.set("prp_last_url", (active.profile,), _pr_url_key)

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
    cached_state.set("prp_current_daily", (active.profile,), current_daily)

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
    cached_state.set("prp_narrative", (active.profile, _pr_url_key), narrative)

if verdict is not None:
    if verdict.error:
        callout(f"Agent error: {verdict.error}", tone="error")
        st.stop()

    st.divider()

    # ---- Verdict banner (theme-styled) --------------------------------
    pill_level = _VERDICT_PILL.get(verdict.direction, "Medium")
    with st.container(border=True):
        st.markdown(pill(pill_level), unsafe_allow_html=True)
        st.markdown(f"### {_md(verdict.verdict)}")

    # ---- Cost tiles: current, projected, daily/monthly impact ---------
    # The precedent-lookup tool returns a range (low, high) when the number
    # is an estimate. Collapse to the signed midpoint for tile headlines so
    # the projected total lines up with the daily/monthly impact numbers.
    _lo = verdict.est_daily_delta_low_usd
    _hi = verdict.est_daily_delta_high_usd
    _has_range = (
        _lo is not None and _hi is not None and abs(_hi - _lo) > 0.01
    )
    _delta_headline = (
        (_lo + _hi) / 2.0 if _has_range else verdict.est_daily_delta_usd
    )
    _projected_headline = float(current_daily or 0.0) + _delta_headline
    _monthly = _delta_headline * 30
    _daily_good = _delta_headline < 0
    _daily_bad = _delta_headline > 0

    cols = st.columns(5, gap="medium")
    with cols[0]:
        metric(
            "Current account $/day",
            f"${current_daily:,.2f}" if current_daily is not None else "—",
        )
    with cols[1]:
        metric(
            "Projected $/day after merge",
            f"${_projected_headline:,.2f}",
            delta=f"{_delta_headline:+,.2f} vs today",
            good=True if _daily_good else False if _daily_bad else None,
        )
    with cols[2]:
        metric(
            "Est. daily impact",
            f"${_delta_headline:+,.2f}",
            delta="cost increase" if _daily_bad else
                  "savings" if _daily_good else None,
            good=True if _daily_good else False if _daily_bad else None,
        )
    with cols[3]:
        metric(
            "Est. monthly impact",
            f"${_monthly:+,.0f}",
            delta="projected" if _delta_headline != 0 else None,
            good=True if _daily_good else False if _daily_bad else None,
        )
    with cols[4]:
        metric("AWS tool calls", verdict.tool_calls)

    _PR_MATERIAL_USD = 5.0
    if verdict.direction == "increase" and _delta_headline >= _PR_MATERIAL_USD:
        render_notification_button(
            button_label="Notify PR cost risk",
            state_key=f"prp::{active.profile}::{pr_url.strip()}",
            draft=NotificationDraft(
                title="PR cost increase risk",
                severity="High",
                reason=(
                    f"PR analysis predicts a material cost increase of "
                    f"${_delta_headline:+,.2f}/day ({verdict.verdict})."
                ),
                recipient="finops-team@example.com",
                subject=(
                    f"[CostSense] PR cost risk — ${_delta_headline:+,.0f}/day "
                    f"({active.profile})"
                ),
                body=(
                    f"CostSense PR Predictor flagged a material cost increase.\n\n"
                    f"Account: {active.profile}\n"
                    f"PR: {pr_url.strip()}\n"
                    f"Verdict: {verdict.verdict}\n"
                    f"Estimated daily impact: ${_delta_headline:+,.2f}\n"
                    f"Estimated monthly impact: ${_monthly:+,.0f}\n"
                    f"Confidence: {(verdict.confidence or 'medium').title()}\n\n"
                    f"Please review the PR diff and recommendations in CostSense."
                ),
                source_page="PR Predictor",
                source_type="pr_cost_increase",
            ),
        )

    if _has_range:
        _lo_signed, _hi_signed = min(_lo, _hi), max(_lo, _hi)
        st.caption(
            f"Daily range \\${_lo_signed:+,.2f} to \\${_hi_signed:+,.2f}  ·  "
            f"Monthly range \\${_lo_signed * 30:+,.0f} to "
            f"\\${_hi_signed * 30:+,.0f}"
        )

    # ---- Basis + Confidence pills (grounding transparency) ------------
    _basis = (verdict.estimation_basis or "measured").lower()
    _measured = bool(verdict.measured)
    _conf = (verdict.confidence or "medium").lower()
    if _basis == "measured" and _measured:
        _basis_pill_level = "Low"
        _basis_text = "Measured (CloudWatch / Cost Explorer)"
    elif _basis == "sibling_account":
        _basis_pill_level = "Medium"
        _basis_text = "Estimated from peer AWS account (historical precedent)"
    elif _basis == "unknown":
        _basis_pill_level = "High"
        _basis_text = "Unquantifiable — no reachable AWS grounding"
    else:
        _basis_pill_level = "Medium"
        _basis_text = "Basis unknown"
    _conf_pill_level = {"high": "Low", "medium": "Medium",
                        "low": "High"}.get(_conf, "Medium")

    meta_row([
        ("Basis", _basis_pill_level, _basis_text),
        ("Confidence", _conf_pill_level, _conf.title()),
    ])

    fc1, fc2 = st.columns([2, 1], gap="medium", vertical_alignment="bottom")
    with fc1:
        deploy_date = st.date_input(
            "Expected merge / deploy date",
            value=date.today() + timedelta(days=14),
            key=f"prp_fc_deploy::{active.account_id}::{pr_url.strip()}",
            help="When this PR's cost impact is expected to land in the forecast.",
        )
    with fc2:
        if st.button(
            "Add to future forecast",
            key=f"prp_fc_add::{active.account_id}::{pr_url.strip()}",
            use_container_width=True,
        ):
            ev = event_from_pr_predictor(
                pr_url.strip(),
                verdict,
                expected_deploy=deploy_date,
            )
            if ev is None:
                st.warning("No material cost delta to add to the forecast.")
            else:
                queue_pending_event(active.profile, ev)
                callout(
                    "Queued for Future Forecast — open that page to review "
                    "and toggle the event.",
                    tone="success",
                )

    if not _measured or _basis != "measured":
        if _basis == "sibling_account":
            st.caption(
                "This number is an estimate, not a measured cost. It comes "
                "from historical precedent: a prior scope-expansion PR in "
                "this repo, and the step change it caused in a sibling AWS "
                "account around its merge date."
            )
        else:
            st.caption(
                "No reachable AWS account runs the service this PR affects, "
                "and no historical precedent was found in this repo. The "
                "true cost delta is greater than \\$0/day but cannot be "
                "quantified from available data."
            )

    if narrative:
        st.markdown("**Cost summary:**")
        st.markdown(_md(narrative))

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
