"""CostSense AI Chat — a FinOps bot with read-only AWS access.

Full conversational interface: ask any question about the selected AWS
account, follow up with more questions, and the bot uses its AWS tools
(CloudWatch, CloudTrail, Cost Explorer, Compute Optimizer, Budgets, S3,
DynamoDB, etc.) to ground every answer in real data.

Secrets are auto-redacted from every tool output before the model sees them.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import load_config
from src.env import load_env

load_env()
load_config()

import streamlit as st

from src.ai_agent.chat_agent import chat_step
from src.dashboard.costsense_theme import section
from src.dashboard.nav import inject_css, render as render_nav
from src.dashboard.nav import render_sidebar_footer, render_sidebar_header


st.set_page_config(page_title="CostSense · AI Chat", layout="wide")
inject_css()

# Render the Diligent brand card FIRST — before any AWS calls — so it
# appears instantly regardless of STS latency.
render_sidebar_header()

# Render title first so the page isn't blank while SSO/STS profile
# resolution runs inside render_nav().
section(
    "Ask CostSense",
    "Chat with a FinOps agent that has read-only access to your AWS "
    "account. Ask about spend, resources, PRs, or recommendations — "
    "answers are grounded in live AWS APIs with secrets auto-redacted.",
    kicker="Assistant",
)

with st.spinner("Resolving AWS profiles…"):
    sel = render_nav(include_model=True)
active = sel.profile
model_id = sel.model_id


# ---------- sidebar ----------

with st.sidebar:
    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.chat_display = []
        st.rerun()

    # Probe GitHub availability once per render so we can be honest about
    # what the bot can actually reach. We call chat_agent's own probe (same
    # signal fed to the model) so sidebar and bot stay in sync.
    from src.ai_agent.chat_agent import _detect_github_read_available
    _gh_read_ok = _detect_github_read_available()
    _gh_label = ("GitHub repo browsing (search repos, files, code, PRs) — "
                 f"{'✓ available' if _gh_read_ok else '✗ not configured'}")

    st.caption("**Tools available to the bot**")
    st.caption(
        "• Cost Explorer (spend by day / service)  \n"
        "• CloudWatch metrics (Lambda, RDS, ECS, Logs)  \n"
        "• CloudTrail events (recent console changes)  \n"
        "• Resource inventory (EC2, RDS, Lambda, NAT, EBS, S3, DynamoDB)  \n"
        "• Compute Optimizer (EC2 + Lambda rightsizing)  \n"
        "• AWS Budgets, Service Quotas, S3 lifecycle policies  \n"
        "• AWS Pricing API  \n"
        f"• {_gh_label}"
    )
    st.caption("**Auto-redacted**")
    st.caption("Secrets, tokens, IAM policy docs, private keys, JWTs, and "
               "AWS access keys are stripped before the model sees them.")

    render_sidebar_footer(
        active_profile=active.profile,
        account_id=active.account_id,
        extra_rows=[("Model", (model_id or "").split(".")[-1] or "-")],
    )


# ---------- session-state chat history ----------

# `chat_history` = list of {role, content} sent to Claude (grows over time)
# `chat_display` = list of {role, text, tool_calls} for rendering
# `pending_question` = a question queued for processing on THIS rerun, after
# the render loop below has already echoed the user's bubble — otherwise the
# UI shows nothing for the entire (often 30s+) tool-calling loop and looks
# stuck, which is what pushes people to submit twice.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# ---------- suggested-question chips ----------

SUGGESTIONS = [
    "What's my biggest cost driver in the last 14 days?",
    "Find idle resources I could shut down.",
    "Are any of my Lambdas oversized?",
    "What changed in cost yesterday vs last Wednesday?",
    "Which S3 buckets have no lifecycle policy?",
    "Am I close to any service quotas?",
]


def _safe_md(text: str) -> None:
    """Render markdown without Streamlit treating $...$ as LaTeX math."""
    st.markdown(text.replace("$", "\\$"))


def _queue_question(q: str) -> None:
    """Echo the user's message right away and defer the (slow) agent call
    to the pending-question block below, which runs after this bubble is
    already on screen."""
    if st.session_state.pending_question:
        return  # a question is already in flight — ignore double submits
    st.session_state.chat_display.append(
        {"role": "user", "text": q, "tool_calls": []}
    )
    st.session_state.pending_question = q


def _run_pending_question(q: str) -> None:
    """Actually call the agent. Runs inside a spinner so multi-tool-call
    turns (AWS + GitHub, sometimes 30s+) show visible progress."""
    try:
        turn = chat_step(
            profile=active.profile,
            model_id=model_id,
            history=st.session_state.chat_history,
            user_msg=q,
            account_id=active.account_id,
            github_read_available=_gh_read_ok,
        )
    except Exception as e:  # noqa: BLE001
        st.session_state.chat_display.append({
            "role": "assistant",
            "text": f"crashed: {e}",
            "tool_calls": [],
        })
        st.session_state.chat_display[-1]["_trace"] = traceback.format_exc()
        return
    if turn.error:
        st.session_state.chat_display.append({
            "role": "assistant",
            "text": f"Error: {turn.error}",
            "tool_calls": turn.tool_calls,
        })
        return
    st.session_state.chat_history = turn.updated_history
    st.session_state.chat_display.append({
        "role": "assistant",
        "text": turn.reply,
        "tool_calls": turn.tool_calls,
        "guard_triggered": turn.guard_triggered,
        "guard_reason": turn.guard_reason,
    })


# ---------- render existing chat ----------

for msg in st.session_state.chat_display:
    with st.chat_message(msg["role"]):
        if msg.get("guard_triggered"):
            reason = msg.get("guard_reason") or ""
            if reason.startswith("substitution"):
                banner_text = (
                    "**Scope-substitution guard intercepted this reply.** "
                    "The model refused the account you actually asked "
                    "about, then handed over the currently connected "
                    "account's data as a consolation. Below is the honest "
                    "scope message instead."
                )
            else:
                banner_text = (
                    "**Hallucination guard intercepted this reply.** "
                    "One or more AWS tool calls were denied by IAM, and "
                    "the model's original answer contained dollar figures "
                    "that weren't grounded in a successful API response. "
                    "Below is the honest fallback message."
                )
            st.error(banner_text, icon="🛑")
            if reason:
                with st.expander("Why the guard fired"):
                    st.caption(reason)
        _safe_md(msg["text"])
        if msg.get("_trace"):
            with st.expander("Traceback"):
                st.code(msg["_trace"])


# ---------- suggested chips (only show when chat is empty) ----------

if not st.session_state.chat_display:
    with st.container(border=True):
        section(
            "Suggested questions",
            "Pick one to get started, or type your own below.",
            kicker="Start here",
        )
        cols = st.columns(3, gap="medium")
        for i, s in enumerate(SUGGESTIONS):
            if cols[i % 3].button(
                s, key=f"sug_{i}", use_container_width=True, type="secondary",
            ):
                _queue_question(s)
                st.rerun()


# ---------- input box ----------

if st.session_state.chat_display:
    st.divider()

user_q = st.chat_input(
    "Ask about your AWS costs, resources, or PRs…",
    disabled=bool(st.session_state.pending_question),
)
if user_q:
    _queue_question(user_q)
    st.rerun()


# ---------- process the queued question (user bubble above is already
# visible by this point) ----------

if st.session_state.pending_question:
    q = st.session_state.pending_question
    with st.chat_message("assistant"):
        with st.spinner("Thinking — querying AWS / GitHub tools… this can "
                        "take a while for multi-step questions."):
            _run_pending_question(q)
    st.session_state.pending_question = None
    st.rerun()
