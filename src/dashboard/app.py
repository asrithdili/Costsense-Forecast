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

import streamlit as st

from src.ai_agent.chat_agent import chat_step
from src.dashboard.nav import render as render_nav


st.set_page_config(page_title="CostSense · AI Chat", layout="wide",
                   page_icon="🤖")

# Render title first so the page isn't blank while SSO/STS profile
# resolution runs inside render_nav().
st.title("CostSense AI")
st.caption("Chat with a FinOps agent that has read-only access to your AWS "
           "account. Ask anything about spend, resources, PRs, or "
           "recommendations. The bot uses live AWS APIs and auto-redacts "
           "anything that looks like a secret.")

sel = render_nav(include_model=True)
active = sel.profile
model_id = sel.model_id


# ---------- sidebar ----------

with st.sidebar:
    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.chat_display = []
        st.rerun()

    st.caption("**Tools available to the bot**")
    st.caption(
        "• Cost Explorer (spend by day / service)  \n"
        "• CloudWatch metrics (Lambda, RDS, ECS, Logs)  \n"
        "• CloudTrail events (recent console changes)  \n"
        "• Resource inventory (EC2, RDS, Lambda, NAT, EBS, S3, DynamoDB)  \n"
        "• Compute Optimizer (EC2 + Lambda rightsizing)  \n"
        "• AWS Budgets, Service Quotas, S3 lifecycle policies  \n"
        "• AWS Pricing API  \n"
        "• GitHub repo browsing (search repos, files, code, PRs)"
    )
    st.caption("**Auto-redacted**")
    st.caption("Secrets, tokens, IAM policy docs, private keys, JWTs, and "
               "AWS access keys are stripped before the model sees them.")


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
            "text": f"❌ {turn.error}",
            "tool_calls": turn.tool_calls,
        })
        return
    st.session_state.chat_history = turn.updated_history
    st.session_state.chat_display.append({
        "role": "assistant",
        "text": turn.reply,
        "tool_calls": turn.tool_calls,
    })


# ---------- render existing chat ----------

for msg in st.session_state.chat_display:
    with st.chat_message(msg["role"]):
        # Escape "$" so dollar figures (e.g. "$400-800") don't get parsed as
        # LaTeX math by Streamlit's markdown renderer — that mangles the
        # font (serif/italic, wrong size) for anything between two "$".
        st.markdown(msg["text"].replace("$", "\\$"))
        if msg.get("_trace"):
            with st.expander("Traceback"):
                st.code(msg["_trace"])


# ---------- suggested chips (only show when chat is empty) ----------

if not st.session_state.chat_display:
    st.markdown("**Try a suggested question:**")
    cols = st.columns(3)
    for i, s in enumerate(SUGGESTIONS):
        if cols[i % 3].button(s, key=f"sug_{i}", use_container_width=True):
            _queue_question(s)
            st.rerun()


# ---------- input box ----------

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
