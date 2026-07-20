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
from src.aws.profiles import resolve_all


st.set_page_config(page_title="CostSense · AI Chat", layout="wide",
                   page_icon="🤖")

st.title("CostSense AI")
st.caption("Chat with a FinOps agent that has read-only access to your AWS "
           "account. Ask anything about spend, resources, PRs, or "
           "recommendations. The bot uses live AWS APIs and auto-redacts "
           "anything that looks like a secret.")


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
    model_id = st.selectbox(
        "Bedrock model",
        index=1,   # default to Sonnet
        options=[
            "anthropic.claude-3-haiku-20240307-v1:0",
            "us.anthropic.claude-sonnet-4-6",
        ],
        help="Sonnet handles multi-turn tool-use well and is recommended.",
    )

    st.divider()
    if st.button("Clear chat"):
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
        "• AWS Pricing API"
    )
    st.caption("**Auto-redacted**")
    st.caption("Secrets, tokens, IAM policy docs, private keys, JWTs, and "
               "AWS access keys are stripped before the model sees them.")


# ---------- session-state chat history ----------

# `chat_history` = list of {role, content} sent to Claude (grows over time)
# `chat_display` = list of {role, text, tool_calls} for rendering
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_display" not in st.session_state:
    st.session_state.chat_display = []


# ---------- suggested-question chips ----------

SUGGESTIONS = [
    "What's my biggest cost driver in the last 14 days?",
    "Find idle resources I could shut down.",
    "Are any of my Lambdas oversized?",
    "What changed in cost yesterday vs last Wednesday?",
    "Which S3 buckets have no lifecycle policy?",
    "Am I close to any service quotas?",
]


def _handle_question(q: str) -> None:
    """Run one chat step, append to both histories, rerun to display."""
    st.session_state.chat_display.append(
        {"role": "user", "text": q, "tool_calls": []}
    )
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
        st.markdown(msg["text"])
        if msg["role"] == "assistant" and msg["tool_calls"]:
            with st.expander(f"🔧 {len(msg['tool_calls'])} AWS tool call(s)"):
                for tc in msg["tool_calls"]:
                    st.markdown(f"**`{tc.name}`** · args: `{tc.input}`")
                    st.code(tc.output_summary, language="json")
        if msg.get("_trace"):
            with st.expander("Traceback"):
                st.code(msg["_trace"])


# ---------- suggested chips (only show when chat is empty) ----------

if not st.session_state.chat_display:
    st.markdown("**Try a suggested question:**")
    cols = st.columns(3)
    for i, s in enumerate(SUGGESTIONS):
        if cols[i % 3].button(s, key=f"sug_{i}", use_container_width=True):
            _handle_question(s)
            st.rerun()


# ---------- input box ----------

user_q = st.chat_input("Ask about your AWS costs, resources, or PRs…")
if user_q:
    _handle_question(user_q)
    st.rerun()
