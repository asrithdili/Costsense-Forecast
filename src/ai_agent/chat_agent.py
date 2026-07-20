"""Conversational FinOps bot with read-only AWS access.

Maintains a message history across turns so the user can ask follow-ups.
Every tool call goes through `scrub()` (from aws_tools_broad) which recursively
strips anything that looks like a secret, IAM policy document, JWT, private
key, or AWS access key ID — before Claude sees the tool output.

Public API:
    chat_step(profile, model_id, history, user_msg) -> ChatTurn

`history` is the running list of {role, content} messages Claude will see.
`ChatTurn` bundles the assistant reply text, tool-call transcript, and the
updated history to store back in session_state.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import boto3

from src.ai_agent.aws_tools import TOOLS as BASE_TOOLS
from src.ai_agent.aws_tools_broad import BROAD_TOOLS, all_broad_specs, scrub


BEDROCK_REGION = "us-west-2"
DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"
MAX_TOOL_TURNS_PER_QUESTION = 12
MAX_TOKENS = 3000


SYSTEM = """You are CostSense — a senior AWS FinOps analyst embedded in a \
chat interface. The user will ask questions about their AWS account and you \
will ground every answer in real data by calling the read-only tools you \
have available.

Your tools cover: Cost Explorer, CloudWatch metrics, CloudTrail events, \
resource inventory (Lambda, RDS, EC2, NAT, EBS, S3, DynamoDB), AWS Compute \
Optimizer, AWS Budgets, Service Quotas, S3 lifecycle policies, and rightsizing \
recommendations.

Rules:
- NEVER guess. When the user asks "what's my biggest cost driver," call \
`cost_by_service` and report the actual top service. When they ask "is my \
Lambda oversized," call `compute_optimizer_lambda`.
- Answer the question they asked. Don't hallucinate follow-up asks the user \
didn't make. Keep replies short and direct.
- When you cite a number, say WHICH tool call it came from ("via \
cost_by_service") so the user can trust it.
- If a tool errors or returns empty, say so plainly. Don't pretend a value \
you fabricated came from a tool.
- Assume you cannot see secrets. Fields that come back as \
"[REDACTED-BY-COSTSENSE]" have been stripped by policy — don't demand them.
- If the user asks something out of AWS scope (e.g. "write me a poem"), \
politely redirect to FinOps.

Response format: plain markdown. Use bullet points and short paragraphs. \
No JSON unless the user explicitly asks for JSON."""


@dataclass
class ToolCall:
    name: str
    input: dict
    output_summary: str        # first ~200 chars of the (scrubbed) result


@dataclass
class ChatTurn:
    reply: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    updated_history: list[dict] = field(default_factory=list)
    error: str | None = None
    model_id: str = ""


def _merged_tool_specs() -> list[dict]:
    """All tools available to the chat agent: base + broad."""
    from src.ai_agent.aws_tools import tool_specs as base_specs
    return base_specs() + all_broad_specs()


def _run_tool(name: str, args: dict, profile: str | None) -> dict:
    """Route to whichever registry has the tool, then scrub secrets."""
    if name in BASE_TOOLS:
        fn, _ = BASE_TOOLS[name]
    elif name in BROAD_TOOLS:
        fn, _ = BROAD_TOOLS[name]
    else:
        return {"error": f"unknown tool: {name}"}
    kwargs = dict(args or {})
    kwargs["profile"] = profile
    try:
        result = fn(**kwargs)
    except Exception as e:  # noqa: BLE001
        return {"error": f"tool crashed: {e}"}
    return scrub(result)


def _summarize_output(obj) -> str:
    s = json.dumps(obj, default=str)
    return s[:220] + ("…" if len(s) > 220 else "")


def chat_step(
    profile: str | None,
    model_id: str,
    history: list[dict],
    user_msg: str,
) -> ChatTurn:
    """Advance one user turn. Runs the tool-use loop for this question and
    returns the assistant reply plus updated history."""
    from src.ai_agent.bedrock_client import make_client
    client = make_client(profile, region=BEDROCK_REGION)

    messages = list(history) + [{"role": "user", "content": user_msg}]
    tool_calls: list[ToolCall] = []

    for _ in range(MAX_TOOL_TURNS_PER_QUESTION + 1):
        try:
            resp = client.invoke_model(
                modelId=model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.2,
                    "system": SYSTEM,
                    "tools": _merged_tool_specs(),
                    "messages": messages,
                }),
            )
            payload = json.loads(resp["body"].read())
        except Exception as e:  # noqa: BLE001
            return ChatTurn(reply="", error=f"bedrock invoke failed: {e}",
                            model_id=model_id,
                            updated_history=history + [
                                {"role": "user", "content": user_msg}])

        content = payload.get("content", [])
        tool_use_blocks = [b for b in content if b.get("type") == "tool_use"]

        if tool_use_blocks:
            messages.append({"role": "assistant", "content": content})
            results = []
            for blk in tool_use_blocks:
                out = _run_tool(blk["name"], blk.get("input") or {}, profile)
                tool_calls.append(ToolCall(
                    name=blk["name"],
                    input=blk.get("input") or {},
                    output_summary=_summarize_output(out),
                ))
                # Cap payload sent back to Claude
                serialized = json.dumps(out, default=str)
                if len(serialized) > 8000:
                    # Structural trim: send only top-level keys + counts if huge
                    if isinstance(out, dict):
                        trimmed = {k: (f"<large list, len={len(v)}>"
                                       if isinstance(v, list) and len(v) > 30
                                       else v)
                                   for k, v in out.items()}
                        serialized = json.dumps(trimmed, default=str)[:8000]
                    else:
                        serialized = serialized[:8000] + "…[truncated]"
                results.append({
                    "type": "tool_result",
                    "tool_use_id": blk["id"],
                    "content": serialized,
                })
            messages.append({"role": "user", "content": results})
            continue

        # Terminal — no more tool calls, model wants to reply.
        text_block = next((b for b in content if b.get("type") == "text"), None)
        reply = text_block.get("text", "") if text_block else "(no reply)"

        # Persist the final assistant turn in the history so follow-ups see it.
        messages.append({"role": "assistant", "content": content})
        return ChatTurn(
            reply=reply,
            tool_calls=tool_calls,
            updated_history=messages,
            model_id=model_id,
        )

    return ChatTurn(
        reply="",
        tool_calls=tool_calls,
        updated_history=messages,
        error=f"exceeded {MAX_TOOL_TURNS_PER_QUESTION} tool turns without "
              "a final reply",
        model_id=model_id,
    )
