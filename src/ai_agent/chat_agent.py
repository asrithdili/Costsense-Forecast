"""Conversational FinOps bot with read-only AWS + GitHub access.

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
from src.ai_agent.github_tools import GITHUB_TOOLS, all_github_specs


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
recommendations. You ALSO have read-only GitHub tools — you CAN and SHOULD \
access GitHub repos directly: search for a repo by name, list its files, \
read file contents, search code, and list/diff pull requests. Never claim \
GitHub access is "outside your scope" — it isn't.

Rules:
- NEVER guess. When the user asks "what's my biggest cost driver," call \
`cost_by_service` and report the actual top service. When they ask "is my \
Lambda oversized," call `compute_optimizer_lambda`.
- If the user references a repo by name only (not "org/repo"), ALWAYS call \
`github_search_repositories` first to resolve it — it checks the \
DiligentCorp org first, then falls back to a global search — then use the \
matched `full_name` in the other `github_*` tools. Only tell the user you \
couldn't find a repo after this search comes back empty.
- To analyze a repo's cost impact (e.g. "analyse the cost of X repo"), read \
its IaC/config files (`github_get_file` on Terraform/CDK/Dockerfiles, etc.) \
and cross-reference the resources you find with AWS tools (cost_by_service, \
CloudWatch, resource inventory) to ground the analysis in real spend.
- Answer the question they asked. Don't hallucinate follow-up asks the user \
didn't make. Keep replies short and direct.
- When you cite a number or fact, say WHICH tool call it came from ("via \
cost_by_service") so the user can trust it.
- If a tool errors or returns empty, say so plainly. Don't pretend a value \
you fabricated came from a tool.
- Assume you cannot see secrets. Fields that come back as \
"[REDACTED-BY-COSTSENSE]" have been stripped by policy — don't demand them.
- If the user asks something truly out of scope (e.g. "write me a poem"), \
politely redirect to FinOps or the connected GitHub repos.

Response format: plain markdown. Use bullet points and short paragraphs. \
No JSON unless the user explicitly asks for JSON. Always write dollar \
figures with a leading "$" (e.g. "$400–800", "$65K/year", never a bare \
"400–800" or "65K/year") since these are all USD amounts."""


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
    """All tools available to the chat agent: AWS base + broad + GitHub."""
    from src.ai_agent.aws_tools import tool_specs as base_specs
    return base_specs() + all_broad_specs() + all_github_specs()


def _run_tool(name: str, args: dict, profile: str | None) -> dict:
    """Route to whichever registry has the tool, then scrub secrets."""
    kwargs = dict(args or {})
    if name in BASE_TOOLS:
        fn, _ = BASE_TOOLS[name]
        kwargs["profile"] = profile
    elif name in BROAD_TOOLS:
        fn, _ = BROAD_TOOLS[name]
        kwargs["profile"] = profile
    elif name in GITHUB_TOOLS:
        fn, _ = GITHUB_TOOLS[name]  # GitHub tools auth via gh CLI / GITHUB_TOKEN
    else:
        return {"error": f"unknown tool: {name}"}
    try:
        result = fn(**kwargs)
    except Exception as e:  # noqa: BLE001
        return {"error": f"tool crashed: {e}"}
    return scrub(result)


def _summarize_output(obj) -> str:
    s = json.dumps(obj, default=str)
    return s[:220] + ("…" if len(s) > 220 else "")


def run_agent_with_tools(
    profile: str | None,
    model_id: str,
    *,
    system: str,
    user_msg: str,
    history: list[dict] | None = None,
    max_tool_turns: int = MAX_TOOL_TURNS_PER_QUESTION,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.2,
) -> ChatTurn:
    """Run one agent turn with AWS + GitHub tools and a custom system prompt."""
    from src.ai_agent.bedrock_client import make_client
    client = make_client(profile, region=BEDROCK_REGION)

    base_history = list(history or [])
    messages = base_history + [{"role": "user", "content": user_msg}]
    tool_calls: list[ToolCall] = []

    for _ in range(max_tool_turns + 1):
        try:
            resp = client.invoke_model(
                modelId=model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "system": system,
                    "tools": _merged_tool_specs(),
                    "messages": messages,
                }),
            )
            payload = json.loads(resp["body"].read())
        except Exception as e:  # noqa: BLE001
            return ChatTurn(
                reply="",
                error=f"bedrock invoke failed: {e}",
                model_id=model_id,
                updated_history=base_history + [
                    {"role": "user", "content": user_msg},
                ],
            )

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
                serialized = json.dumps(out, default=str)
                if len(serialized) > 8000:
                    if isinstance(out, dict):
                        trimmed = {
                            k: (
                                f"<large list, len={len(v)}>"
                                if isinstance(v, list) and len(v) > 30
                                else v
                            )
                            for k, v in out.items()
                        }
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

        text_block = next((b for b in content if b.get("type") == "text"), None)
        reply = text_block.get("text", "") if text_block else "(no reply)"
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
        error=f"exceeded {max_tool_turns} tool turns without a final reply",
        model_id=model_id,
    )


def chat_step(
    profile: str | None,
    model_id: str,
    history: list[dict],
    user_msg: str,
) -> ChatTurn:
    """Advance one user turn. Runs the tool-use loop for this question and
    returns the assistant reply plus updated history."""
    turn = run_agent_with_tools(
        profile,
        model_id,
        system=SYSTEM,
        user_msg=user_msg,
        history=history,
    )
    if turn.error:
        return ChatTurn(
            reply="",
            error=turn.error,
            model_id=model_id,
            updated_history=history + [{"role": "user", "content": user_msg}],
        )
    return turn
