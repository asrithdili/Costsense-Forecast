"""Conversational FinOps bot with read-only AWS + GitHub access.

Maintains a message history across turns so the user can ask follow-ups.
Every tool call goes through `scrub()` (from aws_tools_broad) which recursively
strips anything that looks like a secret, IAM policy document, JWT, private
key, or AWS access key ID — before Claude sees the tool output.

Hallucination hardening (added 2026-07):
    LLMs will fabricate dollar figures when a tool errors, even with a system
    prompt telling them not to. Three layers of defense keep the assistant
    honest when the user's AWS profile can't read a resource:
      1. Every tool_result whose payload contains an ``error`` field is sent
         back to Claude with Anthropic's ``is_error: true`` flag — the model
         treats explicit is_error results very differently from a JSON string
         that happens to contain the word "error".
      2. The system prompt injects the ACTIVE profile + account id so the
         model can't answer questions about a different account.
      3. A post-loop guard scans the final reply. If any tool call was denied
         AND the reply contains a $-figure, we replace the reply with an
         explicit "I don't have access to X" message. Deterministic backstop
         for the LLM-based rules above.

Public API:
    chat_step(profile, model_id, history, user_msg, account_id) -> ChatTurn

`history` is the running list of {role, content} messages Claude will see.
`ChatTurn` bundles the assistant reply text, tool-call transcript, and the
updated history to store back in session_state.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import boto3

from src.ai_agent.aws_tools import TOOLS as BASE_TOOLS
from src.ai_agent.aws_tools_broad import BROAD_TOOLS, all_broad_specs, scrub
from src.ai_agent.github_tools import GITHUB_TOOLS, all_github_specs


BEDROCK_REGION = "us-west-2"
DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"
MAX_TOOL_TURNS_PER_QUESTION = 12
MAX_TOKENS = 3000


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------
#
# Every tool in aws_tools / aws_tools_broad / github_tools wraps its
# implementation in a `_safe` decorator that returns
# ``{"error": "<ExceptionClassName>: <message>"}`` when boto3 or the GitHub
# API raises. We inspect that string to bucket failures — the buckets drive
# both the is_error flag we send to Claude and the post-loop guard.

_NO_ACCESS_MARKERS = (
    "AccessDenied",
    "UnauthorizedOperation",
    "AuthFailure",
    "InvalidClientTokenId",
    "SignatureDoesNotMatch",
    "NoCredentials",
    "TokenRefreshRequired",
    "ExpiredToken",
    "SSOTokenLoadError",
    "UnrecognizedClientException",
    "not authorized",
    "is not authorized to perform",
    "403",
)

_TRANSIENT_MARKERS = (
    "Throttling",
    "ThrottlingException",
    "RequestLimitExceeded",
    "TooManyRequests",
    "ServiceUnavailable",
    "InternalServerError",
    "500",
    "502",
    "503",
    "504",
    "EndpointConnectionError",
    "ReadTimeoutError",
    "ConnectTimeoutError",
)

_NOT_FOUND_MARKERS = (
    "NoSuchEntity",
    "NotFound",
    "ResourceNotFoundException",
    "NoSuchBucket",
    "InvalidParameterValueException",
    "404",
)


def _classify_error(error_text: str) -> str:
    """Bucket a raw error string into a coarse kind. The buckets are
    deliberately coarse: the goal is to detect *access denial* reliably so
    the post-loop guard can catch fabrications; other kinds are informational.
    """
    if not error_text:
        return "unknown"
    lowered = str(error_text)
    for marker in _NO_ACCESS_MARKERS:
        if marker in lowered:
            return "no_access"
    for marker in _TRANSIENT_MARKERS:
        if marker in lowered:
            return "transient"
    for marker in _NOT_FOUND_MARKERS:
        if marker in lowered:
            return "not_found"
    return "unknown"


def _tool_result_meta(result) -> tuple[bool, str | None, str | None]:
    """Return ``(is_error, error_kind, error_text)`` for a tool result.

    Success case returns ``(False, None, None)``. Any dict with a truthy
    ``error`` key is treated as a failure.
    """
    if isinstance(result, dict) and result.get("error"):
        error_text = str(result["error"])
        return True, _classify_error(error_text), error_text
    return False, None, None


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_BASE = """You are CostSense — a senior AWS FinOps analyst embedded in a \
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

CRITICAL ANTI-HALLUCINATION RULES:
- If ANY tool_result comes back with is_error=true, treat that call as \
having produced NO DATA. You may not use its content as a source of numbers.
- If an is_error tool_result mentions access denial (AccessDenied, not \
authorized, NoCredentials, ExpiredToken), tell the user their profile lacks \
access to that resource — do NOT retry the same call and do NOT substitute \
a plausible number in its place.
- If ALL tools relevant to the user's question failed, your reply MUST NOT \
contain any dollar figure. Say plainly that you can't answer without \
access and suggest switching profiles or asking the account owner.
- Numbers you cannot ground in a successful tool_result MUST be omitted. \
Prose ("Lambda is your biggest driver") is fine when supported by a tool; \
figures ($400/day, ~$65K/month) are ONLY fine when a tool call actually \
returned them.

Response format: plain markdown. Use bullet points and short paragraphs. \
No JSON unless the user explicitly asks for JSON. Always write dollar \
figures with a leading "$" (e.g. "$400–800", "$65K/year", never a bare \
"400–800" or "65K/year") since these are all USD amounts."""


def _build_system(profile: str | None, account_id: str | None) -> str:
    """Prepend the active-scope block. Front-loading the profile + account
    before the rules block anchors the model to the correct scope earlier
    in the context window."""
    scope_lines = ["ACTIVE SCOPE:"]
    if profile:
        scope_lines.append(f"- AWS profile: {profile}")
    else:
        scope_lines.append("- AWS profile: (none — tools will fail)")
    if account_id:
        scope_lines.append(f"- AWS account id: {account_id}")
    else:
        scope_lines.append(
            "- AWS account id: (unknown — the profile could not resolve "
            "get-caller-identity; assume you have no AWS access)"
        )
    scope_lines.append(
        "You MUST NOT report figures for any OTHER account. If the user "
        "asks about a different account id, tell them to switch profiles "
        "in the sidebar — do not attempt to answer."
    )
    return "\n".join(scope_lines) + "\n\n" + _SYSTEM_BASE


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    name: str
    input: dict
    output_summary: str        # first ~200 chars of the (scrubbed) result
    is_error: bool = False
    error_kind: str | None = None
    error_text: str | None = None


@dataclass
class ChatTurn:
    reply: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    updated_history: list[dict] = field(default_factory=list)
    error: str | None = None
    model_id: str = ""
    guard_triggered: bool = False   # True when post-loop guard rewrote reply
    guard_reason: str | None = None


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Post-loop hallucination guard
# ---------------------------------------------------------------------------

# Matches "$400", "$1,200.50", "$65K", "$65k", "$1.2M", "$400/day", etc.
_DOLLAR_FIGURE_RE = re.compile(r"\$\s?[\d][\d,\.]*\s?[KkMmBb]?")


def _apply_hallucination_guard(
    reply: str,
    tool_calls: list[ToolCall],
    profile: str | None,
    account_id: str | None,
) -> tuple[str, bool, str | None]:
    """If any tool call failed with ``no_access`` AND the reply contains a
    dollar figure, replace the reply with an honest denial. This is the
    deterministic backstop for the anti-hallucination rules in the system
    prompt — it catches cases where the model produces $-figures anyway.

    Returns ``(new_reply, guard_triggered, reason)``.
    """
    denied_tools = [tc for tc in tool_calls
                    if tc.is_error and tc.error_kind == "no_access"]
    if not denied_tools:
        return reply, False, None

    figures = _DOLLAR_FIGURE_RE.findall(reply or "")
    if not figures:
        # Model correctly avoided fabricating a number — nothing to override,
        # but we still return a "guard did not trigger" so the caller can
        # optionally surface a caveat.
        return reply, False, None

    tool_names = sorted({tc.name for tc in denied_tools})
    scope_line = (
        f"profile `{profile}` (account {account_id})"
        if account_id else f"profile `{profile or '(none)'}`"
    )
    honest_reply = (
        f"I can't answer this with real numbers — {scope_line} was denied "
        f"access on: {', '.join(tool_names)}.\n\n"
        f"AWS returned an authorization error, so any dollar figures I "
        f"gave you here would be a guess, not real data. Switch to a "
        f"profile that has read access to this account (Cost Explorer + "
        f"the relevant service APIs), or ask the account owner to run the "
        f"question — I'll ground the answer in the actual API response."
    )
    reason = (
        f"reply contained {len(figures)} dollar figure(s) but "
        f"{len(denied_tools)} tool call(s) were denied: "
        f"{', '.join(tool_names)}"
    )
    return honest_reply, True, reason


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def chat_step(
    profile: str | None,
    model_id: str,
    history: list[dict],
    user_msg: str,
    account_id: str | None = None,
) -> ChatTurn:
    """Advance one user turn. Runs the tool-use loop for this question and
    returns the assistant reply plus updated history.

    ``account_id`` is the id resolved from ``profile`` via STS
    get-caller-identity. It's used to scope the system prompt and to phrase
    the denial message when the post-loop guard fires. Callers that don't
    have it can pass ``None`` — the guard still works, it just can't name
    the account in the denial.
    """
    from src.ai_agent.bedrock_client import make_client
    client = make_client(profile, region=BEDROCK_REGION)

    system_prompt = _build_system(profile, account_id)
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
                    "system": system_prompt,
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
                is_err, err_kind, err_text = _tool_result_meta(out)
                tool_calls.append(ToolCall(
                    name=blk["name"],
                    input=blk.get("input") or {},
                    output_summary=_summarize_output(out),
                    is_error=is_err,
                    error_kind=err_kind,
                    error_text=err_text,
                ))
                # Cap payload sent back to Claude
                serialized = json.dumps(out, default=str)
                if len(serialized) > 8000:
                    if isinstance(out, dict):
                        trimmed = {k: (f"<large list, len={len(v)}>"
                                       if isinstance(v, list) and len(v) > 30
                                       else v)
                                   for k, v in out.items()}
                        serialized = json.dumps(trimmed, default=str)[:8000]
                    else:
                        serialized = serialized[:8000] + "…[truncated]"
                result_block = {
                    "type": "tool_result",
                    "tool_use_id": blk["id"],
                    "content": serialized,
                }
                # Anthropic Messages API: setting is_error=true on the
                # tool_result changes how the model treats the payload —
                # it's much less likely to hallucinate a value from an
                # explicitly-erroring call than from a JSON blob that
                # happens to contain the word "error".
                if is_err:
                    result_block["is_error"] = True
                results.append(result_block)
            messages.append({"role": "user", "content": results})
            continue

        # Terminal — no more tool calls, model wants to reply.
        text_block = next((b for b in content if b.get("type") == "text"), None)
        reply = text_block.get("text", "") if text_block else "(no reply)"

        # Post-loop hallucination guard: if the model produced $-figures
        # despite one or more denied tool calls, replace the reply with an
        # honest "I don't have access" message. This is deterministic — no
        # LLM in the loop, so it can't be talked around.
        reply, guard_triggered, guard_reason = _apply_hallucination_guard(
            reply, tool_calls, profile, account_id,
        )

        # Persist the final assistant turn in the history so follow-ups see
        # the OVERRIDDEN reply, not the model's original fabrication.
        if guard_triggered:
            # Replace the text block in the assistant message before saving
            # to history, so subsequent turns don't see the fabricated one.
            content = [
                {"type": "text", "text": reply}
                if b.get("type") == "text" else b
                for b in content
            ]
            if not any(b.get("type") == "text" for b in content):
                content.append({"type": "text", "text": reply})
        messages.append({"role": "assistant", "content": content})
        return ChatTurn(
            reply=reply,
            tool_calls=tool_calls,
            updated_history=messages,
            model_id=model_id,
            guard_triggered=guard_triggered,
            guard_reason=guard_reason,
        )

    return ChatTurn(
        reply="",
        tool_calls=tool_calls,
        updated_history=messages,
        error=f"exceeded {MAX_TOOL_TURNS_PER_QUESTION} tool turns without "
              "a final reply",
        model_id=model_id,
    )
