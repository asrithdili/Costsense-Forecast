"""Bedrock Claude PR diff analyzer with CloudWatch tool-use.

Claude reads the PR diff and, when a config change needs runtime context to
be priced (e.g. Lambda memory 10240 -> 4096), calls the CloudWatch tool to
fetch actual invocation/duration/CPU stats. It then returns a structured
verdict with a real $ impact instead of $0.

Model choice: Haiku by default (~$0.001 per PR). Sonnet for complex diffs.
Tool loop cap: MAX_TOOL_TURNS keeps runaway costs bounded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

import boto3

from src.pr_scanner.cloudwatch_tool import TOOL_SPEC, get_metric_statistics


DEFAULT_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"
BEDROCK_REGION = "us-west-2"

MAX_DIFF_CHARS = 40_000
MAX_TOOL_TURNS = 4  # Claude may call the tool up to this many times per PR
MAX_TOKENS = 1500


SYSTEM = """You are an AWS FinOps engineer. Given a git diff from a pull \
request, identify every AWS resource change that will meaningfully affect \
daily cost and estimate the daily USD impact.

Ground your estimates in reality:
- REQUIRED: when a change affects a resource whose cost depends on usage \
(Lambda memory or provisioned concurrency, ECS task size, RDS instance \
class, DDB capacity, etc.), you MUST call get_cloudwatch_metric FIRST to \
fetch real usage before pricing. Do not guess invocation rates or \
durations. If the tool returns an error or empty data, then set \
est_daily_delta_usd = 0 and note "no metrics available" in rationale.
- For Lambda memory changes: fetch Invocations (Sum) and Duration \
(Average) for the changed function; compute $ delta as \
`invocations_per_day * duration_seconds * (new_gb - old_gb) * 0.0000166667` \
(the on-demand $ per GB-second).
- For provisioned concurrency changes: cost = (new_pc - old_pc) * $0.000004097 * 86400 seconds.
- For instance-type changes: price the difference between the new and old on-demand rates.
- Ignore purely cosmetic changes (comments, tests, docs, formatting, JSON \
schema tweaks, dependency version bumps that don't add resources).
- If the PR does NOT add, remove, or resize an AWS resource, return \
{"changes": [], "total_daily_delta_usd": 0, "summary": "no cost impact"}. \
This is the correct answer for the vast majority of PRs.
- Do NOT copy numbers from the schema example. Every PR is different — \
compute the number for THIS diff from THIS diff, or return 0.
- If you cannot show your arithmetic (e.g. old GB, new GB, invocations/day \
from CloudWatch), the answer is 0.

When done, return ONLY the final JSON — no prose outside JSON, no code \
fences. The JSON has this shape:
{
  "changes": [
    {"resource_type": "<aws_service_type>",
     "resource_name": "<name from the diff>",
     "action": "add" | "remove" | "modify",
     "instance_hint": "<optional detail like memory:X->Y>",
     "est_daily_delta_usd": <YOUR computed number for this specific PR, NOT the example below>,
     "rationale": "<show the arithmetic that produced the number>"}
  ],
  "total_daily_delta_usd": <sum of the changes>,
  "summary": "one sentence describing what THIS PR does"
}

The following example is FOR SCHEMA REFERENCE ONLY — do NOT copy these \
numbers into your response:
[EXAMPLE-DO-NOT-COPY]
{
  "changes": [{"resource_type": "aws_lambda_function", "resource_name": "example",
   "action": "modify", "instance_hint": "memory:X->Y",
   "est_daily_delta_usd": <computed>, "rationale": "<arithmetic>"}],
  "total_daily_delta_usd": <sum>, "summary": "..."
}
[/EXAMPLE-DO-NOT-COPY]"""


@dataclass
class LlmChange:
    resource_type: str
    resource_name: str
    action: str
    instance_hint: str | None
    est_daily_delta_usd: float
    rationale: str


@dataclass
class LlmVerdict:
    changes: list[LlmChange] = field(default_factory=list)
    total_daily_delta_usd: float = 0.0
    summary: str = ""
    model_id: str = ""
    tool_calls: int = 0
    error: str | None = None


@lru_cache(maxsize=1)
def _client(profile: str | None):
    from src.ai_agent.bedrock_client import make_client
    return make_client(profile, region=BEDROCK_REGION)


def _truncate_diff(diff: str) -> str:
    if len(diff) <= MAX_DIFF_CHARS:
        return diff
    head = diff[: MAX_DIFF_CHARS // 2]
    tail = diff[-MAX_DIFF_CHARS // 2 :]
    return f"{head}\n\n[... {len(diff) - MAX_DIFF_CHARS} chars truncated ...]\n\n{tail}"


def _extract_json(text: str) -> dict:
    """Find the first balanced JSON object in text. Tolerant of prose
    before/after and fenced code blocks."""
    t = text.strip()
    # strip a leading ```json / ``` fence
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if "```" in t:
            t = t.split("```", 1)[0]
    # scan for the first balanced { ... }
    start = t.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = t.find("{", start + 1)
    raise json.JSONDecodeError("no balanced JSON object found", text, 0)


def _parse_verdict(text: str, model_id: str, tool_calls: int) -> LlmVerdict:
    try:
        parsed = _extract_json(text)
    except json.JSONDecodeError as e:
        return LlmVerdict(
            model_id=model_id, tool_calls=tool_calls,
            error=f"claude did not return valid JSON: {e}. raw={text[:200]}",
        )
    changes = []
    for c in parsed.get("changes", []) or []:
        try:
            changes.append(LlmChange(
                resource_type=str(c.get("resource_type", "unknown")),
                resource_name=str(c.get("resource_name", "")),
                action=str(c.get("action", "modify")),
                instance_hint=(c.get("instance_hint") or None),
                est_daily_delta_usd=float(c.get("est_daily_delta_usd", 0.0) or 0.0),
                rationale=str(c.get("rationale", "")),
            ))
        except (TypeError, ValueError):
            continue
    total = float(parsed.get("total_daily_delta_usd", 0.0) or 0.0)
    if changes and not total:
        total = round(sum(c.est_daily_delta_usd for c in changes), 4)
    return LlmVerdict(
        changes=changes,
        total_daily_delta_usd=round(total, 4),
        summary=str(parsed.get("summary", "")),
        model_id=model_id,
        tool_calls=tool_calls,
    )


def _invoke(client, model_id: str, messages: list) -> dict:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "system": SYSTEM,
        "tools": [TOOL_SPEC],
        "messages": messages,
    }
    resp = client.invoke_model(modelId=model_id, body=json.dumps(body))
    return json.loads(resp["body"].read())


def analyze_pr_diff(
    diff: str,
    pr_title: str = "",
    profile: str | None = None,
    model_id: str = DEFAULT_MODEL,
    max_tokens: int = MAX_TOKENS,   # kept for backwards compat; unused in loop
) -> LlmVerdict:
    """Send diff to Claude with a CloudWatch tool; run the tool-use loop."""
    if not diff.strip():
        return LlmVerdict(model_id=model_id, summary="empty diff")

    client = _client(profile)
    truncated = _truncate_diff(diff)
    user_first = (
        f"PR title: {pr_title}\n\n"
        f"Diff:\n```diff\n{truncated}\n```\n\n"
        "Analyze this PR for AWS cost impact. Use get_cloudwatch_metric "
        "when needed to size usage-dependent changes. Return the final "
        "JSON verdict when done."
    )
    messages: list = [{"role": "user", "content": user_first}]

    tool_calls = 0
    for turn in range(MAX_TOOL_TURNS + 1):
        try:
            payload = _invoke(client, model_id, messages)
        except Exception as e:  # noqa: BLE001
            return LlmVerdict(model_id=model_id, tool_calls=tool_calls,
                              error=f"bedrock invoke failed: {e}")

        stop_reason = payload.get("stop_reason", "")
        content_blocks = payload.get("content", [])
        tool_use_blocks = [b for b in content_blocks if b.get("type") == "tool_use"]

        if tool_use_blocks:
            # Always answer every tool_use block, whatever stop_reason says.
            messages.append({"role": "assistant", "content": content_blocks})
            tool_results = []
            for blk in tool_use_blocks:
                tool_calls += 1
                inp = blk.get("input", {}) or {}
                try:
                    result = get_metric_statistics(
                        namespace=inp.get("namespace", ""),
                        metric_name=inp.get("metric_name", ""),
                        dimensions=inp.get("dimensions", []),
                        statistic=inp.get("statistic", "Sum"),
                        days=min(int(inp.get("days", 7) or 7), 30),
                        region=inp.get("region", "us-east-1"),
                        profile=profile,
                    )
                except Exception as e:  # noqa: BLE001
                    result = {"error": f"tool failed: {e}"}
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": blk["id"],
                    "content": json.dumps(result)[:6000],
                })
            messages.append({"role": "user", "content": tool_results})
            continue

        # No tool_use blocks — Claude has (or should have) produced final text
        text_block = next(
            (blk for blk in content_blocks if blk.get("type") == "text"), None
        )
        if text_block is None:
            return LlmVerdict(
                model_id=model_id, tool_calls=tool_calls,
                error=f"no text block; stop={stop_reason}",
            )
        text = text_block.get("text", "")
        verdict = _parse_verdict(text, model_id, tool_calls)
        # If Claude produced prose without JSON (typical of Sonnet's first
        # exploratory turn), nudge it once to commit to JSON.
        if verdict.error and turn < MAX_TOOL_TURNS:
            messages.append({"role": "assistant", "content": content_blocks})
            messages.append({"role": "user", "content": (
                "Return the final JSON verdict now. Do not include any text "
                "outside JSON. If you need runtime data, call "
                "get_cloudwatch_metric. Otherwise output the verdict JSON."
            )})
            continue
        return verdict

    return LlmVerdict(model_id=model_id, tool_calls=tool_calls,
                      error=f"tool-use loop exceeded {MAX_TOOL_TURNS} turns")
