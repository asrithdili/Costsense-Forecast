"""Anomaly / recommendation agent.

Takes pre-computed repo sweep + AWS sweep summaries as INPUT (no tool calls
needed for the base analysis — Claude just reasons over the facts) and
produces a ranked list of concrete cost-cutting actions.

Optionally the agent can still call AWS tools for follow-up drilling
(get_cloudwatch_metric for a specific resource, cloudtrail_lookup for a
suspicious day).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import boto3

from src.ai_agent.aws_tools import call_tool, tool_specs
from src.ai_agent.aws_tools_broad import all_broad_specs


BEDROCK_REGION = "us-west-2"
DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"
MAX_TOOL_TURNS = 12
MAX_TOKENS = 16000
# Cap the number of actions the model can return so it doesn't blow the
# token budget and cause the final JSON to truncate mid-string.
MAX_ACTIONS = 8
# Each code snippet cap — enforced in prompt so approaches stay small.
MAX_CODE_CHARS = 500


SYSTEM = """You are a senior AWS FinOps analyst. You receive two pre-computed \
sweeps — one from the user's GitHub repos (recent PRs, IaC files, scheduled \
rules) and one from their AWS account (idle resources, rightsizing recs, top \
services, budgets). Your job:

  Produce a RANKED LIST of concrete cost issues. Each has three narrative \
fields (Issue, Reason, Recommendation) PLUS 2-3 concrete fix approaches with \
optional code snippets.

Rules:
  1. Ground every entry in the provided sweep data. Cite which sweep field \
you used (e.g. "compute_optimizer_lambda.top[0]" or "idle_ebs_volumes.sample[2]").
  2. Rank by est_daily_savings_usd, largest first.
  3. Use PLAIN TEXT dollars — write "~2375" or "roughly $2,375" but NEVER \
wrap numbers in $...$ or use LaTeX. The UI renders these strings as \
markdown and $..$ triggers math mode.
  4. Every action has these fields:
     - issue: ONE short sentence naming the problem + the specific resource.
     - reason: ONE short sentence explaining WHY it costs money.
     - recommendation: ONE short sentence summarizing the primary fix.
     - approaches: array of 2-3 concrete alternative fixes. Each is:
         { "title": short verb phrase (e.g. "Enable S3 Intelligent-Tiering"),
           "description": 1-2 sentence description of the trade-off,
           "code": optional code snippet the user could copy-paste (Terraform,
                   CDK TypeScript, boto3, or aws-cli). Omit or set to null \
if code doesn't help.
           "language": "terraform" | "typescript" | "python" | "bash" | \
"yaml" | "json" — required when `code` is present. }
     - est_daily_savings_usd (positive number)
     - confidence ("high" | "medium" | "low")
     - category ("idle" | "oversized" | "log-inefficiency" | \
"missing-lifecycle" | "risky-upcoming-pr" | "trending-up")
     - source (sweep field path)
  5. REQUIRED: every action MUST include an `approaches` array with 2 or 3 \
entries — never fewer, never zero, never omit the field. Prefer approaches \
that trade off differently (e.g. delete-now vs migrate-to-cheaper-tier; \
console-click vs IaC change; conservative vs aggressive). If you truly \
cannot think of alternatives, at minimum split "do it via console" and \
"do it via IaC/CLI" into two approaches.
  6. If the fix requires code, put it under `approaches[i].code`. Otherwise \
omit code and just describe the console/CLI steps in `description`.
  7. Keep issue/reason/recommendation ONE short sentence each (<25 words). \
approaches[i].description up to 2 SHORT sentences. Code snippets MUST be \
under 500 characters — no full CDK stacks, just the essential 5-15 lines. \
Never repeat the recommendation verbatim in a description.
  8. Return AT MOST 8 actions total. Prioritize highest $/day savings. \
Terse beats verbose — the JSON must fit in the response budget without \
truncation.
  9. If the sweeps don't have enough info, you may call an AWS tool to drill \
deeper (up to 4 calls).
  10. DO NOT make up numbers. If a saving amount isn't in the sweep, set \
confidence to "low".

Return ONLY JSON (no prose outside JSON, no code fences):
{
  "summary": "one-sentence high-level readout, PLAIN TEXT dollar amounts",
  "total_daily_savings_usd": <number>,
  "actions": [
    {"issue": "...", "reason": "...", "recommendation": "...",
     "approaches": [
       {"title": "...", "description": "...",
        "code": "...", "language": "terraform"},
       {"title": "...", "description": "..."}
     ],
     "est_daily_savings_usd": <number>, "confidence": "high|medium|low",
     "category": "...", "source": "sweep.field.path"}
  ]
}"""


@dataclass
class Approach:
    title: str = ""
    description: str = ""
    code: str = ""
    language: str = ""


@dataclass
class Action:
    issue: str = ""
    reason: str = ""
    recommendation: str = ""
    approaches: list[Approach] = field(default_factory=list)
    category: str = ""
    est_daily_savings_usd: float = 0.0
    confidence: str = "medium"
    source: str = ""


@dataclass
class AnomalyReport:
    summary: str = ""
    total_daily_savings_usd: float = 0.0
    actions: list[Action] = field(default_factory=list)
    tool_calls: int = 0
    model_id: str = ""
    error: str | None = None
    raw_text: str = ""


def _extract_json(text: str) -> dict:
    """Extract the outermost balanced JSON object that looks like our
    envelope (has an `actions` array). Handles prose-before-JSON, fenced
    ```json blocks, and truncated responses.

    Strategy: try to parse the entire text first (works when the model
    returned pure JSON). If that fails, walk `{` positions and prefer the
    object with the largest size that (a) parses AND (b) contains an
    `actions` key — that avoids picking up a nested Action object when
    the outer envelope is truncated.
    """
    t = text.strip()

    # Strip leading prose up to the first ```json fence, or general ``` fence
    for fence in ("```json", "```"):
        idx = t.find(fence)
        if idx >= 0:
            t = t[idx + len(fence):]
            break
    # Strip trailing fence
    if "```" in t:
        t = t.split("```", 1)[0]

    # Fast path: entire string is one JSON object
    try:
        return json.loads(t.strip())
    except json.JSONDecodeError:
        pass

    # Walk all `{` positions, collect every balanced object that parses,
    # then prefer the one with an `actions` array (or the largest).
    candidates: list[tuple[int, dict]] = []
    start = t.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(t[start:i+1])
                        if isinstance(obj, dict):
                            candidates.append((i - start, obj))
                    except json.JSONDecodeError:
                        pass
                    break
        start = t.find("{", start + 1)

    # Prefer objects that look like our envelope
    with_actions = [c for c in candidates if "actions" in c[1]]
    if with_actions:
        return max(with_actions, key=lambda c: c[0])[1]

    # Truncation recovery: the outer envelope didn't close, so we scan for
    # individual Action objects (those with issue+reason+recommendation)
    # and rebuild a synthetic envelope from them.
    action_shaped = [
        c[1] for c in candidates
        if isinstance(c[1], dict)
        and "issue" in c[1] and "reason" in c[1] and "recommendation" in c[1]
    ]
    if action_shaped:
        # Try to pull the summary from the raw text as a bonus
        summary = ""
        s_idx = t.find('"summary"')
        if s_idx >= 0:
            colon = t.find(":", s_idx)
            q1 = t.find('"', colon + 1)
            q2 = t.find('"', q1 + 1)
            if q1 >= 0 and q2 > q1:
                summary = t[q1 + 1: q2]
        return {
            "summary": summary,
            "total_daily_savings_usd": sum(
                float(a.get("est_daily_savings_usd", 0) or 0)
                for a in action_shaped
            ),
            "actions": action_shaped,
        }

    if not candidates:
        raise json.JSONDecodeError("no JSON object", text, 0)
    return max(candidates, key=lambda c: c[0])[1]


def _parse(text: str, model_id: str, tool_calls: int) -> AnomalyReport:
    try:
        parsed = _extract_json(text)
    except json.JSONDecodeError as e:
        return AnomalyReport(model_id=model_id, tool_calls=tool_calls,
                             error=f"no JSON in response: {e}",
                             raw_text=text[:500])
    actions = []
    for a in (parsed.get("actions") or []):
        try:
            issue = a.get("issue") or a.get("resource") or ""
            reason = a.get("reason") or a.get("rationale") or ""
            recommendation = a.get("recommendation") or a.get("action") or ""
            approaches: list[Approach] = []
            for ap in (a.get("approaches") or []):
                if not isinstance(ap, dict):
                    continue
                approaches.append(Approach(
                    title=str(ap.get("title", "")),
                    description=str(ap.get("description", "")),
                    code=str(ap.get("code") or ""),
                    language=str(ap.get("language") or ""),
                ))
            actions.append(Action(
                issue=str(issue),
                reason=str(reason),
                recommendation=str(recommendation),
                approaches=approaches,
                category=str(a.get("category", "")),
                est_daily_savings_usd=float(
                    a.get("est_daily_savings_usd", 0.0) or 0.0
                ),
                confidence=str(a.get("confidence", "medium")),
                source=str(a.get("source", "")),
            ))
        except (TypeError, ValueError):
            continue
    # Sort by savings desc — highest first
    actions.sort(key=lambda a: -a.est_daily_savings_usd)
    return AnomalyReport(
        summary=str(parsed.get("summary", "")),
        total_daily_savings_usd=float(
            parsed.get("total_daily_savings_usd", 0.0) or 0.0
        ),
        actions=actions,
        tool_calls=tool_calls,
        model_id=model_id,
        raw_text=text,
    )


def analyze_anomalies(
    aws_summary: dict,
    repo_summary: dict,
    profile: str | None = None,
    model_id: str = DEFAULT_MODEL,
) -> AnomalyReport:
    from src.ai_agent.bedrock_client import make_client
    client = make_client(profile, region=BEDROCK_REGION)

    user_msg = (
        "AWS SWEEP:\n"
        + json.dumps(aws_summary, default=str)[:10000]
        + "\n\nREPO SWEEP:\n"
        + json.dumps(repo_summary, default=str)[:12000]
        + "\n\nProduce the ranked recommendation JSON now."
    )
    messages: list = [{"role": "user", "content": user_msg}]
    tool_calls = 0
    all_specs = tool_specs() + all_broad_specs()

    for _ in range(MAX_TOOL_TURNS + 1):
        try:
            resp = client.invoke_model(
                modelId=model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.0,
                    "system": SYSTEM,
                    "tools": all_specs,
                    "messages": messages,
                }),
            )
            payload = json.loads(resp["body"].read())
        except Exception as e:  # noqa: BLE001
            return AnomalyReport(model_id=model_id, tool_calls=tool_calls,
                                 error=f"bedrock invoke failed: {e}")

        content = payload.get("content", [])
        tool_use = [b for b in content if b.get("type") == "tool_use"]

        if tool_use:
            messages.append({"role": "assistant", "content": content})
            results = []
            for blk in tool_use:
                tool_calls += 1
                out = call_tool(blk["name"], blk.get("input") or {}, profile)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": blk["id"],
                    "content": json.dumps(out, default=str)[:6000],
                })
            messages.append({"role": "user", "content": results})
            continue

        text_block = next((b for b in content if b.get("type") == "text"), None)
        if not text_block:
            return AnomalyReport(model_id=model_id, tool_calls=tool_calls,
                                 error="no text in final response")
        candidate = _parse(text_block.get("text", ""), model_id, tool_calls)
        # If any action came back without approaches, push back once —
        # the prompt says approaches is REQUIRED, so this is a violation.
        missing_approaches = (
            candidate.error is None
            and candidate.actions
            and any(not a.approaches for a in candidate.actions)
        )
        already_retried = any(
            isinstance(m.get("content"), str)
            and "approaches missing" in m["content"]
            for m in messages if m.get("role") == "user"
        )
        if missing_approaches and not already_retried:
            messages.append({"role": "assistant", "content": content})
            missing_count = sum(1 for a in candidate.actions if not a.approaches)
            messages.append({"role": "user", "content": (
                f"approaches missing on {missing_count} action(s). Per the "
                "system rules, EVERY action MUST include an `approaches` "
                "array with 2-3 entries. Resend the SAME actions, but this "
                "time include the approaches array on every one. If you "
                "can't think of code-based approaches, split into "
                "'console approach' and 'CLI/IaC approach'."
            )})
            continue
        return candidate

    return AnomalyReport(model_id=model_id, tool_calls=tool_calls,
                         error=f"exceeded {MAX_TOOL_TURNS} tool turns")
