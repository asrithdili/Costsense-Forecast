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
MAX_TOOL_TURNS = 8
MAX_TOKENS = 4000


SYSTEM = """You are a senior AWS FinOps analyst. You receive two pre-computed \
sweeps — one from the user's GitHub repos (recent PRs, IaC files, scheduled \
rules) and one from their AWS account (idle resources, rightsizing recs, top \
services, budgets). Your job:

  Produce a RANKED LIST of concrete cost issues, each formatted as EXACTLY \
three fields: Issue, Reason, Recommendation.

Rules:
  1. Ground every entry in the provided sweep data. Cite which sweep field \
you used (e.g. "compute_optimizer_lambda.top[0]" or "idle_ebs_volumes.sample[2]").
  2. Rank by est_daily_savings_usd, largest first.
  3. Every action has EXACTLY these fields — no more, no less:
     - issue: ONE short sentence naming the problem + the specific resource. \
E.g. "3 idle NAT gateways cost ~$96/month". Include the resource id in the \
sentence.
     - reason: ONE short sentence explaining WHY this is happening / WHY it \
costs money. E.g. "Each active NAT gateway is billed at $0.045/hour even \
when idle."
     - recommendation: ONE short sentence with a concrete action. E.g. \
"Delete nat-0abc, nat-0def, nat-0ghi via the VPC console."
     - est_daily_savings_usd (positive number)
     - confidence ("high" | "medium" | "low")
     - category ("idle" | "oversized" | "log-inefficiency" | \
"missing-lifecycle" | "risky-upcoming-pr" | "trending-up")
     - source (sweep field path)
  4. Keep each of issue/reason/recommendation ONE sentence. No paragraphs. \
No lists inside. If you have multiple resources of the same kind, either list \
them in the issue sentence or split into multiple actions.
  5. If the sweeps don't have enough info, you may call one of the AWS tools \
to drill deeper — but keep it to at most 4 targeted calls.
  6. DO NOT make up numbers. If a saving amount isn't in the sweep, set \
confidence to "low".

Return ONLY JSON (no prose outside JSON, no code fences):
{
  "summary": "one-sentence high-level readout",
  "total_daily_savings_usd": <number>,
  "actions": [
    {"issue": "...", "reason": "...", "recommendation": "...",
     "est_daily_savings_usd": <number>, "confidence": "high|medium|low",
     "category": "...", "source": "sweep.field.path"}
  ]
}"""


@dataclass
class Action:
    issue: str = ""
    reason: str = ""
    recommendation: str = ""
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
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if "```" in t:
            t = t.split("```", 1)[0]
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
                        return json.loads(t[start:i+1])
                    except json.JSONDecodeError:
                        break
        start = t.find("{", start + 1)
    raise json.JSONDecodeError("no JSON object", text, 0)


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
            # Prefer new field names; fall back to old ones if the model
            # emitted the legacy schema.
            issue = a.get("issue") or a.get("resource") or ""
            reason = a.get("reason") or a.get("rationale") or ""
            recommendation = a.get("recommendation") or a.get("action") or ""
            actions.append(Action(
                issue=str(issue),
                reason=str(reason),
                recommendation=str(recommendation),
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
        + json.dumps(repo_summary, default=str)[:4000]
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
        return _parse(text_block.get("text", ""), model_id, tool_calls)

    return AnomalyReport(model_id=model_id, tool_calls=tool_calls,
                         error=f"exceeded {MAX_TOOL_TURNS} tool turns")
