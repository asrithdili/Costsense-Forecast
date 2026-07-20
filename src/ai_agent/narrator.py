"""Cost-narration agent.

Given a history series, a forecast, and the PR-scan output, ask Claude to
explain the peaks and troughs in words, and predict the future direction with
concrete reasons. The agent also has access to the read-only AWS tools
(CloudTrail, CloudWatch, cost-by-service) so it can look up what actually
changed in the account on the days it needs to explain.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta

import boto3
import pandas as pd

from src.ai_agent.aws_tools import call_tool, tool_specs


BEDROCK_REGION = "us-west-2"
DEFAULT_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"
MAX_TOOL_TURNS = 8
MAX_TOKENS = 3500


@dataclass
class NarrationEvent:
    date: str
    kind: str            # "peak" | "trough" | "step-up" | "step-down"
    amount_usd: float
    change_vs_prior: float
    explanation: str = ""
    confidence: str = "medium"


@dataclass
class Narration:
    past_events: list[NarrationEvent] = field(default_factory=list)
    future_outlook: str = ""
    future_direction: str = "unknown"     # "up" | "down" | "flat"
    future_est_daily_usd: float = 0.0
    key_drivers: list[str] = field(default_factory=list)
    tool_calls: int = 0
    model_id: str = ""
    raw_text: str = ""
    error: str | None = None


def _detect_events(hist_df: pd.DataFrame, top_k: int = 5) -> list[dict]:
    """Find the biggest daily jumps + level shifts in history. This is what
    the agent will be asked to explain."""
    if hist_df.empty or len(hist_df) < 3:
        return []
    df = hist_df.sort_values("day").reset_index(drop=True).copy()
    df["prev"] = df["actual_usd"].shift(1)
    df["delta"] = df["actual_usd"] - df["prev"]
    df["pct"] = df["delta"] / df["prev"].replace(0, float("nan"))
    df = df.dropna(subset=["delta"])
    df["abs_delta"] = df["delta"].abs()
    top = df.sort_values("abs_delta", ascending=False).head(top_k)
    events = []
    for _, r in top.iterrows():
        events.append({
            "date": r["day"],
            "amount_usd": round(float(r["actual_usd"]), 2),
            "prev_usd": round(float(r["prev"]), 2),
            "delta_usd": round(float(r["delta"]), 2),
            "pct_change": round(float(r["pct"]) * 100, 1)
            if pd.notna(r["pct"]) else None,
            "kind": "peak" if r["delta"] > 0 else "trough",
        })
    return events


SYSTEM = """You are an AWS FinOps analyst. You will be given:
  1. A recent daily-cost series for an AWS account (or a combined series \
across coupled accounts).
  2. The top anomalies (biggest daily jumps and drops).
  3. The list of PRs merged during that window with their estimated cost \
impact.
  4. A 7-day forecast.

You have READ-ONLY tools to inspect the account: CloudTrail events on any \
day, CloudWatch metrics for any resource, resource inventory, and cost-by-\
service breakdowns.

REQUIRED WORKFLOW for each past anomaly:
  1. Call cost_by_service to see which SERVICES moved on that day. \
Compare the anomaly-day service breakdown vs the day before. This tells \
you WHICH service caused the change.
  2. Call cloudtrail_lookup with a narrow date range around the anomaly \
day. Look for events like DisableOrganizationAdminAccount, StopDBInstance, \
DeleteFunction, DeleteNatGateway, PutBucketLifecycle, EnableGuardDuty.
  3. Only after seeing tool output, write the explanation.

Do NOT return "no clear signal" or "workload fluctuation" without first \
making at least one tool call for that day. Guessing is worse than \
admitting uncertainty AFTER checking.

Return ONLY a JSON object (no prose outside JSON, no code fences):
{
  "past_events": [
    {"date": "2026-07-10", "kind": "trough",
     "amount_usd": 137.4, "change_vs_prior": -200.1,
     "explanation": "GuardDuty disabled on 2026-07-09 (CloudTrail: \
DisableOrganizationAdminAccount by user X), removing ~$150/day. Lambda \
right-sizing from PR #844 accounted for another $10/day. Remaining $40 \
is workload-driven.",
     "confidence": "high"}
  ],
  "future_outlook": "one paragraph combining the current level, momentum, \
open PRs, and any pending scheduled events",
  "future_direction": "up" | "down" | "flat",
  "future_est_daily_usd": <predicted average daily cost for the next 7 days>,
  "key_drivers": ["short bullet 1", "short bullet 2", ...]
}

Rules:
- Every past_event MUST have a concrete explanation grounded in either a \
tool call or a PR from the input. If you cannot ground it, set \
confidence="low" and say "no signal found".
- future_direction reflects where the level line is heading, not the \
noise. Base it on the last 7 days' trajectory + any PRs whose delta hasn't \
been absorbed yet.
- Prefer specific over vague: "$150/day GuardDuty toggle" beats \
"security service change"."""


def narrate(
    hist_df: pd.DataFrame,
    forecast_rows: list[dict],
    pr_impacts: list[dict],
    profile: str | None = None,
    model_id: str = DEFAULT_MODEL,
    account_context: str = "",
) -> Narration:
    """Ask Claude to explain the peaks + troughs + future outlook.

    hist_df: DataFrame with columns [day (iso str), actual_usd].
    forecast_rows: list of {"target_date", "adjusted_usd", ...} dicts.
    pr_impacts: list of the pr_scan.impacts dicts."""
    events_hint = _detect_events(hist_df, top_k=6)
    recent_history = (
        hist_df.tail(45).to_dict(orient="records") if not hist_df.empty else []
    )
    pr_summary = [
        {
            "repo": p.get("repo"),
            "pr_number": p.get("pr_number"),
            "title": p.get("pr_title"),
            "merged_at": p.get("merged_at", "")[:10],
            "est_daily_delta_usd": p.get("est_daily_delta_usd", 0.0),
            "llm_summary": p.get("llm_summary", ""),
        }
        for p in pr_impacts
        if p.get("est_daily_delta_usd", 0)  # skip zero-impact PRs
    ]

    user_msg = (
        (f"Context: {account_context}\n\n" if account_context else "")
        + "Recent history (last 45 days, day + USD):\n"
        + json.dumps(recent_history, default=str)
        + "\n\nTop anomalies to explain:\n"
        + json.dumps(events_hint, default=str)
        + "\n\nPRs merged in this window (only PRs with non-zero cost impact):\n"
        + json.dumps(pr_summary, default=str)
        + "\n\n7-day forecast:\n"
        + json.dumps(forecast_rows, default=str)
        + "\n\nProduce the JSON narration now. Use tools when needed."
    )

    from src.ai_agent.bedrock_client import make_client
    client = make_client(profile, region=BEDROCK_REGION)
    messages: list = [{"role": "user", "content": user_msg}]
    tool_calls = 0

    for _ in range(MAX_TOOL_TURNS + 1):
        try:
            resp = client.invoke_model(
                modelId=model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.0,
                    "system": SYSTEM,
                    "tools": tool_specs(),
                    "messages": messages,
                }),
            )
            payload = json.loads(resp["body"].read())
        except Exception as e:  # noqa: BLE001
            return Narration(model_id=model_id, tool_calls=tool_calls,
                             error=f"bedrock invoke failed: {e}")

        content = payload.get("content", [])
        tool_use_blocks = [b for b in content if b.get("type") == "tool_use"]

        if tool_use_blocks:
            messages.append({"role": "assistant", "content": content})
            results = []
            for blk in tool_use_blocks:
                tool_calls += 1
                result = call_tool(blk["name"], blk.get("input") or {}, profile)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": blk["id"],
                    "content": json.dumps(result, default=str)[:8000],
                })
            messages.append({"role": "user", "content": results})
            continue

        text_block = next((b for b in content if b.get("type") == "text"), None)
        if not text_block:
            return Narration(model_id=model_id, tool_calls=tool_calls,
                             error="no text in final response",
                             raw_text=json.dumps(content)[:500])
        return _parse(text_block.get("text", ""), model_id, tool_calls)

    return Narration(model_id=model_id, tool_calls=tool_calls,
                     error=f"exceeded {MAX_TOOL_TURNS} tool turns")


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
    raise json.JSONDecodeError("no JSON object found", text, 0)


def _parse(text: str, model_id: str, tool_calls: int) -> Narration:
    try:
        parsed = _extract_json(text)
    except json.JSONDecodeError as e:
        return Narration(model_id=model_id, tool_calls=tool_calls,
                         error=f"no JSON in response: {e}",
                         raw_text=text[:500])
    events = []
    for e in (parsed.get("past_events") or []):
        events.append(NarrationEvent(
            date=str(e.get("date", "")),
            kind=str(e.get("kind", "")),
            amount_usd=float(e.get("amount_usd", 0.0) or 0.0),
            change_vs_prior=float(e.get("change_vs_prior", 0.0) or 0.0),
            explanation=str(e.get("explanation", "")),
            confidence=str(e.get("confidence", "medium")),
        ))
    return Narration(
        past_events=events,
        future_outlook=str(parsed.get("future_outlook", "")),
        future_direction=str(parsed.get("future_direction", "unknown")),
        future_est_daily_usd=float(parsed.get("future_est_daily_usd", 0.0) or 0.0),
        key_drivers=[str(x) for x in (parsed.get("key_drivers") or [])],
        tool_calls=tool_calls,
        model_id=model_id,
        raw_text=text,
    )
