"""Validate manual forecast events using the same AWS + GitHub agent as chat."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from src.ai_agent.chat_agent import DEFAULT_MODEL, ToolCall, run_agent_with_tools
from src.forecast.events import CATEGORIES, CostEvent, Effect

_CATEGORY_KEYS = list(CATEGORIES.keys())
_EFFECT_VALUES = [e.value for e in Effect]

_ALLOWED_ESTIMATION_BASES = frozenset({
    "cost_explorer",
    "cloudwatch",
    "resource_inventory",
    "iac_sizing",
})

_UNGROUNDED_SUMMARY_MARKERS = (
    "typical client",
    "typical onboarding",
    "typical cost",
    "industry standard",
    "no direct cost data",
    "no direct cost",
    "heuristic",
    "best guess",
    "rough estimate",
    "cannot quantify",
    "could not quantify",
    "unable to quantify",
    "cannot be quantified",
    "could not be quantified",
    "no measurable",
    "without evidence",
    "from thin air",
)

_VALIDATION_SYSTEM = f"""You validate and structure future cost events for a \
FinOps forecast ledger. The user describes a planned change in plain language. \
You MUST ground your answer in real data by calling read-only AWS and GitHub \
tools — the same tools Ask CostSense uses (Cost Explorer, CloudWatch, resource \
inventory, Compute Optimizer, GitHub repos/files/PRs, etc.).

Your job:
1. Understand what the event claims will happen and when.
2. Use tools to check whether the claim is plausible for this AWS account \
and any repos or services implied by the description. When \
`github_repos_in_scope` is provided in the user message, read IaC, config, \
and recent changes from those repos FIRST (github_list_files, \
github_get_file, github_search_code) before doing a global repo search.
3. Standardize the event name: fix spelling, capitalization, and wording so \
it reads clearly in a finance review. Do NOT map to a fixed taxonomy of event \
types — preserve the user's intent.
4. Choose the forecast shape, category, dates, confidence, and $ impact \
from evidence — the user does NOT supply these; you must infer them.

Shape guidance (pick one):
- step: sudden ongoing $/day change from a start date
- ramp: change that phases in over ramp_days
- pulse: temporary bump between start_date and end_date
- multiplier: % change to baseline spend (multiplier_pct, not $/day)
- cliff: one-time step like step (use step semantics)

$ IMPACT RULES (same bar as PR Predictor — do NOT invent rates):
- NEVER fabricate $/day or % from "typical", "industry standard", or general \
knowledge when tools return no usable signal.
- Accept (accepted=true) ONLY when amount_daily_usd or multiplier_pct traces \
to a specific tool result you actually called. Valid bases:
  * cost_explorer — measured spend change or service-level totals from CE
  * cloudwatch — metrics-derived sizing (invocations, bytes, CPU, etc.)
  * resource_inventory — counted resources × unit cost from AWS inventory
  * iac_sizing — explicit math from IaC/config files you read via GitHub \
    (cite file paths and the arithmetic)
- If you cannot quantify impact from tool data, set accepted=false, \
estimation_basis="unknown", amount_daily_usd=null, multiplier_pct=null, \
and explain what was checked and what detail is still needed.
- Do NOT accept an event just to be helpful. An unquantified future change \
belongs in the ledger only after evidence exists.
- In validation_summary, cite WHICH tool(s) produced each number and show \
the math (e.g. "via cost_by_service: RDS $X/day in eu-west-2 × 2 regions").

Reject (accepted=false) when:
- The description is too vague to validate after tool calls.
- Tool evidence contradicts the claim.
- You could not run meaningful checks.
- The change is plausible but $/day or % cannot be grounded in tool output.

Accept (accepted=true) only when you can return a complete structured event \
WITH a defensible estimation_basis.

Allowed category keys: {", ".join(_CATEGORY_KEYS)}.
Allowed effect shapes: {", ".join(_EFFECT_VALUES)}.
Allowed estimation_basis values: {", ".join(sorted(_ALLOWED_ESTIMATION_BASES))}, \
unknown (reject only — never accept with unknown).

After tool calls, reply with ONLY a JSON object (no markdown fences):
{{
  "accepted": boolean,
  "standardized_name": string,
  "effect": one of the effect shapes above,
  "category": one of the category keys above,
  "confidence": number 0-100,
  "start_date": "YYYY-MM-DD or null",
  "end_date": "YYYY-MM-DD or null (required for pulse)",
  "ramp_days": integer >= 0 (required > 0 for ramp),
  "amount_daily_usd": number or null (required for step/ramp/pulse/cliff),
  "multiplier_pct": number or null (required for multiplier),
  "estimation_basis": one of the allowed values above,
  "evidence_tools": ["tool names you used to size the impact"],
  "rationale": string,
  "validation_summary": string
}}

Rules:
- amount_daily_usd: signed $/day at full strength (negative = saving).
- multiplier_pct: signed % change to baseline (e.g. 25 = +25% spend).
- confidence: probability-weighting for the forecast; keep low when sizing is \
indirect.
- validation_summary: what you checked, which tools, and the sizing math."""


@dataclass
class EventValidationResult:
    accepted: bool
    standardized_name: str
    effect: str
    category: str
    confidence: float
    start_date: date | None
    end_date: date | None
    ramp_days: int
    amount_daily: float | None
    multiplier_pct: float | None
    note: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    error: str | None = None

    def to_cost_event(self) -> CostEvent | None:
        if not self.accepted or self.error or not self.start_date:
            return None
        try:
            effect = Effect(self.effect)
        except ValueError:
            return None
        if effect is Effect.MULTIPLIER:
            if self.multiplier_pct is None:
                return None
        elif self.amount_daily is None:
            return None
        if effect is Effect.PULSE and self.end_date is None:
            return None
        return CostEvent(
            name=self.standardized_name,
            start_date=self.start_date,
            effect=effect,
            category=self.category,
            amount_daily=float(self.amount_daily or 0.0),
            end_date=self.end_date if effect is Effect.PULSE else None,
            ramp_days=self.ramp_days if effect is Effect.RAMP else 0,
            multiplier_pct=float(self.multiplier_pct or 0.0),
            confidence=self.confidence,
            source="manual",
            note=self.note,
        )


def _loads_json_object(raw: str) -> dict[str, Any]:
    """Parse JSON from the model; allow unescaped control chars in strings."""
    data = json.loads(raw, strict=False)
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    return data


def _extract_json(text: str) -> dict[str, Any]:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        return _loads_json_object(t.strip())
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    if start >= 0:
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
                    return _loads_json_object(t[start:i + 1])
    raise ValueError("no JSON object in model reply")


def _parse_date(raw: Any) -> date | None:
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


def _optional_float(raw: Any, default: float | None = None) -> float | None:
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _optional_int(raw: Any, default: int = 0) -> int:
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _failed_result(
    draft: dict[str, Any],
    *,
    error: str,
    tool_calls: list[ToolCall] | None = None,
) -> EventValidationResult:
    return EventValidationResult(
        accepted=False,
        standardized_name=str(draft.get("description", "")),
        effect=Effect.STEP.value,
        category="demand",
        confidence=0.0,
        start_date=None,
        end_date=None,
        ramp_days=0,
        amount_daily=None,
        multiplier_pct=None,
        note="",
        tool_calls=tool_calls or [],
        error=error,
    )


def draft_from_user_input(
    *,
    description: str,
    expected_start: date | None = None,
    forecast_horizon_start: date | None = None,
    github_repos: list[str] | None = None,
) -> dict[str, Any]:
    """Minimal user input passed to the validator — shape and $ come from AI."""
    payload: dict[str, Any] = {
        "description": description.strip(),
    }
    if expected_start is not None:
        payload["user_expected_start_hint"] = expected_start.isoformat()
    if forecast_horizon_start is not None:
        payload["forecast_horizon_starts"] = forecast_horizon_start.isoformat()
    if github_repos:
        payload["github_repos_in_scope"] = list(github_repos)
    return payload


def _draft_message(draft: dict[str, Any]) -> str:
    return (
        "Validate and structure this forecast event. Infer effect shape, "
        "category, dates, and confidence from AWS/GitHub evidence. "
        "Only set $/day or % when a tool result supports the math — "
        "otherwise reject as unquantified:\n"
        f"{json.dumps(draft, indent=2, default=str)}"
    )


def _summary_looks_ungrounded(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in _UNGROUNDED_SUMMARY_MARKERS)


def _grounding_error(
    parsed: dict[str, Any],
    tool_calls: list[ToolCall],
) -> str | None:
    """Reject accepted events that lack defensible sizing (server-side guard)."""
    basis = str(parsed.get("estimation_basis", "unknown")).strip().lower()
    if basis not in _ALLOWED_ESTIMATION_BASES:
        return (
            "Impact could not be quantified from AWS/GitHub data. "
            "Add more detail (services, regions, scale, or repo paths) and "
            "try again once related spend appears in Cost Explorer."
        )
    if not tool_calls:
        return (
            "No AWS or GitHub checks were run — cannot save a sized event "
            "without tool evidence."
        )
    summary = str(parsed.get("validation_summary", "")).strip()
    rationale = str(parsed.get("rationale", "")).strip()
    if _summary_looks_ungrounded(f"{summary} {rationale}"):
        return (
            "The agent could not ground $/day in measured data. "
            "Include specific services, resource counts, or regions."
        )
    return None


def _reject_ungrounded(
    parsed: dict[str, Any],
    draft: dict[str, Any],
    tool_calls: list[ToolCall],
    *,
    error: str,
) -> EventValidationResult:
    summary = str(parsed.get("validation_summary", "")).strip()
    rationale = str(parsed.get("rationale", "")).strip()
    return EventValidationResult(
        accepted=False,
        standardized_name=str(
            parsed.get("standardized_name") or draft.get("description", "")
        ).strip(),
        effect=str(parsed.get("effect") or Effect.STEP.value),
        category=str(parsed.get("category") or "demand"),
        confidence=_optional_float(parsed.get("confidence"), 0.0) or 0.0,
        start_date=_parse_date(parsed.get("start_date")),
        end_date=_parse_date(parsed.get("end_date")),
        ramp_days=_optional_int(parsed.get("ramp_days"), 0),
        amount_daily=_optional_float(parsed.get("amount_daily_usd")),
        multiplier_pct=_optional_float(parsed.get("multiplier_pct")),
        note=summary or rationale or error,
        tool_calls=tool_calls,
        error=error,
    )


def _parse_validation(
    parsed: dict[str, Any],
    draft: dict[str, Any],
    tool_calls: list[ToolCall],
) -> EventValidationResult:
    if not parsed.get("accepted"):
        summary = str(parsed.get("validation_summary", "")).strip()
        rationale = str(parsed.get("rationale", "")).strip()
        return EventValidationResult(
            accepted=False,
            standardized_name=str(
                parsed.get("standardized_name") or draft.get("description", "")
            ).strip(),
            effect=str(parsed.get("effect") or Effect.STEP.value),
            category=str(parsed.get("category") or "demand"),
            confidence=_optional_float(parsed.get("confidence"), 0.0) or 0.0,
            start_date=_parse_date(parsed.get("start_date")),
            end_date=_parse_date(parsed.get("end_date")),
            ramp_days=_optional_int(parsed.get("ramp_days"), 0),
            amount_daily=_optional_float(parsed.get("amount_daily_usd")),
            multiplier_pct=_optional_float(parsed.get("multiplier_pct")),
            note=summary or rationale,
            tool_calls=tool_calls,
        )

    effect_raw = str(parsed.get("effect", "")).strip().lower()
    if effect_raw not in _EFFECT_VALUES:
        return _failed_result(
            draft,
            error=f"invalid effect from agent: {effect_raw!r}",
            tool_calls=tool_calls,
        )

    category = str(parsed.get("category", "demand"))
    if category not in _CATEGORY_KEYS:
        category = "demand"

    start_date = _parse_date(parsed.get("start_date"))
    if start_date is None:
        return _failed_result(
            draft,
            error="accepted event missing start_date",
            tool_calls=tool_calls,
        )

    end_date = _parse_date(parsed.get("end_date"))
    ramp_days = _optional_int(parsed.get("ramp_days"), 0)
    amount_raw = parsed.get("amount_daily_usd")
    mult_raw = parsed.get("multiplier_pct")

    if effect_raw == Effect.MULTIPLIER.value:
        if mult_raw is None:
            return _failed_result(
                draft,
                error="multiplier event missing multiplier_pct",
                tool_calls=tool_calls,
            )
    elif amount_raw is None:
        return _failed_result(
            draft,
            error="event missing amount_daily_usd",
            tool_calls=tool_calls,
        )

    if effect_raw == Effect.PULSE.value and end_date is None:
        return _failed_result(
            draft,
            error="pulse event missing end_date",
            tool_calls=tool_calls,
        )
    if effect_raw == Effect.RAMP.value and ramp_days <= 0:
        return _failed_result(
            draft,
            error="ramp event missing ramp_days",
            tool_calls=tool_calls,
        )

    grounding_err = _grounding_error(parsed, tool_calls)
    if grounding_err:
        return _reject_ungrounded(
            parsed, draft, tool_calls, error=grounding_err,
        )

    summary = str(parsed.get("validation_summary", "")).strip()
    rationale = str(parsed.get("rationale", "")).strip()

    return EventValidationResult(
        accepted=True,
        standardized_name=str(
            parsed.get("standardized_name") or draft.get("description", "")
        ).strip(),
        effect=effect_raw,
        category=category,
        confidence=_optional_float(parsed.get("confidence"), 75.0) or 75.0,
        start_date=start_date,
        end_date=end_date,
        ramp_days=ramp_days,
        amount_daily=_optional_float(amount_raw),
        multiplier_pct=_optional_float(mult_raw),
        note=summary or rationale,
        tool_calls=tool_calls,
    )


def validate_forecast_event(
    *,
    profile: str,
    model_id: str | None,
    draft: dict[str, Any],
    max_tool_turns: int = 10,
) -> EventValidationResult:
    """Run the CostSense agent to validate and structure a ledger event."""
    turn = run_agent_with_tools(
        profile,
        model_id or DEFAULT_MODEL,
        system=_VALIDATION_SYSTEM,
        user_msg=_draft_message(draft),
        history=[],
        max_tool_turns=max_tool_turns,
        max_tokens=2500,
        temperature=0.1,
    )
    if turn.error:
        return _failed_result(draft, error=turn.error, tool_calls=turn.tool_calls)
    try:
        parsed = _extract_json(turn.reply)
    except ValueError as e:
        return _failed_result(
            draft,
            error=f"could not parse validation JSON: {e}",
            tool_calls=turn.tool_calls,
        )
    return _parse_validation(parsed, draft, turn.tool_calls)
