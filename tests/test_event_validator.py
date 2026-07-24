"""Tests for forecast event validation helpers."""
from __future__ import annotations

from datetime import date

import pytest

from src.ai_agent.chat_agent import ToolCall
from src.ai_agent.event_validator import (
    EventValidationResult,
    _extract_json,
    _parse_validation,
    draft_from_user_input,
)
from src.forecast.events import Effect


def test_extract_json_plain_object():
    parsed = _extract_json('{"accepted": true, "standardized_name": "Org onboarding"}')
    assert parsed["accepted"] is True
    assert parsed["standardized_name"] == "Org onboarding"


def test_extract_json_fenced():
    text = """```json
{"accepted": false, "rationale": "no evidence"}
```"""
    parsed = _extract_json(text)
    assert parsed["accepted"] is False


def test_extract_json_unescaped_control_chars_in_strings():
    text = """{
  "accepted": true,
  "standardized_name": "TPRI region onboarding",
  "validation_summary": "Checked runbook in vendor-vigilance.
Found eu-west-2 stack templates and CDK config."
}"""
    parsed = _extract_json(text)
    assert parsed["accepted"] is True
    assert "runbook" in parsed["validation_summary"]


def test_draft_from_user_input_minimal():
    draft = draft_from_user_input(
        description="org onboarding in Q3",
        expected_start=date(2026, 8, 1),
        forecast_horizon_start=date(2026, 7, 25),
    )
    assert draft["description"] == "org onboarding in Q3"
    assert draft["user_expected_start_hint"] == "2026-08-01"
    assert "effect" not in draft


def test_draft_from_user_input_includes_repos():
    draft = draft_from_user_input(
        description="org onboarding",
        github_repos=["DiligentCorp/platform", "DiligentCorp/api"],
    )
    assert draft["github_repos_in_scope"] == [
        "DiligentCorp/platform",
        "DiligentCorp/api",
    ]


def test_parse_validation_accepted_step():
    draft = {"description": "new region"}
    parsed = {
        "accepted": True,
        "standardized_name": "EU region launch",
        "effect": "step",
        "category": "demand",
        "confidence": 82,
        "start_date": "2026-08-01",
        "end_date": None,
        "ramp_days": 0,
        "amount_daily_usd": 450.0,
        "multiplier_pct": None,
        "estimation_basis": "cost_explorer",
        "validation_summary": (
            "via cost_by_service: RDS $225/day and EC2 $225/day in eu-central-1."
        ),
    }
    tool_calls = [
        ToolCall(name="cost_by_service", input={}, output_summary="RDS+EC2"),
    ]
    result = _parse_validation(parsed, draft, tool_calls)
    assert result.accepted is True
    assert result.effect == "step"
    assert result.amount_daily == pytest.approx(450.0)
    ev = result.to_cost_event()
    assert ev is not None
    assert ev.name == "EU region launch"
    assert ev.effect is Effect.STEP


def test_parse_validation_rejects_ungrounded_typical():
    draft = {"description": "two clients onboarding"}
    parsed = {
        "accepted": True,
        "standardized_name": "Client onboarding",
        "effect": "step",
        "category": "demand",
        "confidence": 80,
        "start_date": "2026-08-08",
        "end_date": None,
        "ramp_days": 0,
        "amount_daily_usd": 1000.0,
        "multiplier_pct": None,
        "estimation_basis": "unknown",
        "validation_summary": (
            "Estimated typical client onboarding costs for this platform."
        ),
    }
    tool_calls = [
        ToolCall(name="github_search_code", input={}, output_summary="runbook"),
    ]
    result = _parse_validation(parsed, draft, tool_calls)
    assert result.accepted is False
    assert result.error
    assert "quantified" in result.error.lower()


def test_parse_validation_rejects_ungrounded_summary_even_with_basis():
    draft = {"description": "client onboarding"}
    parsed = {
        "accepted": True,
        "standardized_name": "Client onboarding",
        "effect": "step",
        "category": "demand",
        "confidence": 80,
        "start_date": "2026-08-08",
        "amount_daily_usd": 1000.0,
        "estimation_basis": "iac_sizing",
        "validation_summary": "No direct cost data; used a heuristic estimate.",
    }
    tool_calls = [
        ToolCall(name="github_get_file", input={}, output_summary="cdk"),
    ]
    result = _parse_validation(parsed, draft, tool_calls)
    assert result.accepted is False
    assert result.error
    assert "ground" in result.error.lower()


def test_parse_validation_rejects_without_tool_calls():
    draft = {"description": "new region"}
    parsed = {
        "accepted": True,
        "standardized_name": "EU region launch",
        "effect": "step",
        "category": "demand",
        "confidence": 82,
        "start_date": "2026-08-01",
        "amount_daily_usd": 450.0,
        "estimation_basis": "cost_explorer",
        "validation_summary": "via cost_by_service: RDS in eu-central-1.",
    }
    result = _parse_validation(parsed, draft, [])
    assert result.accepted is False
    assert "tool" in result.error.lower()


def test_parse_validation_handles_null_fields_on_reject():
    draft = {"description": "client onboarding"}
    parsed = {
        "accepted": False,
        "standardized_name": "Client onboarding",
        "effect": None,
        "category": None,
        "confidence": None,
        "start_date": None,
        "amount_daily_usd": None,
        "multiplier_pct": None,
        "estimation_basis": "unknown",
        "validation_summary": "No Cost Explorer signal for onboarding yet.",
    }
    result = _parse_validation(parsed, draft, [])
    assert result.accepted is False
    assert result.confidence == 0.0
    assert result.amount_daily is None


def test_parse_validation_rejects_incomplete_accept():
    draft = {"description": "vague thing"}
    parsed = {
        "accepted": True,
        "standardized_name": "Vague",
        "effect": "step",
        "category": "demand",
        "confidence": 50,
        "start_date": "2026-08-01",
        "amount_daily_usd": None,
    }
    result = _parse_validation(parsed, draft, [])
    assert result.accepted is False
    assert result.error


def test_to_cost_event_multiplier():
    result = EventValidationResult(
        accepted=True,
        standardized_name="Traffic surge",
        effect="multiplier",
        category="demand",
        confidence=70,
        start_date=date(2026, 8, 1),
        end_date=None,
        ramp_days=0,
        amount_daily=None,
        multiplier_pct=15.0,
        note="Checked CloudWatch Lambda invocations.",
    )
    ev = result.to_cost_event()
    assert ev is not None
    assert ev.multiplier_pct == pytest.approx(15.0)
