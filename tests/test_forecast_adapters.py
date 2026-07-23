"""Tests for PR → forecast event adapters."""
from __future__ import annotations

from datetime import date

from src.ai_agent.agent import AgentVerdict
from src.ai_agent.anomaly_agent import Action
from src.forecast.adapters import (
    event_from_anomaly_action,
    event_from_pr_impact,
    event_from_pr_predictor,
    event_from_priced_open_pr,
    merge_events,
)
from src.forecast.events import CostEvent, Effect
from src.pr_scanner.open_prs import OpenPr, PricedOpenPr
from src.pr_scanner.scan import PrImpact


def _open_pr() -> OpenPr:
    return OpenPr(
        repo="org/repo",
        number=42,
        title="Scale DynamoDB",
        author="dev",
        url="https://github.com/org/repo/pull/42",
        is_draft=False,
        created_at="2026-07-01T00:00:00Z",
        updated_at="2026-07-10T00:00:00Z",
        review_state="APPROVED",
        mergeable="MERGEABLE",
        checks_state="SUCCESS",
    )


def test_event_from_priced_open_pr():
    p = PricedOpenPr(
        open_pr=_open_pr(),
        est_daily_delta_usd=120.0,
        direction="increase",
        llm_summary="Adds provisioned capacity",
        merge_probability=0.8,
        expected_merge_day="2026-08-01",
        expected_daily_delta_usd=96.0,
    )
    ev = event_from_priced_open_pr(p)
    assert ev is not None
    assert ev.amount_daily == 120.0
    assert ev.confidence == 80.0
    assert ev.external_id == "open_pr:org/repo#42"
    assert ev.source == "pr_predictor"


def test_event_from_pr_impact():
    imp = PrImpact(
        repo="org/repo",
        pr_number=7,
        pr_title="Rightsize RDS",
        pr_url="https://github.com/org/repo/pull/7",
        author="dev",
        merged_at="2026-07-15T12:00:00Z",
        est_daily_delta_usd=-50.0,
        analyzer="hybrid",
        llm_summary="Smaller instance class",
    )
    ev = event_from_pr_impact(imp)
    assert ev is not None
    assert ev.start_date == date(2026, 7, 15)
    assert ev.amount_daily == -50.0
    assert ev.category == "optimization"
    assert ev.external_id == "merged_pr:org/repo#7"
    assert ev.source == "costsense_pr"


def test_event_from_anomaly_action_negative_savings():
    action = Action(
        issue="Idle EBS volume vol-abc",
        reason="Unattached volume still billed",
        recommendation="Delete or snapshot the volume",
        est_daily_savings_usd=42.5,
        confidence="high",
        category="idle",
    )
    ev = event_from_anomaly_action(
        action,
        account_id="123456789012",
        report_key="anom::v1::profile::repo",
        action_idx=0,
        expected_apply=date(2026, 8, 1),
    )
    assert ev is not None
    assert ev.amount_daily == -42.5
    assert ev.category == "optimization"
    assert ev.confidence == 90.0
    assert ev.source == "anomalies"
    assert ev.external_id == "anomaly:123456789012:anom::v1::profile::repo:0"


def test_event_from_pr_predictor_uses_range_midpoint():
    verdict = AgentVerdict(
        verdict="Moderate increase",
        est_daily_delta_usd=10.0,
        est_daily_delta_low_usd=8.0,
        est_daily_delta_high_usd=12.0,
        direction="increase",
        confidence="high",
    )
    ev = event_from_pr_predictor(
        "org/repo#99",
        verdict,
        expected_deploy=date(2026, 8, 5),
    )
    assert ev is not None
    assert ev.amount_daily == 10.0
    assert ev.confidence == 90.0
    assert ev.start_date == date(2026, 8, 5)


def test_merge_events_dedupes_by_external_id():
    existing = [
        CostEvent(
            name="a",
            start_date=date(2026, 7, 1),
            effect=Effect.STEP,
            external_id="open_pr:org/repo#1",
        ),
    ]
    incoming = [
        CostEvent(
            name="dup",
            start_date=date(2026, 7, 2),
            effect=Effect.STEP,
            external_id="open_pr:org/repo#1",
        ),
        CostEvent(
            name="new",
            start_date=date(2026, 7, 3),
            effect=Effect.STEP,
            external_id="open_pr:org/repo#2",
        ),
    ]
    merged, added = merge_events(existing, incoming)
    assert added == 1
    assert len(merged) == 2
    assert merged[-1].name == "new"


def test_cost_event_roundtrip_dict():
    from src.forecast.events import CostEvent, Effect

    ev = CostEvent(
        name="test",
        start_date=date(2026, 8, 1),
        effect=Effect.RAMP,
        amount_daily=-12.5,
        ramp_days=14,
        external_id="x",
    )
    restored = CostEvent.from_dict(ev.to_dict())
    assert restored.name == ev.name
    assert restored.effect is Effect.RAMP
    assert restored.amount_daily == -12.5
