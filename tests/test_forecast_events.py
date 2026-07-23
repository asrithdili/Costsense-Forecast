"""Unit tests for event-based forecast engine."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.forecast import baselines as B
from src.forecast.events import CostEvent, Effect
from src.forecast.scenario import project


ANCHOR = date(2026, 7, 23)


def _hist(n: int = 30, base: float = 5_000.0) -> tuple[list[date], list[float]]:
    dates = [ANCHOR - timedelta(days=n - i) for i in range(n)]
    values = [base + i * 10 for i in range(n)]
    return dates, values


def test_step_additive_saving():
    ev = CostEvent(
        name="save",
        start_date=ANCHOR,
        effect=Effect.STEP,
        amount_daily=-100.0,
        confidence=100,
    )
    base = B.baseline_run_rate([5000.0] * 10, [ANCHOR] * 10, 5)
    proj = project(
        base.values, method=base.method, explanation=base.explanation,
        residual_std=0, start_day=ANCHOR, events=[ev],
    )
    assert proj.expected[0] == pytest.approx(4900.0)
    assert proj.total_expected == pytest.approx(4900.0 * 5)


def test_ramp_reaches_full_effect():
    ev = CostEvent(
        name="ramp up",
        start_date=ANCHOR,
        effect=Effect.RAMP,
        amount_daily=300.0,
        ramp_days=3,
        confidence=100,
    )
    base = B.baseline_run_rate([1000.0] * 5, [ANCHOR] * 5, 5)
    proj = project(
        base.values, method=base.method, explanation=base.explanation,
        residual_std=0, start_day=ANCHOR, events=[ev],
    )
    assert proj.expected[0] == pytest.approx(1100.0)
    assert proj.expected[1] == pytest.approx(1200.0)
    assert proj.expected[2] == pytest.approx(1300.0)
    assert proj.expected[3] == pytest.approx(1300.0)


def test_pulse_bounded():
    ev = CostEvent(
        name="pulse",
        start_date=ANCHOR,
        effect=Effect.PULSE,
        amount_daily=200.0,
        end_date=ANCHOR + timedelta(days=1),
        confidence=100,
    )
    base = B.baseline_run_rate([1000.0] * 5, [ANCHOR] * 5, 5)
    proj = project(
        base.values, method=base.method, explanation=base.explanation,
        residual_std=0, start_day=ANCHOR, events=[ev],
    )
    assert proj.expected[0] == pytest.approx(1200.0)
    assert proj.expected[1] == pytest.approx(1200.0)
    assert proj.expected[2] == pytest.approx(1000.0)


def test_multiplier_compounds():
    ev1 = CostEvent(
        name="m1", start_date=ANCHOR, effect=Effect.MULTIPLIER,
        multiplier_pct=30, confidence=100,
    )
    ev2 = CostEvent(
        name="m2", start_date=ANCHOR, effect=Effect.MULTIPLIER,
        multiplier_pct=30, confidence=100,
    )
    base = B.baseline_run_rate([1000.0] * 3, [ANCHOR] * 3, 3)
    proj = project(
        base.values, method=base.method, explanation=base.explanation,
        residual_std=0, start_day=ANCHOR, events=[ev1, ev2],
    )
    assert proj.expected[0] == pytest.approx(1000.0 * 1.3 * 1.3)


def test_confidence_weights_expected():
    ev = CostEvent(
        name="half",
        start_date=ANCHOR,
        effect=Effect.STEP,
        amount_daily=200.0,
        confidence=50,
    )
    base = B.baseline_run_rate([1000.0] * 3, [ANCHOR] * 3, 3)
    proj = project(
        base.values, method=base.method, explanation=base.explanation,
        residual_std=0, start_day=ANCHOR, events=[ev],
    )
    assert proj.expected[0] == pytest.approx(1100.0)


def test_worst_includes_increases_not_savings():
    ev = CostEvent(
        name="maybe save",
        start_date=ANCHOR,
        effect=Effect.STEP,
        amount_daily=-500.0,
        confidence=50,
    )
    base = B.baseline_run_rate([1000.0] * 3, [ANCHOR] * 3, 3)
    proj = project(
        base.values, method=base.method, explanation=base.explanation,
        residual_std=0, start_day=ANCHOR, events=[ev],
    )
    assert proj.expected[0] == pytest.approx(975.0)
    assert proj.worst[0] == pytest.approx(1000.0)
    assert proj.best[0] == pytest.approx(750.0)


def test_disabled_event_ignored():
    ev = CostEvent(
        name="off",
        start_date=ANCHOR,
        effect=Effect.STEP,
        amount_daily=999.0,
        enabled=False,
    )
    base = B.baseline_run_rate([1000.0] * 3, [ANCHOR] * 3, 3)
    proj = project(
        base.values, method=base.method, explanation=base.explanation,
        residual_std=0, start_day=ANCHOR, events=[ev],
    )
    assert proj.expected == proj.baseline
    assert proj.contributions() == []


def test_contributions_sum_to_headline_delta():
    events = [
        CostEvent(
            name="a", start_date=ANCHOR, effect=Effect.STEP,
            amount_daily=100.0, confidence=100,
        ),
        CostEvent(
            name="b", start_date=ANCHOR + timedelta(days=2),
            effect=Effect.STEP, amount_daily=-50.0, confidence=100,
        ),
    ]
    dates, values = _hist(14)
    base = B.baseline_seasonal(values, dates, 30)
    proj = project(
        base.values, method=base.method, explanation=base.explanation,
        residual_std=base.residual_std, start_day=dates[-1] + timedelta(days=1),
        events=events,
    )
    contrib_sum = sum(amt for _, amt in proj.contributions())
    assert contrib_sum == pytest.approx(proj.event_contribution, rel=1e-6)


def test_budget_crossing():
    base = B.baseline_run_rate([1000.0] * 5, [ANCHOR] * 5, 10)
    proj = project(
        base.values, method=base.method, explanation=base.explanation,
        residual_std=0, start_day=ANCHOR, events=[],
    )
    crossing = proj.budget_crossing(5_500)
    assert crossing == ANCHOR + timedelta(days=4)


def test_all_baseline_methods_build():
    dates, values = _hist(21)
    for label, method in B.BASELINE_METHODS.items():
        units = B.ramp_units(750, 1400, 30, 45) if method == "driver" else None
        base = B.build_baseline(
            method, values, dates, 30,
            unit_counts=units, cost_per_unit_day=4.53,
        )
        assert len(base.values) == 30, label
        assert base.explanation
