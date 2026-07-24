"""Tests for forecast event overlay."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.forecast.event_overlay import daily_event_delta, overlay_events_on_forecast_df
from src.forecast.events import CostEvent, Effect
from src.forecast.events import CostEvent, Effect


def test_daily_event_delta_additive():
    ev = CostEvent(
        name="test",
        start_date=date(2026, 8, 1),
        effect=Effect.STEP,
        amount_daily=100.0,
        confidence=100,
    )
    delta = daily_event_delta(1000.0, date(2026, 8, 1), [ev])
    assert delta == pytest.approx(100.0)


def test_overlay_bumps_adjusted_band():
    fc_df = pd.DataFrame([
        {
            "target_date": "2026-08-01",
            "baseline_usd": 1000.0,
            "pr_delta_usd": 50.0,
            "adjusted_usd": 1050.0,
            "lower_usd": 900.0,
            "upper_usd": 1200.0,
        },
    ])
    ev = CostEvent(
        name="save",
        start_date=date(2026, 8, 1),
        effect=Effect.STEP,
        amount_daily=-100.0,
        confidence=100,
    )
    out = overlay_events_on_forecast_df(fc_df, [ev])
    assert out["event_delta_usd"].iloc[0] == pytest.approx(-100.0)
    assert out["adjusted_usd"].iloc[0] == pytest.approx(950.0)
    assert out["lower_usd"].iloc[0] == pytest.approx(800.0)
    assert out["upper_usd"].iloc[0] == pytest.approx(1100.0)


def test_projection_for_ledger_extends_past_forecast_window():
    from src.forecast.event_overlay import projection_for_ledger

    fc_df = pd.DataFrame([
        {
            "target_date": "2026-07-29",
            "baseline_usd": 1000.0,
            "adjusted_usd": 1000.0,
            "lower_usd": 900.0,
            "upper_usd": 1100.0,
            "pr_delta_usd": 0.0,
        },
        {
            "target_date": "2026-08-04",
            "baseline_usd": 1000.0,
            "adjusted_usd": 1000.0,
            "lower_usd": 900.0,
            "upper_usd": 1100.0,
            "pr_delta_usd": 0.0,
        },
    ])
    ev = CostEvent(
        name="onboarding",
        start_date=date(2026, 8, 15),
        effect=Effect.RAMP,
        amount_daily=100.0,
        ramp_days=14,
        confidence=100,
    )
    proj = projection_for_ledger(fc_df, [ev])
    assert proj.horizon_days > len(fc_df)
    contribs = proj.contributions()
    assert len(contribs) == 1
    assert contribs[0][1] > 0
