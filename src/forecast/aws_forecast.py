"""AWS Cost Explorer native forecast via ``GetCostForecast``.

This is AWS's own statistical forecast — useful as a baseline comparison
against CostSense's auto-tuned EWM model. It does **not** incorporate PR
deltas; those are layered on afterward by ``pipeline.adjust`` like EWM.

Limitations (AWS API constraints):
  - Forecast is always **as-of now** — you cannot replay what CE would
    have predicted at a historical cutoff (walk-forward backtest skips).
  - ``TimePeriod.Start`` must be on or before today.
  - Daily granularity supports up to ~3 months forward.
  - No PR / code-change awareness.
"""
from __future__ import annotations

from datetime import date

from botocore.exceptions import BotoCoreError, ClientError

from src.aws.cost_explorer import fetch_cost_forecast
from src.forecast.timeseries import ForecastPoint


def forecast_from_ce(
    cutoff: date,
    horizon_days: int = 7,
    profile: str | None = None,
    service: str | None = None,
    prediction_interval_level: int = 80,
) -> list[ForecastPoint]:
    """Return ``horizon_days`` daily points starting the day after *cutoff*."""
    try:
        rows = fetch_cost_forecast(
            cutoff=cutoff,
            horizon_days=horizon_days,
            profile=profile,
            service=service,
            prediction_interval_level=prediction_interval_level,
        )
    except (BotoCoreError, ClientError) as e:
        raise RuntimeError(f"GetCostForecast failed: {e}") from e
    return [
        ForecastPoint(
            target_date=r.target_date,
            predicted_usd=r.predicted_usd,
            lower_usd=r.lower_usd,
            upper_usd=r.upper_usd,
        )
        for r in rows
    ]
