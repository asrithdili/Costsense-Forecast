"""Combine a Prophet forecast with per-day PR-attributable step values.

The regressor already builds the PR delta into `yhat`. So `adjusted_usd == yhat`.
We still expose `baseline_usd` (= yhat - pr_step) and `pr_delta_usd` (the daily
cumulative PR step) so the UI can decompose the forecast.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date

import pandas as pd

from src.forecast.timeseries import ForecastPoint


@dataclass
class AdjustedPoint:
    target_date: date
    baseline_usd: float
    pr_delta_usd: float
    adjusted_usd: float
    lower_usd: float
    upper_usd: float


def adjust(
    forecast: list[ForecastPoint],
    pr_step_series: pd.DataFrame | None = None,
) -> list[AdjustedPoint]:
    """pr_step_series: DataFrame(ds, pr_cum_usd). If None, assumes zero PR effect."""
    lookup: dict[date, float] = {}
    if pr_step_series is not None and not pr_step_series.empty:
        for _, r in pr_step_series.iterrows():
            lookup[pd.Timestamp(r["ds"]).date()] = float(r["pr_cum_usd"])

    out: list[AdjustedPoint] = []
    for p in forecast:
        pr_step = lookup.get(p.target_date, 0.0)
        out.append(AdjustedPoint(
            target_date=p.target_date,
            baseline_usd=p.predicted_usd - pr_step,
            pr_delta_usd=pr_step,
            adjusted_usd=p.predicted_usd,
            lower_usd=p.lower_usd,
            upper_usd=p.upper_usd,
        ))
    return out


def to_dict(points: list[AdjustedPoint]) -> list[dict]:
    return [
        {**asdict(p), "target_date": p.target_date.isoformat()} for p in points
    ]
