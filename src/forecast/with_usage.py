"""Prophet with CloudWatch usage as an additive regressor.

Empirical hypothesis (validated on `dil-data-platform-dev`):
  cost = c0 + c1 * daily_lambda_duration + weekly_seasonality + noise
where c1 * daily_lambda_duration accounts for the workload-driven variance
the fast-level model treats as unexplained noise.

We ONLY enable this path when the best CloudWatch metric has |correlation| >
0.5 with daily cost. Otherwise we fall back to fast-level.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

try:
    from prophet import Prophet
except ImportError:
    Prophet = None

from src.forecast.ensemble import _ewm_forecast
from src.forecast.timeseries import ForecastPoint
from src.forecast.usage_regressors import (
    UsageSeries,
    build_future_regressor,
    pick_best_regressor,
)


def forecast_with_usage_regressor(
    history: pd.DataFrame,
    cutoff: date,
    usage: UsageSeries,
    horizon_days: int = 7,
) -> list[ForecastPoint]:
    """Fit Prophet(cost ~ trend + weekly + usage) and forecast 7 days."""
    if Prophet is None:
        raise RuntimeError("prophet is not installed")

    train = history[history["day"] < pd.Timestamp(cutoff)].copy()
    if train.empty:
        raise ValueError(f"no rows before {cutoff}")

    # Attach usage as a column (all history days)
    usage_full = build_future_regressor(usage, cutoff=cutoff,
                                         horizon_days=horizon_days)
    train = train.rename(columns={"day": "ds", "amount_usd": "y"})
    train["ds"] = pd.to_datetime(train["ds"])
    train["usage"] = train["ds"].dt.date.map(usage_full).astype(float)
    train = train.dropna(subset=["usage"])
    if len(train) < 14:
        raise ValueError("not enough joined rows to fit regressor")

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        changepoint_prior_scale=0.15,
        changepoint_range=0.9,
    )
    model.add_regressor("usage", mode="additive")
    model.fit(train)

    future_days = pd.to_datetime(
        [cutoff + timedelta(days=i) for i in range(1, horizon_days + 1)]
    )
    future = pd.DataFrame({"ds": future_days})
    future["usage"] = future["ds"].dt.date.map(usage_full).astype(float)
    fc = model.predict(future)

    return [
        ForecastPoint(
            target_date=row["ds"].date(),
            predicted_usd=max(0.0, float(row["yhat"])),
            lower_usd=max(0.0, float(row["yhat_lower"])),
            upper_usd=max(0.0, float(row["yhat_upper"])),
        )
        for _, row in fc.iterrows()
    ]


def try_forecast_with_usage(
    history: pd.DataFrame,
    cutoff: date,
    profile: str | None,
    region: str = "us-west-2",
    horizon_days: int = 7,
) -> tuple[list[ForecastPoint], UsageSeries | None]:
    """Attempt the usage-regressor path. Fall back to fast-level if no
    correlated CloudWatch metric is available. Returns (points, chosen_series)."""
    cost_daily = {ts.date(): float(v) for ts, v in
                  zip(history["day"], history["amount_usd"])}
    best = pick_best_regressor(profile=profile, region=region,
                               cost_daily=cost_daily)
    if best is None:
        # No high-correlation regressor. Fall back cleanly.
        return _ewm_forecast(history, cutoff=cutoff,
                             horizon_days=horizon_days), None
    try:
        pts = forecast_with_usage_regressor(history, cutoff, best,
                                             horizon_days=horizon_days)
        return pts, best
    except Exception:  # noqa: BLE001
        return _ewm_forecast(history, cutoff=cutoff,
                             horizon_days=horizon_days), None
