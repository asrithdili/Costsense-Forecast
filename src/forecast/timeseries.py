"""Time-series forecast: Prophet trained on data STRICTLY before the cutoff.

Two modes:
  - forecast_next_7_days(history, cutoff) — baseline-only, no PR awareness.
  - forecast_with_pr_regressor(history, cutoff, pr_steps) — adds a
    cumulative PR-step regressor so Prophet can attribute historical
    step-ups/-downs to specific PRs and project the current level forward.

The cutoff discipline is what makes the predictions log honest — a forecast
tagged for 2026-07-27 must have been produced using only data up to 2026-07-20.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

try:
    from prophet import Prophet
except ImportError:
    Prophet = None


@dataclass
class ForecastPoint:
    target_date: date
    predicted_usd: float
    lower_usd: float
    upper_usd: float


@dataclass
class PrStep:
    """A single PR's effect on daily cost: `delta_usd` starting `from_day`."""
    from_day: date
    delta_usd: float
    pr_id: str = ""


def build_step_series(
    steps: list[PrStep], start: date, end: date,
) -> pd.DataFrame:
    """Cumulative daily $ from all steps on each day in [start, end]."""
    days = pd.date_range(start=start, end=end, freq="D")
    series = pd.Series(0.0, index=days)
    for s in steps:
        mask = series.index >= pd.Timestamp(s.from_day)
        series.loc[mask] = series.loc[mask] + s.delta_usd
    return pd.DataFrame({"ds": series.index, "pr_cum_usd": series.values})


def forecast_next_7_days(history: pd.DataFrame, cutoff: date) -> list[ForecastPoint]:
    """history: DataFrame with columns [day, amount_usd] aggregated across services."""
    if Prophet is None:
        raise RuntimeError("prophet is not installed. `pip install prophet`.")

    train = history[history["day"] < pd.Timestamp(cutoff)].copy()
    if train.empty:
        raise ValueError(f"no training rows strictly before cutoff {cutoff}")

    train = train.rename(columns={"day": "ds", "amount_usd": "y"})
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality="auto",     # only enable when >= 2 full weeks of data
        changepoint_prior_scale=0.15,  # modest flexibility — bend on true level shifts, not noise
        changepoint_range=0.9,         # allow changepoints up to the last 10% of history
        seasonality_prior_scale=1.0,   # dampen seasonal component when signal is weak
    )
    model.fit(train)

    future = pd.DataFrame({"ds": [cutoff + timedelta(days=i) for i in range(1, 8)]})
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


def forecast_with_pr_regressor(
    history: pd.DataFrame,
    cutoff: date,
    pr_steps: list[PrStep],
    horizon_days: int = 7,
) -> tuple[list[ForecastPoint], pd.DataFrame]:
    """Prophet + PR-cumulative-delta regressor.

    Returns (forecast_points, pr_step_df) — the second element is the daily
    cumulative-PR series so the UI can render "PR-attributable" vs baseline.
    """
    if Prophet is None:
        raise RuntimeError("prophet is not installed. `pip install prophet`.")

    train = history[history["day"] < pd.Timestamp(cutoff)].copy()
    if train.empty:
        raise ValueError(f"no training rows strictly before cutoff {cutoff}")

    hist_min = train["day"].min().date()
    fut_max = cutoff + timedelta(days=horizon_days)
    step_df = build_step_series(pr_steps, start=hist_min, end=fut_max)

    train = train.rename(columns={"day": "ds", "amount_usd": "y"})
    train["ds"] = pd.to_datetime(train["ds"])
    step_df["ds"] = pd.to_datetime(step_df["ds"])
    train = train.merge(step_df, on="ds", how="left").fillna({"pr_cum_usd": 0.0})

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality="auto",     # only enable when >= 2 full weeks of data
        changepoint_prior_scale=0.15,  # modest flexibility — bend on true level shifts, not noise
        changepoint_range=0.9,         # allow changepoints up to the last 10% of history
        seasonality_prior_scale=1.0,   # dampen seasonal component when signal is weak
    )
    model.add_regressor("pr_cum_usd", mode="additive")
    model.fit(train)

    future_days = pd.DataFrame({
        "ds": pd.to_datetime(
            [cutoff + timedelta(days=i) for i in range(1, horizon_days + 1)]
        )
    })
    step_df_m = step_df.copy()
    step_df_m["ds"] = pd.to_datetime(step_df_m["ds"])
    future = future_days.merge(step_df_m, on="ds", how="left").fillna({"pr_cum_usd": 0.0})
    fc = model.predict(future)

    points = [
        ForecastPoint(
            target_date=row["ds"].date(),
            predicted_usd=max(0.0, float(row["yhat"])),
            lower_usd=max(0.0, float(row["yhat_lower"])),
            upper_usd=max(0.0, float(row["yhat_upper"])),
        )
        for _, row in fc.iterrows()
    ]
    return points, step_df


def in_sample_fit_prophet(
    history: pd.DataFrame,
    cutoff: date,
    lookback_days: int = 30,
) -> list[ForecastPoint]:
    """Train Prophet once at *cutoff*, return in-sample fit on recent days."""
    if Prophet is None:
        raise RuntimeError("prophet is not installed. `pip install prophet`.")

    train = history[history["day"] < pd.Timestamp(cutoff)].copy()
    if train.empty:
        return []

    train = train.rename(columns={"day": "ds", "amount_usd": "y"})
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality="auto",
        changepoint_prior_scale=0.15,
        changepoint_range=0.9,
        seasonality_prior_scale=1.0,
    )
    model.fit(train)
    fitted = model.predict(train)

    lookback_start = cutoff - timedelta(days=lookback_days)
    out: list[ForecastPoint] = []
    for _, row in fitted.iterrows():
        target = row["ds"].date()
        if target < lookback_start:
            continue
        out.append(ForecastPoint(
            target_date=target,
            predicted_usd=max(0.0, float(row["yhat"])),
            lower_usd=max(0.0, float(row["yhat_lower"])),
            upper_usd=max(0.0, float(row["yhat_upper"])),
        ))
    return out
