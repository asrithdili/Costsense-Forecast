"""Backtest helpers for the dashboard.

``training_fit_replay`` — train once at the cutoff, overlay in-sample fit on
recent history (what the user sees as "did the model learn the training data?").

``walk_forward`` — kept for CLI/scripts: retrain at multiple past cutoffs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.forecast.timeseries import PrStep


@dataclass
class ReplayPoint:
    origin_date: date       # cutoff used for training
    target_date: date       # day being predicted
    horizon: int            # days before cutoff (in-sample) or ahead (walk-forward)
    predicted_usd: float
    lower_usd: float
    upper_usd: float
    actual_usd: float | None


def training_fit_replay(
    history: pd.DataFrame,
    cutoff: date,
    model: str = "ewm",
    lookback_days: int = 30,
    pr_steps: list[PrStep] | None = None,
) -> list[ReplayPoint]:
    """Train once at *cutoff* and return in-sample fit on recent training days."""
    _ = pr_steps  # reserved — baseline in-sample fit ignores PR layer for now
    if history.empty:
        return []

    history = history.copy()
    history["day"] = pd.to_datetime(history["day"])
    history = history.sort_values("day").reset_index(drop=True)
    actual_by_day = {
        ts.date(): float(v) for ts, v in zip(history["day"], history["amount_usd"])
    }

    try:
        if model == "ewm":
            from src.forecast.ensemble import in_sample_fit_ewm
            fc = in_sample_fit_ewm(history, cutoff=cutoff,
                                   lookback_days=lookback_days)
        elif model == "lightgbm":
            from src.forecast.lightgbm_model import in_sample_fit_lightgbm
            fc = in_sample_fit_lightgbm(history, cutoff=cutoff,
                                        lookback_days=lookback_days)
        elif model == "prophet":
            from src.forecast.timeseries import in_sample_fit_prophet
            fc = in_sample_fit_prophet(history, cutoff=cutoff,
                                       lookback_days=lookback_days)
        elif model == "aws":
            return []
        else:
            raise ValueError(f"unknown model: {model}")
    except (ValueError, RuntimeError):
        return []

    out: list[ReplayPoint] = []
    for p in fc:
        out.append(ReplayPoint(
            origin_date=cutoff,
            target_date=p.target_date,
            horizon=(cutoff - p.target_date).days,
            predicted_usd=p.predicted_usd,
            lower_usd=p.lower_usd,
            upper_usd=p.upper_usd,
            actual_usd=actual_by_day.get(p.target_date),
        ))
    return out


def walk_forward(
    history: pd.DataFrame,
    end: date,
    n_origins: int = 8,
    stride_days: int = 7,
    horizon_days: int = 7,
    min_train_days: int = 30,
    pr_steps: list[PrStep] | None = None,
    model: str = "ewm",
) -> list[ReplayPoint]:
    """Produce past predictions by retraining at n_origins evenly-spaced cutoffs.

    history: [day (Timestamp), amount_usd].
    Origins are `end - stride*k` for k=1..n_origins, skipping any that leave
    fewer than `min_train_days` of history.
    """
    if history.empty:
        return []

    history = history.copy()
    history["day"] = pd.to_datetime(history["day"])
    history = history.sort_values("day").reset_index(drop=True)

    actual_by_day = {ts.date(): float(v) for ts, v in
                     zip(history["day"], history["amount_usd"])}

    origins: list[date] = []
    for k in range(1, n_origins + 1):
        origin = end - timedelta(days=stride_days * k)
        train_days = (origin - history["day"].min().date()).days
        if train_days < min_train_days:
            break
        origins.append(origin)

    out: list[ReplayPoint] = []
    for origin in origins:
        train = history[history["day"] < pd.Timestamp(origin)]
        if train.empty:
            continue
        eligible_steps = (
            [s for s in pr_steps if s.from_day < origin] if pr_steps else None
        )
        try:
            if model == "ewm":
                from src.forecast.ensemble import _ewm_forecast
                fc = _ewm_forecast(train, cutoff=origin,
                                   horizon_days=horizon_days)
            elif model == "aws":
                continue
            elif model == "lightgbm":
                from src.forecast.lightgbm_model import forecast_lightgbm
                fc = forecast_lightgbm(
                    train, cutoff=origin, horizon_days=horizon_days,
                )
            elif model == "prophet":
                from src.forecast.timeseries import (
                    forecast_next_7_days,
                    forecast_with_pr_regressor,
                )
                if eligible_steps:
                    fc, _ = forecast_with_pr_regressor(
                        train, cutoff=origin, pr_steps=eligible_steps,
                        horizon_days=horizon_days,
                    )
                else:
                    fc = forecast_next_7_days(train, cutoff=origin)
            else:
                raise ValueError(f"unknown model: {model}")
        except (ValueError, RuntimeError):
            continue
        for p in fc[:horizon_days]:
            out.append(ReplayPoint(
                origin_date=origin,
                target_date=p.target_date,
                horizon=(p.target_date - origin).days,
                predicted_usd=p.predicted_usd,
                lower_usd=p.lower_usd,
                upper_usd=p.upper_usd,
                actual_usd=actual_by_day.get(p.target_date),
            ))
    return out


def to_dict(points: list[ReplayPoint]) -> list[dict]:
    return [
        {**asdict(p),
         "origin_date": p.origin_date.isoformat(),
         "target_date": p.target_date.isoformat()}
        for p in points
    ]
