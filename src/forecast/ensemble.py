"""Auto-tuned fast-level forecaster for daily cost.

No hardcoded assumptions:
  - naive_weight, trim_window, trim_fraction, intra-week decay slope are
    ALL selected per account by walk-forward search on that account's own
    history.
  - Day-of-week ratios are computed from the account's recent 4 weeks.
  - Confidence band width is derived from recent stddev.

The tuner returns a `TunedParams` object that gets persisted alongside the
forecast so a reviewer can audit exactly what values were used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from itertools import product

import pandas as pd

from src.forecast.timeseries import ForecastPoint


# Search grid — these are the ONLY places numeric candidates appear, and
# every value is a candidate the tuner will evaluate, not a chosen constant.
NAIVE_WEIGHT_CANDIDATES = (0.3, 0.5, 0.7, 0.85, 1.0)
TRIM_WINDOW_CANDIDATES = (7, 14, 21, 28)
TRIM_FRACTION_CANDIDATES = (0.0, 0.1, 0.15, 0.2)
DECAY_SLOPE_CANDIDATES = (0.0, 0.03, 0.05, 0.08)   # 0 = flat, higher = fade toward trimmed mean


@dataclass
class TunedParams:
    naive_weight: float
    trim_window: int
    trim_fraction: float
    decay_slope: float
    tuning_wape: float           # WAPE achieved on the tuning windows
    tuning_days_scored: int
    dow_ratios: dict[int, float] = field(default_factory=dict)


def _fast_level(history: pd.DataFrame, params: TunedParams) -> tuple[float, float]:
    """Return (near, far). near = the naive-heavy level for day 1; far = the
    trimmed-mean level used for the far end of the horizon."""
    s = history["amount_usd"].astype(float)
    if s.empty:
        return 0.0, 0.0
    naive = float(s.iloc[-1])
    tail = s.tail(params.trim_window).sort_values()
    if len(tail) >= 5 and params.trim_fraction > 0:
        cut = max(1, int(len(tail) * params.trim_fraction))
        if 2 * cut < len(tail):
            trimmed = float(tail.iloc[cut:-cut].mean())
        else:
            trimmed = float(tail.mean())
    else:
        trimmed = float(tail.mean())
    near = params.naive_weight * naive + (1 - params.naive_weight) * trimmed
    return near, trimmed


def _day_of_week_ratios(history: pd.DataFrame, lookback: int = 28) -> dict[int, float]:
    """Ratio of each weekday's average to the overall lookback-window average.
    Clamped to [0.5, 1.5] so a single outlier day can't blow up a weekday."""
    if history.empty:
        return {i: 1.0 for i in range(7)}
    recent = history.tail(lookback).copy()
    recent["dow"] = pd.to_datetime(recent["day"]).dt.weekday
    overall = float(recent["amount_usd"].mean())
    if overall <= 0:
        return {i: 1.0 for i in range(7)}
    ratios: dict[int, float] = {}
    for d in range(7):
        rows = recent[recent["dow"] == d]["amount_usd"]
        if len(rows) < 3:
            ratios[d] = 1.0
            continue
        ratios[d] = max(0.5, min(1.5, float(rows.mean()) / overall))
    return ratios


def _forecast_with_params(
    history: pd.DataFrame, cutoff: date, params: TunedParams,
    horizon_days: int = 7,
) -> list[ForecastPoint]:
    near, far = _fast_level(history, params)
    recent = history.tail(params.trim_window)["amount_usd"]
    std = float(recent.std()) if len(recent) >= 3 else 0.0

    out: list[ForecastPoint] = []
    for i in range(1, horizon_days + 1):
        alpha = max(0.0, 1.0 - params.decay_slope * (i - 1))
        base_level = alpha * near + (1 - alpha) * far
        target = cutoff + timedelta(days=i)
        ratio = params.dow_ratios.get(target.weekday(), 1.0)
        level = base_level * ratio
        out.append(ForecastPoint(
            target_date=target,
            predicted_usd=max(0.0, level),
            lower_usd=max(0.0, level - 1.28 * std),
            upper_usd=max(0.0, level + 1.28 * std),
        ))
    return out


def tune_params(
    history: pd.DataFrame,
    cutoff: date,
    horizon_days: int = 7,
    tune_windows: int = 4,
    tune_offset_weeks: int = 1,
) -> TunedParams:
    """Walk-forward search over the parameter grid.

    Honest cross-validation: the tuning windows are OLDER than the eval
    window. Origins are `cutoff - 7*(tune_offset_weeks + k)` for
    k=1..tune_windows — so the tuner never sees data from the week it's
    about to predict.
    """
    hist = history.copy()
    hist["day"] = pd.to_datetime(hist["day"])
    hist = hist.sort_values("day").reset_index(drop=True)
    actual_by_day = {d.date(): float(a) for d, a in
                     zip(hist["day"], hist["amount_usd"])}

    best: TunedParams | None = None
    best_err = float("inf")

    for nw, tw, tf, ds in product(
        NAIVE_WEIGHT_CANDIDATES, TRIM_WINDOW_CANDIDATES,
        TRIM_FRACTION_CANDIDATES, DECAY_SLOPE_CANDIDATES,
    ):
        candidate = TunedParams(
            naive_weight=nw, trim_window=tw,
            trim_fraction=tf, decay_slope=ds,
            tuning_wape=0.0, tuning_days_scored=0,
        )
        err_sum = 0.0
        act_sum = 0.0
        n = 0
        for k in range(1, tune_windows + 1):
            origin = cutoff - timedelta(days=7 * (tune_offset_weeks + k))
            train = hist[hist["day"] < pd.Timestamp(origin)]
            if train.empty or (origin - train["day"].min().date()).days < 21:
                continue
            candidate.dow_ratios = _day_of_week_ratios(train)
            fc = _forecast_with_params(train, origin, candidate,
                                        horizon_days=horizon_days)
            for p in fc:
                a = actual_by_day.get(p.target_date)
                if a is None:
                    continue
                err_sum += abs(p.predicted_usd - a)
                act_sum += abs(a)
                n += 1
        if n == 0 or act_sum == 0:
            continue
        if err_sum < best_err:
            best_err = err_sum
            best = candidate
            best.tuning_wape = err_sum / act_sum
            best.tuning_days_scored = n

    if best is None:
        best = TunedParams(
            naive_weight=1.0, trim_window=7, trim_fraction=0.0, decay_slope=0.0,
            tuning_wape=float("nan"), tuning_days_scored=0,
        )
    best.dow_ratios = _day_of_week_ratios(hist)
    return best


def forecast_auto(
    history: pd.DataFrame,
    cutoff: date,
    horizon_days: int = 7,
) -> tuple[list[ForecastPoint], TunedParams]:
    """Public entry point. Auto-tunes params on the given history, then
    forecasts `horizon_days` forward. Returns (points, tuned_params) so
    callers can persist the params for audit."""
    params = tune_params(history, cutoff, horizon_days=horizon_days)
    fc = _forecast_with_params(history, cutoff, params, horizon_days=horizon_days)
    return fc, params


# ---------------------------------------------------------------------------
# Backwards-compat shim — old callers used `_ewm_forecast(hist, cutoff, ...)`
# and expected `list[ForecastPoint]`. Route them through the auto-tuner so
# they benefit without a signature change.
# ---------------------------------------------------------------------------

def _ewm_forecast(history: pd.DataFrame, cutoff: date,
                  horizon_days: int = 7,
                  **_ignored) -> list[ForecastPoint]:
    fc, _ = forecast_auto(history, cutoff, horizon_days=horizon_days)
    return fc


def tuned_params_dict(p: TunedParams) -> dict:
    return {
        "naive_weight": p.naive_weight,
        "trim_window": p.trim_window,
        "trim_fraction": p.trim_fraction,
        "decay_slope": p.decay_slope,
        "tuning_wape": None if p.tuning_wape != p.tuning_wape else round(p.tuning_wape, 4),
        "tuning_days_scored": p.tuning_days_scored,
        "dow_ratios": {int(k): round(v, 3) for k, v in p.dow_ratios.items()},
    }
