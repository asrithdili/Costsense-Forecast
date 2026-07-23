"""Baseline projection methods for event-based forecasting."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class Baseline:
    values: List[float]
    method: str
    explanation: str
    residual_std: float = 0.0


BASELINE_METHODS = {
    "Run-rate": "run_rate",
    "Linear trend": "linear",
    "Seasonal": "seasonal",
    "Driver-based": "driver",
}


def _weekday_factors(
    history: Sequence[float], hist_dates: Sequence[date],
) -> Dict[int, float]:
    buckets: Dict[int, List[float]] = {i: [] for i in range(7)}
    for d, v in zip(hist_dates, history):
        buckets[d.weekday()].append(v)
    overall = sum(history) / len(history) if history else 0.0
    if overall <= 0:
        return {i: 1.0 for i in range(7)}
    return {
        i: (sum(vals) / len(vals) / overall) if vals else 1.0
        for i, vals in buckets.items()
    }


def _linear_fit(history: Sequence[float]) -> Tuple[float, float, float]:
    n = len(history)
    if n < 2:
        v = history[0] if history else 0.0
        return 0.0, v, 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(history) / n
    denom = sum((x - mx) ** 2 for x in xs) or 1.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, history)) / denom
    intercept = my - slope * mx
    resid = [y - (slope * x + intercept) for x, y in zip(xs, history)]
    std = math.sqrt(sum(r * r for r in resid) / n) if n else 0.0
    return slope, intercept, std


def baseline_run_rate(
    history: Sequence[float],
    hist_dates: Sequence[date],
    horizon: int,
    trailing: int = 7,
) -> Baseline:
    window = list(history[-trailing:]) or [0.0]
    avg = sum(window) / len(window)
    _, _, std = _linear_fit(window)
    return Baseline(
        [avg] * horizon,
        "run_rate",
        f"Trailing {len(window)}-day average of ${avg:,.0f}/day, held flat. "
        "Ignores trend on purpose.",
        std,
    )


def baseline_linear(
    history: Sequence[float],
    hist_dates: Sequence[date],
    horizon: int,
) -> Baseline:
    slope, intercept, std = _linear_fit(history)
    n = len(history)
    values = [max(0.0, slope * (n + i) + intercept) for i in range(horizon)]
    direction = "rising" if slope > 0 else "falling" if slope < 0 else "flat"
    return Baseline(
        values,
        "linear",
        f"Least-squares trend, {direction} ${abs(slope):,.0f}/day per day. "
        "Sensitive to spikes near the end of the window.",
        std,
    )


def baseline_seasonal(
    history: Sequence[float],
    hist_dates: Sequence[date],
    horizon: int,
) -> Baseline:
    slope, intercept, std = _linear_fit(history)
    factors = _weekday_factors(history, hist_dates)
    n = len(history)
    last = hist_dates[-1] if hist_dates else date.today()
    values = []
    for i in range(horizon):
        day = last + timedelta(days=i + 1)
        trend = max(0.0, slope * (n + i) + intercept)
        values.append(trend * factors.get(day.weekday(), 1.0))
    return Baseline(
        values,
        "seasonal",
        "Linear trend scaled by observed weekday factors. Use when weekday "
        "and weekend spend differ materially.",
        std,
    )


def baseline_driver(
    unit_counts: Sequence[float],
    cost_per_unit_day: float,
    horizon: int,
    residual_std: float = 0.0,
) -> Baseline:
    last_count = unit_counts[-1] if unit_counts else 0.0
    values = [
        cost_per_unit_day * (unit_counts[i] if i < len(unit_counts) else last_count)
        for i in range(horizon)
    ]
    return Baseline(
        values,
        "driver",
        f"${cost_per_unit_day:,.2f} per unit per day × the unit-count plan. "
        "Ties the forecast to the business plan, not infra noise.",
        residual_std,
    )


def ramp_units(
    start_units: float,
    end_units: float,
    horizon: int,
    ramp_days: Optional[int] = None,
) -> List[float]:
    ramp_days = ramp_days or horizon
    out: List[float] = []
    for i in range(horizon):
        frac = min(1.0, (i + 1) / float(ramp_days)) if ramp_days else 1.0
        out.append(start_units + (end_units - start_units) * frac)
    return out


def build_baseline(
    method: str,
    history: Sequence[float],
    hist_dates: Sequence[date],
    horizon: int,
    unit_counts: Optional[Sequence[float]] = None,
    cost_per_unit_day: float = 0.0,
) -> Baseline:
    if method == "run_rate":
        return baseline_run_rate(history, hist_dates, horizon)
    if method == "linear":
        return baseline_linear(history, hist_dates, horizon)
    if method == "seasonal":
        return baseline_seasonal(history, hist_dates, horizon)
    if method == "driver":
        _, _, std = _linear_fit(history)
        return baseline_driver(
            unit_counts or [], cost_per_unit_day, horizon, std,
        )
    raise ValueError(f"unknown baseline method: {method!r}")
