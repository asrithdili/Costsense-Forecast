"""Compose baseline + events into explainable scenario bands."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional, Sequence, Tuple

from src.forecast.events import CostEvent, Effect


@dataclass
class Projection:
    dates: List[date]
    baseline: List[float]
    expected: List[float]
    best: List[float]
    worst: List[float]
    events: List[CostEvent]
    method: str
    explanation: str

    @property
    def horizon_days(self) -> int:
        return len(self.dates)

    @property
    def total_expected(self) -> float:
        return sum(self.expected)

    @property
    def total_best(self) -> float:
        return sum(self.best)

    @property
    def total_worst(self) -> float:
        return sum(self.worst)

    @property
    def total_baseline(self) -> float:
        return sum(self.baseline)

    @property
    def event_contribution(self) -> float:
        return self.total_expected - self.total_baseline

    def contributions(self) -> List[Tuple[CostEvent, float]]:
        return [
            (ev, amt) for ev, amt in self.all_enabled_contributions()
            if abs(amt) > 0.005
        ]

    def all_enabled_contributions(self) -> List[Tuple[CostEvent, float]]:
        """Per-event totals over this projection window (may be zero)."""
        out: List[Tuple[CostEvent, float]] = []
        for ev in self.events:
            if not ev.enabled:
                continue
            p = ev.confidence / 100.0
            if ev.effect is Effect.MULTIPLIER:
                amt = sum(
                    b * (ev.multiplier_on(d) - 1.0) * p
                    for b, d in zip(self.baseline, self.dates)
                )
            else:
                amt = sum(ev.additive_on(d) for d in self.dates) * p
            out.append((ev, amt))
        return sorted(out, key=lambda t: abs(t[1]), reverse=True)

    def forecast_window_contributions(self) -> List[Tuple[CostEvent, float]]:
        """Alias kept for callers that only want material in-window impact."""
        return self.contributions()

    def budget_crossing(self, budget_total: float) -> Optional[date]:
        run = 0.0
        for d, v in zip(self.dates, self.expected):
            run += v
            if run > budget_total:
                return d
        return None


def project(
    baseline_values: Sequence[float],
    *,
    method: str,
    explanation: str,
    residual_std: float,
    start_day: date,
    events: Sequence[CostEvent],
    band_sigma: float = 1.0,
) -> Projection:
    horizon = len(baseline_values)
    dates = [start_day + timedelta(days=i) for i in range(horizon)]
    active = [e for e in events if e.enabled]

    expected: List[float] = []
    best: List[float] = []
    worst: List[float] = []

    for i, day in enumerate(dates):
        base = baseline_values[i]
        drift = residual_std * band_sigma

        mult_exp = mult_best = mult_worst = 1.0
        add_exp = add_best = add_worst = 0.0

        for ev in active:
            p = ev.confidence / 100.0
            if ev.effect is Effect.MULTIPLIER:
                delta = ev.multiplier_on(day) - 1.0
                mult_exp *= 1.0 + delta * p
                mult_best *= 1.0 + (delta if delta < 0 else 0.0)
                mult_worst *= 1.0 + (delta if delta > 0 else 0.0)
            else:
                amt = ev.additive_on(day)
                add_exp += amt * p
                add_best += amt if amt < 0 else 0.0
                add_worst += amt if amt > 0 else 0.0

        expected.append(max(0.0, base * mult_exp + add_exp))
        best.append(max(0.0, (base - drift) * mult_best + add_best))
        worst.append(max(0.0, (base + drift) * mult_worst + add_worst))

    return Projection(
        dates,
        list(baseline_values),
        expected,
        best,
        worst,
        list(events),
        method,
        explanation,
    )
