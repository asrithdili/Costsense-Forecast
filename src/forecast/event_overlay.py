"""Apply ledger events on top of a saved 7-day forecast (display-time overlay)."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Sequence

import pandas as pd

from src.forecast import baselines as B
from src.forecast.events import CostEvent, Effect
from src.forecast.scenario import Projection, project

_DISPLAY_TAIL_DAYS = 7


def _event_display_end(ev: CostEvent) -> date:
    """Last day an event should appear in the ledger / waterfall horizon."""
    if ev.effect is Effect.PULSE and ev.end_date:
        return ev.end_date
    if ev.effect is Effect.RAMP and ev.ramp_days > 0:
        return ev.start_date + timedelta(days=ev.ramp_days - 1)
    return ev.start_date + timedelta(days=_DISPLAY_TAIL_DAYS - 1)


def daily_event_delta(base: float, day: date, events: Sequence[CostEvent]) -> float:
    """Confidence-weighted $/day change from enabled events on ``base`` spend."""
    active = [e for e in events if e.enabled]
    if not active or base < 0:
        return 0.0

    mult_exp = 1.0
    add_exp = 0.0
    for ev in active:
        p = ev.confidence / 100.0
        if ev.effect is Effect.MULTIPLIER:
            delta = ev.multiplier_on(day) - 1.0
            mult_exp *= 1.0 + delta * p
        else:
            add_exp += ev.additive_on(day) * p

    adjusted = max(0.0, base * mult_exp + add_exp)
    return adjusted - base


def overlay_events_on_forecast_df(
    fc_df: pd.DataFrame,
    events: Sequence[CostEvent],
) -> pd.DataFrame:
    """Return a copy of the forecast frame with event deltas layered on adjusted band."""
    if fc_df.empty or not events:
        return fc_df.copy()

    out = fc_df.copy()
    deltas: list[float] = []
    for _, row in out.iterrows():
        day = date.fromisoformat(str(row["target_date"])[:10])
        base = float(row.get("baseline_usd", row.get("adjusted_usd", 0.0)))
        deltas.append(daily_event_delta(base, day, events))

    out["event_delta_usd"] = deltas
    for col in ("adjusted_usd", "lower_usd", "upper_usd"):
        if col in out.columns:
            out[col] = out[col].astype(float) + out["event_delta_usd"]
    return out


def projection_from_forecast_df(
    fc_df: pd.DataFrame,
    events: Sequence[CostEvent],
) -> Projection:
    """Build a projection aligned to forecast rows (chart window only)."""
    dates = [date.fromisoformat(str(d)[:10]) for d in fc_df["target_date"]]
    baselines = [float(v) for v in fc_df["baseline_usd"]]
    return project(
        baselines,
        method="forecast_baseline",
        explanation="Forecast baseline from the saved model run.",
        residual_std=0.0,
        start_day=dates[0],
        events=events,
    )


def projection_for_ledger(
    fc_df: pd.DataFrame,
    events: Sequence[CostEvent],
    *,
    hist_values: Sequence[float] | None = None,
    hist_dates: Sequence[date] | None = None,
    fallback_start: date | None = None,
    min_horizon_days: int = 7,
) -> Projection:
    """Projection for waterfall + ledger — extends through each event's active window."""
    enabled = [e for e in events if e.enabled]

    if not fc_df.empty:
        baselines = [float(v) for v in fc_df["baseline_usd"]]
        start_day = date.fromisoformat(str(fc_df["target_date"].iloc[0])[:10])
        last_day = date.fromisoformat(str(fc_df["target_date"].iloc[-1])[:10])
        explanation = "Forecast baseline from the saved model run."
    elif hist_values and hist_dates and fallback_start is not None:
        built = B.build_baseline(
            "run_rate", hist_values, hist_dates, min_horizon_days,
        )
        baselines = list(built.values)
        start_day = fallback_start
        last_day = start_day + timedelta(days=min_horizon_days - 1)
        explanation = built.explanation
    else:
        raise ValueError("projection_for_ledger needs a forecast or cost history")

    horizon_end = last_day
    for ev in enabled:
        horizon_end = max(horizon_end, _event_display_end(ev))

    total_days = (horizon_end - start_day).days + 1
    fill = baselines[-1] if baselines else 0.0
    while len(baselines) < total_days:
        baselines.append(fill)

    return project(
        baselines,
        method="forecast_baseline",
        explanation=explanation,
        residual_std=0.0,
        start_day=start_day,
        events=events,
    )


def total_event_delta_usd(fc_df: pd.DataFrame) -> float:
    if fc_df.empty or "event_delta_usd" not in fc_df.columns:
        return 0.0
    return float(fc_df["event_delta_usd"].sum())
