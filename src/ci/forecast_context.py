"""Build lightweight forecast context for PR cost-check job summaries."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from src.aws.cost_explorer import fetch_daily_totals
from src.aws.profiles import resolve
from src.forecast.ensemble import forecast_auto
from src.forecast.timeseries import ForecastPoint


@dataclass
class ForecastContext:
    account_id: str
    profile: str | None
    cutoff: date
    model: str
    hist_df: pd.DataFrame
    fc_df: pd.DataFrame
    pr_series_df: pd.DataFrame


def _run_forecast(
    hist_df: pd.DataFrame,
    cutoff: date,
) -> tuple[list[ForecastPoint], str]:
    try:
        from src.forecast.lightgbm_model import forecast_lightgbm

        return forecast_lightgbm(hist_df, cutoff=cutoff), "lightgbm"
    except Exception:  # noqa: BLE001
        points, _ = forecast_auto(hist_df, cutoff=cutoff)
        return points, "ewm"


def build_forecast_context(
    profile: str | None,
    pr_daily_delta_usd: float = 0.0,
    history_days: int = 60,
    cutoff: date | None = None,
) -> ForecastContext:
    """Fetch CE history, forecast 7 days, layer this PR's daily delta."""
    cutoff = cutoff or date.today()
    start = cutoff - timedelta(days=history_days)

    info = resolve(profile) if profile else None
    account_id = (info.account_id if info and info.account_id else None) or "unknown"

    totals = fetch_daily_totals(start, cutoff, profile=profile)
    hist_df = pd.DataFrame([
        {
            "day": pd.Timestamp(d),
            "amount_usd": float(a),
            "actual_usd": float(a),
        }
        for d, a in totals
    ])

    forecast, model = _run_forecast(hist_df, cutoff)

    fc_rows: list[dict] = []
    for point in forecast:
        is_future = point.target_date > cutoff
        pr_delta = pr_daily_delta_usd if is_future else 0.0
        adjusted = max(0.0, point.predicted_usd + pr_delta)
        fc_rows.append({
            "target_date": pd.Timestamp(point.target_date),
            "baseline_usd": point.predicted_usd,
            "pr_delta_usd": pr_delta,
            "adjusted_usd": adjusted,
            "lower_usd": max(0.0, point.lower_usd + pr_delta),
            "upper_usd": max(0.0, point.upper_usd + pr_delta),
        })
    fc_df = pd.DataFrame(fc_rows)

    pr_series_df = pd.DataFrame(columns=["day", "pr_cum_usd"])
    if abs(pr_daily_delta_usd) > 0.001 and not fc_df.empty:
        series_start = (
            hist_df["day"].min()
            if not hist_df.empty
            else pd.Timestamp(cutoff - timedelta(days=history_days))
        )
        series_end = fc_df["target_date"].max()
        days = pd.date_range(series_start, series_end, freq="D")
        pr_series_df = pd.DataFrame({
            "day": days,
            "pr_cum_usd": [
                pr_daily_delta_usd if d.date() > cutoff else 0.0
                for d in days
            ],
        })

    return ForecastContext(
        account_id=account_id,
        profile=profile,
        cutoff=cutoff,
        model=model,
        hist_df=hist_df,
        fc_df=fc_df,
        pr_series_df=pr_series_df,
    )
