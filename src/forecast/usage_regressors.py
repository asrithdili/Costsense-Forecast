"""Fetch daily CloudWatch usage metrics to use as forecast regressors.

Cost tracks usage. If we tell Prophet how many Lambda-seconds ran yesterday,
it can learn cost = k * usage + baseline, which beats time-series shape
extrapolation when usage is volatile.

We fetch across a small set of common namespaces and pick the metric with
the highest daily-cost correlation for the account. That's the causal
regressor — anything with r < 0.5 is dropped as noise.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache

import pandas as pd

from src.aws.session import make_session


# (namespace, metric_name, statistic) — extend as we find more accounts
CANDIDATE_METRICS: list[tuple[str, str, str]] = [
    ("AWS/Lambda", "Invocations", "Sum"),
    ("AWS/Lambda", "Duration", "Sum"),
    ("AWS/ECS", "CPUUtilization", "Average"),
    ("AWS/RDS", "DatabaseConnections", "Average"),
    ("AWS/S3", "NumberOfObjects", "Average"),
    ("AWS/ApiGateway", "Count", "Sum"),
    ("AWS/DynamoDB", "ConsumedReadCapacityUnits", "Sum"),
    ("AWS/DynamoDB", "ConsumedWriteCapacityUnits", "Sum"),
]


@dataclass
class UsageSeries:
    label: str            # e.g. "Lambda/Duration"
    values: dict[date, float]
    correlation: float    # Pearson r with daily cost, on the fetched window


@lru_cache(maxsize=16)
def _cw_client(profile: str | None, region: str):
    session = make_session(profile)
    return session.client("cloudwatch", region_name=region)


def _fetch(profile: str | None, region: str, ns: str, metric: str,
           stat: str, days: int) -> dict[date, float]:
    cw = _cw_client(profile, region)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    try:
        resp = cw.get_metric_statistics(
            Namespace=ns, MetricName=metric,
            Dimensions=[], StartTime=start, EndTime=end,
            Period=86400, Statistics=[stat],
        )
    except Exception:  # noqa: BLE001
        return {}
    return {p["Timestamp"].date(): float(p[stat])
            for p in resp.get("Datapoints", [])}


def pick_best_regressor(
    profile: str | None,
    region: str,
    cost_daily: dict[date, float],
    days: int = 90,
    min_corr: float = 0.35,
) -> UsageSeries | None:
    """Fetch every candidate metric, pick the one most correlated with cost.

    Returns None if no metric clears `min_corr` — no point feeding Prophet
    noise as a regressor.
    """
    cost_series = pd.Series(cost_daily)
    best: UsageSeries | None = None

    for ns, metric, stat in CANDIDATE_METRICS:
        data = _fetch(profile, region, ns, metric, stat, days)
        if len(data) < 14:
            continue
        merged = pd.DataFrame({
            "cost": cost_series,
            "usage": pd.Series(data),
        }).dropna()
        if len(merged) < 14:
            continue
        # need non-zero variance for correlation
        if merged["usage"].std() == 0 or merged["cost"].std() == 0:
            continue
        r = float(merged["cost"].corr(merged["usage"]))
        if abs(r) < min_corr:
            continue
        candidate = UsageSeries(
            label=f"{ns.replace('AWS/', '')}/{metric}",
            values=data,
            correlation=r,
        )
        if best is None or abs(candidate.correlation) > abs(best.correlation):
            best = candidate

    return best


def build_future_regressor(
    series: UsageSeries, cutoff: date, horizon_days: int = 7,
    trim_window: int = 14,
) -> dict[date, float]:
    """Extend the historical series forward using the trimmed-mean of recent
    values. Same reasoning as the cost forecast — future usage is ~ recent
    usage minus outliers."""
    if not series.values:
        return {}
    recent = pd.Series({d: v for d, v in series.values.items()
                        if d <= cutoff}).sort_index().tail(trim_window)
    if len(recent) < 3:
        future_level = float(recent.mean()) if len(recent) else 0.0
    else:
        sorted_vals = recent.sort_values()
        cut = max(1, int(len(sorted_vals) * 0.15))
        future_level = float(sorted_vals.iloc[cut:-cut].mean()
                             if len(sorted_vals) > 2 * cut else sorted_vals.mean())
    out = dict(series.values)  # copy historical
    for i in range(1, horizon_days + 1):
        out[cutoff + timedelta(days=i)] = future_level
    return out
