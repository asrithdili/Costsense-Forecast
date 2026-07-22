"""Cost Explorer client — daily unblended cost by service."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from src.aws.session import make_session


@dataclass
class DailyCost:
    day: date
    service: str
    amount_usd: float


@dataclass
class ForecastDay:
    """Single day from ``GetCostForecast``."""
    target_date: date
    predicted_usd: float
    lower_usd: float
    upper_usd: float


def _client(profile: str | None, region: str = "us-east-1"):
    session = make_session(profile)
    return session.client("ce", region_name=region)


def fetch_daily_costs(
    start: date,
    end: date,
    profile: str | None = None,
    region: str = "us-east-1",
) -> list[DailyCost]:
    """Rows grouped by service. `end` is EXCLUSIVE per Cost Explorer semantics."""
    ce = _client(profile, region)
    rows: list[DailyCost] = []
    next_token: str | None = None
    while True:
        kwargs = {
            "TimePeriod": {"Start": start.isoformat(), "End": end.isoformat()},
            "Granularity": "DAILY",
            "Metrics": ["UnblendedCost"],
            "GroupBy": [{"Type": "DIMENSION", "Key": "SERVICE"}],
        }
        if next_token:
            kwargs["NextPageToken"] = next_token
        resp = ce.get_cost_and_usage(**kwargs)
        for period in resp.get("ResultsByTime", []):
            day = date.fromisoformat(period["TimePeriod"]["Start"])
            for group in period.get("Groups", []):
                service = group["Keys"][0]
                amount = float(group["Metrics"]["UnblendedCost"]["Amount"])
                rows.append(DailyCost(day=day, service=service, amount_usd=amount))
        next_token = resp.get("NextPageToken")
        if not next_token:
            break
    return rows


def fetch_daily_totals(
    start: date,
    end: date,
    profile: str | None = None,
    region: str = "us-east-1",
    service: str | None = None,
) -> list[tuple[date, float]]:
    """Un-grouped daily totals — one row per day.

    If `service` is passed, filter to that Cost Explorer service dimension
    (e.g. "AWS Lambda", "Amazon Simple Storage Service").
    """
    ce = _client(profile, region)
    kwargs = {
        "TimePeriod": {"Start": start.isoformat(), "End": end.isoformat()},
        "Granularity": "DAILY",
        "Metrics": ["UnblendedCost"],
    }
    if service:
        kwargs["Filter"] = {
            "Dimensions": {"Key": "SERVICE", "Values": [service]}
        }
    resp = ce.get_cost_and_usage(**kwargs)
    out: list[tuple[date, float]] = []
    for period in resp.get("ResultsByTime", []):
        day = date.fromisoformat(period["TimePeriod"]["Start"])
        amount = float(period["Total"]["UnblendedCost"]["Amount"])
        out.append((day, amount))
    return out


def fetch_daily_by_service(
    start: date,
    end: date,
    profile: str | None = None,
    region: str = "us-east-1",
    min_daily_usd: float = 1.0,
) -> dict[str, list[tuple[date, float]]]:
    """Return {service_name: [(day, amount), ...]} for every service with
    at least one day above `min_daily_usd`. Used to populate the service
    selector and the cost-driver breakdown panel."""
    rows = fetch_daily_costs(start, end, profile=profile, region=region)
    by_service: dict[str, list[tuple[date, float]]] = {}
    for r in rows:
        by_service.setdefault(r.service, []).append((r.day, r.amount_usd))
    return {
        s: sorted(vals) for s, vals in by_service.items()
        if any(a >= min_daily_usd for _, a in vals)
    }


def fetch_actual_total(day: date, profile: str | None = None) -> float:
    totals = fetch_daily_totals(day, day + timedelta(days=1), profile=profile)
    for d, amt in totals:
        if d == day:
            return amt
    return 0.0


def fetch_cost_forecast(
    cutoff: date,
    horizon_days: int = 7,
    profile: str | None = None,
    region: str = "us-east-1",
    service: str | None = None,
    metric: str = "UNBLENDED_COST",
    prediction_interval_level: int = 80,
) -> list[ForecastDay]:
    """Call ``GetCostForecast`` for daily spend after *cutoff*.

    Returns one point per target day in ``(cutoff, cutoff + horizon_days]``.
    AWS generates the forecast as-of the API call time (not as-of *cutoff*).
    """
    first_target = cutoff + timedelta(days=1)
    last_target = cutoff + timedelta(days=horizon_days)
    today = date.today()

    # CE requires Start <= today; clamp when cutoff is today/tomorrow-boundary.
    period_start = first_target if first_target <= today else today
    period_end = last_target + timedelta(days=1)  # exclusive

    if period_start > last_target:
        return []

    ce = _client(profile, region)
    kwargs: dict = {
        "TimePeriod": {
            "Start": period_start.isoformat(),
            "End": period_end.isoformat(),
        },
        "Metric": metric,
        "Granularity": "DAILY",
        "PredictionIntervalLevel": prediction_interval_level,
    }
    if service:
        kwargs["Filter"] = {
            "Dimensions": {"Key": "SERVICE", "Values": [service]},
        }

    resp = ce.get_cost_forecast(**kwargs)
    out: list[ForecastDay] = []
    for bucket in resp.get("ForecastResultsByTime", []):
        target = date.fromisoformat(bucket["TimePeriod"]["Start"])
        if target <= cutoff or target > last_target:
            continue
        mean = float(bucket.get("MeanValue", 0))
        lower = float(bucket.get("PredictionIntervalLowerBound", mean))
        upper = float(bucket.get("PredictionIntervalUpperBound", mean))
        out.append(ForecastDay(
            target_date=target,
            predicted_usd=max(0.0, mean),
            lower_usd=max(0.0, lower),
            upper_usd=max(0.0, upper),
        ))
    out.sort(key=lambda p: p.target_date)
    return out


