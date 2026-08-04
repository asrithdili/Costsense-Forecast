"""AWS Pricing API adapter — daily-USD estimates per AWS resource type.

The Pricing API returns list prices only (no RI/SP/EDP discounts). So the
delta produced here is an UPPER BOUND on the true cost impact, useful as a
directional signal in the forecast.

Results are cached on disk under data/pricing_cache.json to keep repeat runs
fast — the Pricing API is slow and rate-limited.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.aws.session import make_session


CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "pricing_cache.json"


@dataclass
class PriceEstimate:
    resource_type: str
    instance_hint: str | None
    daily_usd: float
    source: str  # "pricing-api" | "table" | "unknown"


# No fallback price table. If AWS Pricing API doesn't return a rate for a
# resource type, we mark it `unknown` and return $0. This keeps the forecast
# honest — no fabricated per-day numbers leak into predictions.
_FALLBACK: dict[str, float] = {}


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def _pricing_client(profile: str | None):
    # Not cached — temp credentials expire hourly.
    session = make_session(profile)
    # Pricing endpoint only in us-east-1 / ap-south-1
    return session.client("pricing", region_name="us-east-1")


def _query_ec2_ondemand(
    instance_type: str, profile: str | None, region: str = "us-east-1",
) -> float | None:
    """Return USD/day for an on-demand Linux EC2 of the given instance type."""
    region_name_map = {
        "us-east-1": "US East (N. Virginia)",
        "us-east-2": "US East (Ohio)",
        "us-west-2": "US West (Oregon)",
        "eu-west-1": "EU (Ireland)",
        "ap-south-1": "Asia Pacific (Mumbai)",
    }
    location = region_name_map.get(region, "US East (N. Virginia)")
    client = _pricing_client(profile)
    resp = client.get_products(
        ServiceCode="AmazonEC2",
        Filters=[
            {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
            {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
            {"Type": "TERM_MATCH", "Field": "location", "Value": location},
            {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
            {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
            {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
        ],
        MaxResults=1,
    )
    price_list = resp.get("PriceList", [])
    if not price_list:
        return None
    doc = json.loads(price_list[0])
    on_demand = doc["terms"]["OnDemand"]
    first_term = next(iter(on_demand.values()))
    first_dim = next(iter(first_term["priceDimensions"].values()))
    per_hour = float(first_dim["pricePerUnit"]["USD"])
    return per_hour * 24


def _query_nat_gateway(profile: str | None) -> float | None:
    """NAT Gateway hourly base price (traffic excluded)."""
    client = _pricing_client(profile)
    resp = client.get_products(
        ServiceCode="AmazonEC2",
        Filters=[
            {"Type": "TERM_MATCH", "Field": "productFamily", "Value": "NAT Gateway"},
            {"Type": "TERM_MATCH", "Field": "location", "Value": "US East (N. Virginia)"},
        ],
        MaxResults=1,
    )
    price_list = resp.get("PriceList", [])
    if not price_list:
        return None
    doc = json.loads(price_list[0])
    on_demand = doc["terms"]["OnDemand"]
    first_term = next(iter(on_demand.values()))
    for dim in first_term["priceDimensions"].values():
        unit = dim.get("unit", "").lower()
        if "hour" in unit or "hrs" in unit:
            return float(dim["pricePerUnit"]["USD"]) * 24
    return None


def estimate_daily_usd(
    resource_type: str,
    instance_hint: str | None = None,
    profile: str | None = None,
) -> PriceEstimate:
    cache_key = f"{resource_type}|{instance_hint or ''}"
    cache = _load_cache()
    if cache_key in cache:
        c = cache[cache_key]
        return PriceEstimate(**c)

    daily: float | None = None
    source = "unknown"

    if resource_type == "aws_instance" and instance_hint:
        try:
            daily = _query_ec2_ondemand(instance_hint, profile=profile)
            source = "pricing-api" if daily is not None else "unknown"
        except Exception:  # noqa: BLE001
            pass
    elif resource_type == "aws_nat_gateway":
        try:
            daily = _query_nat_gateway(profile=profile)
            source = "pricing-api" if daily is not None else "unknown"
        except Exception:  # noqa: BLE001
            pass

    if daily is None and resource_type in _FALLBACK:
        daily = _FALLBACK[resource_type]
        source = "table"

    if daily is None:
        daily = 0.0
        source = "unknown"

    est = PriceEstimate(
        resource_type=resource_type,
        instance_hint=instance_hint,
        daily_usd=round(daily, 4),
        source=source,
    )
    cache[cache_key] = {
        "resource_type": est.resource_type,
        "instance_hint": est.instance_hint,
        "daily_usd": est.daily_usd,
        "source": est.source,
    }
    _save_cache(cache)
    return est
