"""CloudWatch metric-statistics wrapper exposed to Claude as a Bedrock tool.

Claude asks for a metric when it needs runtime context to price a config
change (e.g. Lambda memory 10240 -> 4096 requires knowing daily invocations
and average duration to compute the $ delta).

Kept read-only. Failures return an empty result rather than raising, so the
LLM can gracefully fall back to a $0 estimate.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any

import boto3


DEFAULT_METRICS_REGION = "us-east-1"


@dataclass
class MetricPoint:
    timestamp: str
    value: float


@lru_cache(maxsize=8)
def _cw_client(profile: str | None, region: str):
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return session.client("cloudwatch", region_name=region)


def get_metric_statistics(
    namespace: str,
    metric_name: str,
    dimensions: list[dict[str, str]] | None = None,
    days: int = 7,
    statistic: str = "Sum",
    period: int = 86400,  # 1 day
    profile: str | None = None,
    region: str = DEFAULT_METRICS_REGION,
) -> dict[str, Any]:
    """Return per-day metric values for the last `days`.

    Returns {"points": [MetricPoint...], "unit": str, "summary": {...}}.
    Errors return {"error": str}.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    try:
        resp = _cw_client(profile, region).get_metric_statistics(
            Namespace=namespace,
            MetricName=metric_name,
            Dimensions=dimensions or [],
            StartTime=start,
            EndTime=end,
            Period=period,
            Statistics=[statistic],
        )
    except Exception as e:  # noqa: BLE001
        return {"error": f"cloudwatch failed: {e}"}

    points = sorted(resp.get("Datapoints", []), key=lambda p: p["Timestamp"])
    out_points = [
        {"timestamp": p["Timestamp"].isoformat(), "value": float(p[statistic])}
        for p in points
    ]
    unit = points[0]["Unit"] if points else ""

    values = [p["value"] for p in out_points]
    summary = {
        "count": len(values),
        "sum": round(sum(values), 4) if values else 0.0,
        "mean": round(sum(values) / len(values), 4) if values else 0.0,
        "min": round(min(values), 4) if values else 0.0,
        "max": round(max(values), 4) if values else 0.0,
    }

    return {"points": out_points, "unit": unit, "summary": summary,
            "namespace": namespace, "metric_name": metric_name,
            "region": region, "days": days, "statistic": statistic}


# The JSON schema handed to Bedrock so Claude knows how to call this tool.
TOOL_SPEC = {
    "name": "get_cloudwatch_metric",
    "description": (
        "Fetch daily CloudWatch metric statistics for a resource to size "
        "the cost impact of a code change. Use this when a PR changes "
        "Lambda memory, ECS task counts, RDS instance class, provisioned "
        "concurrency, or any other config whose $ impact depends on how "
        "much the resource is actually used. Returns per-day values plus "
        "a summary (sum/mean/min/max). "
        "Common queries: "
        "  - Lambda invocations: namespace='AWS/Lambda', metric_name='Invocations', "
        "    dimensions=[{'Name':'FunctionName','Value':<name>}], statistic='Sum'. "
        "  - Lambda duration ms: namespace='AWS/Lambda', metric_name='Duration', "
        "    statistic='Average'. "
        "  - ECS running tasks: namespace='AWS/ECS', metric_name='CPUUtilization', "
        "    dimensions=[{'Name':'ServiceName',...},{'Name':'ClusterName',...}]. "
        "  - RDS CPU: namespace='AWS/RDS', metric_name='CPUUtilization', "
        "    dimensions=[{'Name':'DBInstanceIdentifier','Value':<id>}]. "
        "If the resource name in the PR is a construct id (CDK) or logical name "
        "(Terraform) not a real ARN, do your best guess based on typical naming."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "namespace": {"type": "string",
                          "description": "e.g. AWS/Lambda, AWS/ECS, AWS/RDS"},
            "metric_name": {"type": "string",
                            "description": "e.g. Invocations, Duration, CPUUtilization"},
            "dimensions": {
                "type": "array",
                "description": "List of {Name, Value} pairs",
                "items": {
                    "type": "object",
                    "properties": {
                        "Name": {"type": "string"},
                        "Value": {"type": "string"},
                    },
                    "required": ["Name", "Value"],
                },
                "default": [],
            },
            "statistic": {"type": "string",
                          "enum": ["Sum", "Average", "Maximum", "Minimum",
                                   "SampleCount"],
                          "default": "Sum"},
            "days": {"type": "integer", "default": 7,
                     "description": "How many days back to fetch (max 30)."},
            "region": {"type": "string", "default": DEFAULT_METRICS_REGION},
        },
        "required": ["namespace", "metric_name"],
    },
}
