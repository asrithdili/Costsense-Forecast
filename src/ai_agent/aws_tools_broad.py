"""Broader read-only AWS tools for the chat bot.

Complements `aws_tools` with Compute Optimizer, Trusted Advisor, Budgets,
Config, Service Quotas, S3 bucket listing, and DynamoDB listing.

Every tool goes through `SecretScrubber` to strip anything that could reveal
credentials, secret strings, or IAM policy JSON before Claude sees it.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from botocore.exceptions import BotoCoreError, ClientError

from src.aws.session import make_session


@lru_cache(maxsize=64)
def _client(profile: str | None, service: str, region: str):
    session = make_session(profile)
    return session.client(service, region_name=region)


def _safe(fn):
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (BotoCoreError, ClientError) as e:
            return {"error": f"{type(e).__name__}: {e}"}
        except Exception as e:  # noqa: BLE001
            return {"error": f"unexpected: {e}"}
    wrapped.__name__ = fn.__name__
    return wrapped


# ---------------------------------------------------------------------------
# Secret scrubber — recursive, applied to every tool's output
# ---------------------------------------------------------------------------

# Keys anywhere in a returned dict that look like they'd hold a secret.
_SECRET_KEY_PATTERN = re.compile(
    r"(secret|password|passwd|apikey|api_key|token|credential|"
    r"privatekey|private_key|access_key|accesskey|salt|"
    r"connection_?string|conn_?str|dsn|policy_?document|"
    r"assume_?role_?policy_?document)",
    re.IGNORECASE,
)

# Regexes for value patterns that look like credentials, even under harmless keys.
_VALUE_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),                      # AWS access key
    re.compile(r"ASIA[0-9A-Z]{16}"),                      # AWS temp key
    re.compile(r"aws_secret_access_key\s*=\s*\S+"),
    re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----"),
    re.compile(r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}"),  # JWT
]


def _scrub(obj, depth: int = 0):
    if depth > 8:
        return "…"
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if _SECRET_KEY_PATTERN.search(str(k)):
                out[k] = "[REDACTED-BY-COSTSENSE]"
            else:
                out[k] = _scrub(v, depth + 1)
        return out
    if isinstance(obj, list):
        return [_scrub(x, depth + 1) for x in obj]
    if isinstance(obj, str):
        s = obj
        for pat in _VALUE_PATTERNS:
            s = pat.sub("[REDACTED-BY-COSTSENSE]", s)
        return s
    return obj


# ---------------------------------------------------------------------------
# Compute Optimizer
# ---------------------------------------------------------------------------

@_safe
def compute_optimizer_ec2(profile: str | None = None,
                          region: str = "us-east-1",
                          max_items: int = 20) -> dict:
    """AWS Compute Optimizer EC2 rightsizing recommendations. Free service."""
    co = _client(profile, "compute-optimizer", region)
    resp = co.get_ec2_instance_recommendations(maxResults=max_items)
    out = []
    for r in resp.get("instanceRecommendations", []):
        finding = r.get("finding", "")
        options = r.get("recommendationOptions", [])
        best = options[0] if options else {}
        out.append({
            "instance_arn": r.get("instanceArn"),
            "instance_name": r.get("instanceName"),
            "current_type": r.get("currentInstanceType"),
            "finding": finding,
            "recommended_type": best.get("instanceType"),
            "monthly_savings_usd": (
                best.get("savingsOpportunity", {})
                    .get("estimatedMonthlySavings", {})
                    .get("value")
            ),
            "utilization_max_cpu": (
                r.get("utilizationMetrics", [{}])[0].get("value")
                if r.get("utilizationMetrics") else None
            ),
        })
    return {"recommendations": out, "count": len(out)}


COMPUTE_OPT_EC2_SPEC = {
    "name": "compute_optimizer_ec2",
    "description": (
        "AWS Compute Optimizer's own EC2 rightsizing recommendations. "
        "Returns underutilized instances, their current type, recommended "
        "type, and estimated monthly savings."
    ),
    "input_schema": {"type": "object", "properties": {
        "region": {"type": "string", "default": "us-east-1"},
        "max_items": {"type": "integer", "default": 20, "maximum": 100},
    }},
}


@_safe
def compute_optimizer_lambda(profile: str | None = None,
                             region: str = "us-east-1",
                             max_items: int = 30) -> dict:
    co = _client(profile, "compute-optimizer", region)
    resp = co.get_lambda_function_recommendations(maxResults=max_items)
    out = []
    for r in resp.get("lambdaFunctionRecommendations", []):
        options = r.get("memorySizeRecommendationOptions", [])
        best = options[0] if options else {}
        out.append({
            "function_arn": r.get("functionArn"),
            "current_memory_mb": r.get("currentMemorySize"),
            "recommended_memory_mb": best.get("memorySize"),
            "finding": r.get("finding"),
            "monthly_savings_usd": (
                best.get("savingsOpportunity", {})
                    .get("estimatedMonthlySavings", {}).get("value")
            ),
        })
    return {"recommendations": out, "count": len(out)}


COMPUTE_OPT_LAMBDA_SPEC = {
    "name": "compute_optimizer_lambda",
    "description": ("Compute Optimizer's Lambda memory-size recommendations. "
                    "Returns under- or over-provisioned functions with "
                    "recommended memory + monthly savings."),
    "input_schema": {"type": "object", "properties": {
        "region": {"type": "string", "default": "us-east-1"},
        "max_items": {"type": "integer", "default": 30, "maximum": 100},
    }},
}


# ---------------------------------------------------------------------------
# AWS Budgets
# ---------------------------------------------------------------------------

@_safe
def list_budgets(profile: str | None = None) -> dict:
    session = make_session(profile)
    sts = session.client("sts")
    account_id = sts.get_caller_identity()["Account"]
    b = session.client("budgets", region_name="us-east-1")
    resp = b.describe_budgets(AccountId=account_id, MaxResults=20)
    out = []
    for bg in resp.get("Budgets", []):
        limit = bg.get("BudgetLimit", {})
        actual = bg.get("CalculatedSpend", {}).get("ActualSpend", {})
        forecast = bg.get("CalculatedSpend", {}).get("ForecastedSpend", {})
        out.append({
            "name": bg.get("BudgetName"),
            "type": bg.get("BudgetType"),
            "limit_usd": float(limit.get("Amount")) if limit.get("Amount") else None,
            "actual_usd": float(actual.get("Amount")) if actual.get("Amount") else None,
            "forecast_usd": float(forecast.get("Amount")) if forecast.get("Amount") else None,
            "time_unit": bg.get("TimeUnit"),
        })
    return {"budgets": out, "count": len(out)}


BUDGETS_SPEC = {
    "name": "list_budgets",
    "description": ("List AWS Budgets on this account with current spend vs "
                    "limit and forecast. Useful to spot budgets already breached."),
    "input_schema": {"type": "object", "properties": {}},
}


# ---------------------------------------------------------------------------
# Service Quotas — flags near-limit resources
# ---------------------------------------------------------------------------

@_safe
def service_quotas_summary(profile: str | None = None,
                           region: str = "us-east-1",
                           service_code: str = "ec2") -> dict:
    sq = _client(profile, "service-quotas", region)
    resp = sq.list_service_quotas(ServiceCode=service_code, MaxResults=50)
    out = []
    for q in resp.get("Quotas", []):
        out.append({
            "quota_name": q.get("QuotaName"),
            "value": q.get("Value"),
            "unit": q.get("Unit"),
            "adjustable": q.get("Adjustable"),
        })
    return {"quotas": out, "count": len(out), "service": service_code}


SERVICE_QUOTAS_SPEC = {
    "name": "service_quotas_summary",
    "description": ("List service quotas for a given service (ec2, lambda, "
                    "rds, s3, sqs, ...). Useful when the bot suspects a "
                    "resource is limited by quota."),
    "input_schema": {"type": "object", "properties": {
        "region": {"type": "string", "default": "us-east-1"},
        "service_code": {"type": "string", "default": "ec2"},
    }, "required": ["service_code"]},
}


# ---------------------------------------------------------------------------
# S3 bucket listing (metadata only, no content)
# ---------------------------------------------------------------------------

@_safe
def list_s3_buckets(profile: str | None = None) -> dict:
    session = make_session(profile)
    s3 = session.client("s3")
    resp = s3.list_buckets()
    buckets = []
    for b in resp.get("Buckets", []):
        buckets.append({
            "name": b["Name"],
            "created": b["CreationDate"].isoformat()
            if b.get("CreationDate") else None,
        })
    return {"buckets": buckets, "count": len(buckets)}


S3_SPEC = {
    "name": "list_s3_buckets",
    "description": ("List S3 buckets on the account (name + creation date "
                    "only, no contents). Use to check for old / unused buckets."),
    "input_schema": {"type": "object", "properties": {}},
}


@_safe
def s3_bucket_lifecycle(bucket: str, profile: str | None = None) -> dict:
    session = make_session(profile)
    s3 = session.client("s3")
    try:
        resp = s3.get_bucket_lifecycle_configuration(Bucket=bucket)
        rules = []
        for r in resp.get("Rules", []):
            rules.append({
                "id": r.get("ID"),
                "status": r.get("Status"),
                "prefix": (r.get("Filter") or {}).get("Prefix", ""),
                "transitions": [{"days": t.get("Days"),
                                 "storage_class": t.get("StorageClass")}
                                for t in r.get("Transitions", [])],
                "expiration_days": (r.get("Expiration") or {}).get("Days"),
            })
        return {"bucket": bucket, "rules": rules, "count": len(rules)}
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "NoSuchLifecycleConfiguration":
            return {"bucket": bucket, "rules": [], "note": "no lifecycle policy"}
        raise


S3_LIFECYCLE_SPEC = {
    "name": "s3_bucket_lifecycle",
    "description": ("Get S3 bucket lifecycle configuration (transitions, "
                    "expirations). Buckets without lifecycle policies "
                    "silently accumulate storage cost."),
    "input_schema": {"type": "object", "properties": {
        "bucket": {"type": "string"},
    }, "required": ["bucket"]},
}


# ---------------------------------------------------------------------------
# DynamoDB
# ---------------------------------------------------------------------------

@_safe
def list_dynamodb_tables(profile: str | None = None,
                         region: str = "us-east-1") -> dict:
    d = _client(profile, "dynamodb", region)
    tables = []
    paginator = d.get_paginator("list_tables")
    names: list[str] = []
    for page in paginator.paginate():
        names.extend(page.get("TableNames", []))
    for name in names[:30]:
        info = d.describe_table(TableName=name).get("Table", {})
        tables.append({
            "name": name,
            "status": info.get("TableStatus"),
            "item_count": info.get("ItemCount"),
            "size_bytes": info.get("TableSizeBytes"),
            "billing_mode": (info.get("BillingModeSummary") or {})
                .get("BillingMode"),
            "provisioned_read": (info.get("ProvisionedThroughput") or {})
                .get("ReadCapacityUnits"),
            "provisioned_write": (info.get("ProvisionedThroughput") or {})
                .get("WriteCapacityUnits"),
        })
    return {"tables": tables, "count": len(tables), "region": region}


DDB_SPEC = {
    "name": "list_dynamodb_tables",
    "description": ("List DynamoDB tables with size / item count / billing "
                    "mode. Use to spot large tables and tables using "
                    "provisioned billing (potentially oversized)."),
    "input_schema": {"type": "object", "properties": {
        "region": {"type": "string", "default": "us-east-1"},
    }},
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BROAD_TOOLS: dict[str, tuple[callable, dict]] = {
    "compute_optimizer_ec2": (compute_optimizer_ec2, COMPUTE_OPT_EC2_SPEC),
    "compute_optimizer_lambda": (compute_optimizer_lambda, COMPUTE_OPT_LAMBDA_SPEC),
    "list_budgets": (list_budgets, BUDGETS_SPEC),
    "service_quotas_summary": (service_quotas_summary, SERVICE_QUOTAS_SPEC),
    "list_s3_buckets": (list_s3_buckets, S3_SPEC),
    "s3_bucket_lifecycle": (s3_bucket_lifecycle, S3_LIFECYCLE_SPEC),
    "list_dynamodb_tables": (list_dynamodb_tables, DDB_SPEC),
}


def all_broad_specs() -> list[dict]:
    return [spec for _, spec in BROAD_TOOLS.values()]


def scrub(obj):
    """Public entry: apply the secret redactor to any object."""
    return _scrub(obj)
