"""Read-only AWS tools exposed to the deep-AWS Bedrock agent.

Every function here is *read* only — no create/update/delete calls. The AI
can enumerate resources, pull metrics, look up events, and read AWS's own
rightsizing recommendations. It cannot mutate the account.

Each function has:
  - A plain Python implementation
  - A `TOOL_SPEC` entry consumed by the Bedrock agent to teach Claude the
    tool's shape and when to use it
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

from src.aws.session import make_session


# ---------------------------------------------------------------------------
# session / clients
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _session(profile: str | None):
    return make_session(profile)


@lru_cache(maxsize=64)
def _client(profile: str | None, service: str, region: str):
    return _session(profile).client(service, region_name=region)


def _safe(fn):
    """Wrap a tool to turn any AWS error into a structured dict rather than
    raising into the agent loop."""
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
# CloudTrail — management events
# ---------------------------------------------------------------------------

@_safe
def cloudtrail_lookup(
    event_name: str | None = None,
    resource_type: str | None = None,
    days: int = 7,
    max_results: int = 25,
    region: str = "us-east-1",
    profile: str | None = None,
) -> dict:
    """Look up recent CloudTrail management events. Useful for spotting
    console clicks that changed cost (StopDBInstance, DisableSecurityHub,
    DeleteNatGateway, PutBucketLifecycle, ...).
    """
    ct = _client(profile, "cloudtrail", region)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    attrs = []
    if event_name:
        attrs.append({"AttributeKey": "EventName", "AttributeValue": event_name})
    if resource_type:
        attrs.append({"AttributeKey": "ResourceType", "AttributeValue": resource_type})
    resp = ct.lookup_events(
        LookupAttributes=attrs,
        StartTime=start,
        EndTime=end,
        MaxResults=max_results,
    )
    events = []
    for e in resp.get("Events", []):
        events.append({
            "event_time": e["EventTime"].isoformat(),
            "event_name": e.get("EventName"),
            "username": e.get("Username"),
            "resources": [{"type": r.get("ResourceType"),
                           "name": r.get("ResourceName")}
                          for r in e.get("Resources", [])],
        })
    return {"events": events, "count": len(events), "region": region,
            "days": days}


CLOUDTRAIL_SPEC = {
    "name": "cloudtrail_lookup",
    "description": (
        "Look up recent AWS CloudTrail management events. Use this to find "
        "console changes that affected cost but aren't in git (e.g. someone "
        "disabled GuardDuty, stopped an RDS instance, deleted a NAT gateway). "
        "Filter by event_name (e.g. 'StopDBInstance', 'DisableOrganization"
        "AdminAccount', 'DeleteFunction') and/or resource_type (e.g. "
        "'AWS::RDS::DBInstance'). Default lookback is 7 days."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "event_name": {"type": "string",
                           "description": "CloudTrail event name, e.g. StopDBInstance"},
            "resource_type": {"type": "string",
                              "description": "e.g. AWS::RDS::DBInstance"},
            "days": {"type": "integer", "default": 7, "maximum": 30},
            "region": {"type": "string", "default": "us-east-1"},
            "max_results": {"type": "integer", "default": 25, "maximum": 50},
        },
    },
}


# ---------------------------------------------------------------------------
# Resource inventory
# ---------------------------------------------------------------------------

@_safe
def list_lambda_functions(
    region: str = "us-east-1", profile: str | None = None, max_items: int = 100,
) -> dict:
    lam = _client(profile, "lambda", region)
    fns = []
    paginator = lam.get_paginator("list_functions")
    for page in paginator.paginate(PaginationConfig={"MaxItems": max_items}):
        for f in page.get("Functions", []):
            fns.append({
                "name": f["FunctionName"],
                "runtime": f.get("Runtime"),
                "memory_mb": f.get("MemorySize"),
                "timeout_s": f.get("Timeout"),
                "last_modified": f.get("LastModified"),
                "code_size": f.get("CodeSize"),
            })
    return {"functions": fns, "count": len(fns), "region": region}


LAMBDA_SPEC = {
    "name": "list_lambda_functions",
    "description": ("List Lambda functions in a region with memory / timeout / "
                    "size. Use to spot oversized or unused functions."),
    "input_schema": {
        "type": "object",
        "properties": {
            "region": {"type": "string", "default": "us-east-1"},
            "max_items": {"type": "integer", "default": 100, "maximum": 500},
        },
    },
}


@_safe
def list_rds_instances(
    region: str = "us-east-1", profile: str | None = None,
) -> dict:
    rds = _client(profile, "rds", region)
    out = []
    paginator = rds.get_paginator("describe_db_instances")
    for page in paginator.paginate():
        for db in page.get("DBInstances", []):
            out.append({
                "id": db["DBInstanceIdentifier"],
                "class": db.get("DBInstanceClass"),
                "engine": db.get("Engine"),
                "status": db.get("DBInstanceStatus"),
                "storage_gb": db.get("AllocatedStorage"),
                "multi_az": db.get("MultiAZ"),
                "created": db.get("InstanceCreateTime").isoformat()
                if db.get("InstanceCreateTime") else None,
            })
    return {"instances": out, "count": len(out), "region": region}


RDS_SPEC = {
    "name": "list_rds_instances",
    "description": ("List RDS instances with class, engine, status, storage. "
                    "Use to find idle or oversized DBs."),
    "input_schema": {
        "type": "object",
        "properties": {
            "region": {"type": "string", "default": "us-east-1"},
        },
    },
}


@_safe
def list_ec2_instances(
    region: str = "us-east-1", profile: str | None = None,
) -> dict:
    ec2 = _client(profile, "ec2", region)
    out = []
    paginator = ec2.get_paginator("describe_instances")
    for page in paginator.paginate():
        for res in page.get("Reservations", []):
            for i in res.get("Instances", []):
                out.append({
                    "id": i.get("InstanceId"),
                    "type": i.get("InstanceType"),
                    "state": i.get("State", {}).get("Name"),
                    "launch_time": i.get("LaunchTime").isoformat()
                    if i.get("LaunchTime") else None,
                    "tags": {t["Key"]: t["Value"] for t in i.get("Tags", [])},
                })
    return {"instances": out, "count": len(out), "region": region}


EC2_SPEC = {
    "name": "list_ec2_instances",
    "description": ("List EC2 instances with type / state / launch time. "
                    "Use to find unused instances (state=stopped or old and "
                    "untagged)."),
    "input_schema": {
        "type": "object",
        "properties": {
            "region": {"type": "string", "default": "us-east-1"},
        },
    },
}


@_safe
def list_nat_gateways(
    region: str = "us-east-1", profile: str | None = None,
) -> dict:
    ec2 = _client(profile, "ec2", region)
    resp = ec2.describe_nat_gateways()
    out = []
    for n in resp.get("NatGateways", []):
        out.append({
            "id": n.get("NatGatewayId"),
            "state": n.get("State"),
            "vpc": n.get("VpcId"),
            "subnet": n.get("SubnetId"),
            "created": n.get("CreateTime").isoformat()
            if n.get("CreateTime") else None,
        })
    return {"nat_gateways": out, "count": len(out), "region": region}


NAT_SPEC = {
    "name": "list_nat_gateways",
    "description": ("List NAT gateways with state. Each running NAT gateway "
                    "costs ~$32/month + traffic. Use to find NATs that may "
                    "not be needed."),
    "input_schema": {
        "type": "object",
        "properties": {"region": {"type": "string", "default": "us-east-1"}},
    },
}


@_safe
def list_ebs_volumes(
    region: str = "us-east-1", profile: str | None = None,
    state: str | None = None,
) -> dict:
    ec2 = _client(profile, "ec2", region)
    filters = [{"Name": "status", "Values": [state]}] if state else []
    out = []
    paginator = ec2.get_paginator("describe_volumes")
    for page in paginator.paginate(Filters=filters):
        for v in page.get("Volumes", []):
            out.append({
                "id": v.get("VolumeId"),
                "type": v.get("VolumeType"),
                "size_gb": v.get("Size"),
                "state": v.get("State"),
                "attachments": len(v.get("Attachments", [])),
                "created": v.get("CreateTime").isoformat()
                if v.get("CreateTime") else None,
            })
    return {"volumes": out, "count": len(out), "region": region}


EBS_SPEC = {
    "name": "list_ebs_volumes",
    "description": ("List EBS volumes. Filter state='available' to find "
                    "unattached (orphan) volumes that keep charging."),
    "input_schema": {
        "type": "object",
        "properties": {
            "region": {"type": "string", "default": "us-east-1"},
            "state": {"type": "string",
                      "enum": ["in-use", "available", "creating", "deleting"]},
        },
    },
}


# ---------------------------------------------------------------------------
# Cost Explorer — recommendations
# ---------------------------------------------------------------------------

@_safe
def rightsizing_recommendations(
    profile: str | None = None, max_items: int = 25,
) -> dict:
    """AWS's own rightsizing suggestions from Cost Explorer."""
    ce = _client(profile, "ce", "us-east-1")
    resp = ce.get_rightsizing_recommendation(
        Service="AmazonEC2",
        Configuration={
            "RecommendationTarget": "SAME_INSTANCE_FAMILY",
            "BenefitsConsidered": True,
        },
        PageSize=max_items,
    )
    out = []
    for r in resp.get("RightsizingRecommendations", []):
        out.append({
            "instance": r.get("CurrentInstance", {}).get("ResourceId"),
            "current_type": r.get("CurrentInstance", {})
                .get("ResourceDetails", {})
                .get("EC2ResourceDetails", {}).get("InstanceType"),
            "recommendation": r.get("RightsizingType"),
            "estimated_monthly_savings": r.get("ModifyRecommendationDetail", {})
                .get("TargetInstances", [{}])[0]
                .get("EstimatedMonthlySavings"),
        })
    return {"recommendations": out, "count": len(out)}


RIGHTSIZING_SPEC = {
    "name": "rightsizing_recommendations",
    "description": ("AWS's own EC2 rightsizing recommendations from Cost "
                    "Explorer. Returns current instance type and suggested "
                    "target with estimated monthly savings."),
    "input_schema": {
        "type": "object",
        "properties": {
            "max_items": {"type": "integer", "default": 25, "maximum": 100},
        },
    },
}


@_safe
def cost_by_service(
    days: int = 14, profile: str | None = None,
) -> dict:
    """Daily UnblendedCost grouped by service for the last `days`."""
    ce = _client(profile, "ce", "us-east-1")
    end = date.today()
    start = end - timedelta(days=days)
    resp = ce.get_cost_and_usage(
        TimePeriod={"Start": start.isoformat(), "End": end.isoformat()},
        Granularity="DAILY",
        Metrics=["UnblendedCost"],
        GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
    )
    totals: dict[str, float] = {}
    daily: list[dict] = []
    for period in resp.get("ResultsByTime", []):
        day = period["TimePeriod"]["Start"]
        row: dict[str, Any] = {"day": day}
        for g in period.get("Groups", []):
            svc = g["Keys"][0]
            amt = float(g["Metrics"]["UnblendedCost"]["Amount"])
            row[svc] = round(amt, 2)
            totals[svc] = totals.get(svc, 0.0) + amt
        daily.append(row)
    top = sorted(totals.items(), key=lambda kv: -kv[1])[:10]
    return {
        "top_services_by_total": [{"service": s, "total_usd": round(t, 2)}
                                  for s, t in top],
        "daily": daily,
        "days": days,
    }


COST_BY_SERVICE_SPEC = {
    "name": "cost_by_service",
    "description": ("Daily UnblendedCost grouped by service for the last N "
                    "days. Use this to spot services that trended up or "
                    "collapsed to zero (which signals a config change worth "
                    "cross-referencing with CloudTrail)."),
    "input_schema": {
        "type": "object",
        "properties": {
            "days": {"type": "integer", "default": 14, "maximum": 30},
        },
    },
}


# ---------------------------------------------------------------------------
# CloudWatch — reused from the PR analyzer
# ---------------------------------------------------------------------------

# Re-export the existing tool spec + fn so the agent has one uniform surface.
from src.pr_scanner.cloudwatch_tool import (  # noqa: E402
    TOOL_SPEC as CLOUDWATCH_SPEC,
    get_metric_statistics as cloudwatch_metric,
)


# ---------------------------------------------------------------------------
# Registry the agent iterates over
# ---------------------------------------------------------------------------

PRECEDENT_SPEC: dict = {
    "name": "precedent_lookup",
    "description": (
        "Look up historical PRECEDENT cost impact for scope-expansion PRs "
        "(PRs that onboard new tenants/orgs/customers by adding entries to "
        "a whitelist/allowlist/config collection). Only call this when the "
        "diff you were given actually onboards new entries into an "
        "existing service — NOT for refactors, bug fixes, or PRs where "
        "the added lines are function arguments, imports, or logic. "
        "The tool re-analyses the current PR's diff, finds prior merged "
        "PRs in the same repo that touched the same file(s), then measures "
        "the step change in daily cost caused by those precedents in any "
        "sibling AWS account (dev/staging/prod of the same repo) whose "
        "credentials are locally reachable. Returns a measured "
        "$/entry/day rate + sample details, or an explanation of why no "
        "rate could be derived. This is the ONLY way to get a "
        "measured (not fabricated) delta when the PR affects a service "
        "that does not run in the current AWS account."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


TOOLS: dict[str, tuple[callable, dict]] = {
    "cloudtrail_lookup": (cloudtrail_lookup, CLOUDTRAIL_SPEC),
    "list_lambda_functions": (list_lambda_functions, LAMBDA_SPEC),
    "list_rds_instances": (list_rds_instances, RDS_SPEC),
    "list_ec2_instances": (list_ec2_instances, EC2_SPEC),
    "list_nat_gateways": (list_nat_gateways, NAT_SPEC),
    "list_ebs_volumes": (list_ebs_volumes, EBS_SPEC),
    "rightsizing_recommendations": (rightsizing_recommendations, RIGHTSIZING_SPEC),
    "cost_by_service": (cost_by_service, COST_BY_SERVICE_SPEC),
    "get_cloudwatch_metric": (cloudwatch_metric, CLOUDWATCH_SPEC),
    # precedent_lookup is registered via call_tool's special-case below;
    # its spec is included in tool_specs() so the LLM sees it.
}


def tool_specs() -> list[dict]:
    return [spec for _, spec in TOOLS.values()] + [PRECEDENT_SPEC]


# `precedent_lookup` needs (repo, diff), which are per-invocation context
# that the LLM shouldn't have to (and can't reliably) pass in args. The
# agent loop stashes them here right before invoking the model.
_precedent_ctx: dict = {"repo": None, "diff": None}


def set_precedent_context(repo: str | None, diff: str | None) -> None:
    _precedent_ctx["repo"] = repo
    _precedent_ctx["diff"] = diff


def _precedent_lookup_impl() -> dict:
    from src.ai_agent.precedent import find_precedents
    repo = _precedent_ctx.get("repo")
    diff = _precedent_ctx.get("diff")
    if not repo or not diff:
        return {
            "error": (
                "precedent_lookup requires PR context. This tool is only "
                "usable inside analyze_pr()."
            )
        }
    try:
        agg = find_precedents(repo, diff)
    except Exception as e:  # noqa: BLE001
        return {"error": f"precedent lookup failed: {e}"}
    return {
        "usable": agg.usable,
        "note": agg.note,
        "per_entry_daily_usd": agg.mean_per_tenant_daily_usd,
        "range_low_per_entry_daily_usd": agg.low_per_tenant_daily_usd,
        "range_high_per_entry_daily_usd": agg.high_per_tenant_daily_usd,
        "samples": [
            {
                "pr_number": s.pr_number,
                "pr_url": s.pr_url,
                "merged_at": s.merged_at,
                "entries_added": s.tenants_added,
                "sibling_profile": s.sibling_profile,
                "account_id": s.account_id,
                "step_daily_usd": s.step_daily_usd,
                "per_entry_daily_usd": s.per_tenant_daily_usd,
            }
            for s in agg.samples
        ],
    }


def call_tool(name: str, args: dict, profile: str | None) -> dict:
    """Return a dict the caller can pass to Claude. Large results are
    shrunk structurally (drop long inner arrays) rather than truncated
    at byte-N, which would produce invalid JSON."""
    if name == "precedent_lookup":
        return _shrink(_precedent_lookup_impl())
    if name not in TOOLS:
        return {"error": f"unknown tool: {name}"}
    fn, _ = TOOLS[name]
    kwargs = dict(args or {})
    kwargs["profile"] = profile
    result = fn(**kwargs)
    return _shrink(result)


def _shrink(obj, max_len: int = 40, depth: int = 0) -> object:
    """Recursively cap any list to `max_len` entries so we never blow the
    token budget on a runaway paginator output. Returns a NEW dict/list —
    always valid JSON, no mid-string truncation."""
    if depth > 6:
        return "…"
    if isinstance(obj, dict):
        return {k: _shrink(v, max_len, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > max_len:
            return [_shrink(x, max_len, depth + 1) for x in obj[:max_len]] + \
                   [f"…and {len(obj) - max_len} more"]
        return [_shrink(x, max_len, depth + 1) for x in obj]
    return obj
