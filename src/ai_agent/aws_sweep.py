"""Full-AWS sweep for anomaly / recommendation surface.

Pre-fetches a compact picture of the account:
  - Cost by service (14 days daily)
  - Compute Optimizer EC2 + Lambda recommendations
  - Idle EBS volumes (available state)
  - NAT gateways
  - RDS instances
  - Rightsizing recommendations
  - Budgets

Everything runs in parallel and results are passed to the anomaly LLM as
grounded context. The LLM only needs to *reason* over this, not fetch it.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from src.ai_agent.aws_tools import call_tool
from src.ai_agent.aws_tools_broad import scrub


SWEEP_TASKS = [
    ("cost_by_service", {"days": 14}),
    ("list_nat_gateways", {}),
    ("list_ebs_volumes", {"state": "available"}),
    ("list_rds_instances", {}),
    ("list_lambda_functions", {"max_items": 50}),
    ("rightsizing_recommendations", {"max_items": 15}),
]

BROAD_SWEEP_TASKS = [
    ("compute_optimizer_ec2", {"max_items": 15}),
    ("compute_optimizer_lambda", {"max_items": 30}),
    ("list_budgets", {}),
    ("list_s3_buckets", {}),
    ("list_dynamodb_tables", {}),
]


def _call(name: str, args: dict, profile: str | None) -> tuple[str, dict]:
    try:
        result = call_tool(name, args, profile)
    except Exception as e:  # noqa: BLE001
        result = {"error": f"tool crashed: {e}"}
    return name, scrub(result)


def _call_broad(name: str, args: dict, profile: str | None) -> tuple[str, dict]:
    from src.ai_agent.aws_tools_broad import BROAD_TOOLS
    fn, _ = BROAD_TOOLS[name]
    kwargs = dict(args)
    kwargs["profile"] = profile
    try:
        result = fn(**kwargs)
    except Exception as e:  # noqa: BLE001
        result = {"error": f"tool crashed: {e}"}
    return name, scrub(result)


def sweep_account(profile: str | None) -> dict[str, dict]:
    """Run every sweep task in parallel. Returns {tool_name: result}."""
    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = []
        for name, args in SWEEP_TASKS:
            futures.append(pool.submit(_call, name, args, profile))
        for name, args in BROAD_SWEEP_TASKS:
            futures.append(pool.submit(_call_broad, name, args, profile))
        for f in as_completed(futures):
            try:
                name, result = f.result()
                out[name] = result
            except Exception as e:  # noqa: BLE001
                pass
    return out


def sweep_to_summary(sweep: dict[str, dict], max_items: int = 20) -> dict:
    """Trim the sweep to a compact summary the LLM can consume."""
    summary: dict = {}

    cbs = sweep.get("cost_by_service", {})
    summary["top_services"] = cbs.get("top_services_by_total", [])[:10]
    daily = cbs.get("daily", [])
    if daily:
        summary["cost_last_14d_sample"] = daily[-7:]

    nats = sweep.get("list_nat_gateways", {}).get("nat_gateways", [])
    active = [n for n in nats if n.get("state") == "available"]
    summary["nat_gateways"] = {
        "active_count": len(active),
        "cost_estimate_per_day": round(len(active) * 1.05, 2),  # ~$32/mo each
        "sample": active[:5],
    }

    ebs = sweep.get("list_ebs_volumes", {}).get("volumes", [])
    summary["idle_ebs_volumes"] = {
        "count": len(ebs),
        "total_gb": sum(v.get("size_gb", 0) or 0 for v in ebs),
        "monthly_cost_estimate": round(
            sum(v.get("size_gb", 0) or 0 for v in ebs) * 0.10, 2
        ),
        "sample": ebs[:8],
    }

    rds = sweep.get("list_rds_instances", {}).get("instances", [])
    summary["rds_instances"] = {
        "count": len(rds),
        "sample": rds[:8],
    }

    co_ec2 = sweep.get("compute_optimizer_ec2", {}).get("recommendations", [])
    summary["ec2_rightsize"] = {
        "count": len(co_ec2),
        "monthly_savings_total": sum(
            float(r.get("monthly_savings_usd") or 0) for r in co_ec2
        ),
        "top": co_ec2[:5],
    }

    co_lam = sweep.get("compute_optimizer_lambda", {}).get("recommendations", [])
    summary["lambda_rightsize"] = {
        "count": len(co_lam),
        "monthly_savings_total": sum(
            float(r.get("monthly_savings_usd") or 0) for r in co_lam
        ),
        "top": co_lam[:5],
    }

    rs = sweep.get("rightsizing_recommendations", {}).get("recommendations", [])
    summary["ce_rightsizing"] = rs[:5]

    b = sweep.get("list_budgets", {}).get("budgets", [])
    summary["budgets"] = b[:5]

    s3 = sweep.get("list_s3_buckets", {}).get("buckets", [])
    summary["s3_bucket_count"] = len(s3)

    ddb = sweep.get("list_dynamodb_tables", {}).get("tables", [])
    summary["dynamodb_tables"] = ddb[:8]

    return summary
