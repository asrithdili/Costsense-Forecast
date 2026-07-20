"""Scan merged PRs on a given base branch across N repos.

Uses `gh` CLI (already authenticated). For each PR we extract added/removed
AWS resources from its unified diff — regex over Terraform `resource "aws_X"`
declarations and CDK `new X.Y(...)` calls that reference AWS constructs.

Config changes (attribute-only tweaks) show up as `modify` with no cost delta
until the pricing layer learns to compare instance types etc.
"""
from __future__ import annotations

import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterable


# ---------- data classes ----------

@dataclass
class ResourceChange:
    kind: str            # "terraform" | "cdk"
    resource_type: str   # e.g. "aws_instance", "aws_lambda_function"
    resource_name: str   # logical name from the code
    action: str          # "add" | "remove" | "modify"
    instance_hint: str | None = None  # e.g. "t3.micro" if we can spot it


@dataclass
class PrRecord:
    number: int
    title: str
    repo: str
    author: str
    merged_at: str
    url: str
    changes: list[ResourceChange] = field(default_factory=list)


# ---------- regexes ----------

_TF_RESOURCE_RE = re.compile(
    r'^([+-])\s*resource\s+"(aws_[a-zA-Z0-9_]+)"\s+"([a-zA-Z0-9_\-]+)"'
)
_TF_INSTANCE_TYPE_RE = re.compile(r'instance_type\s*=\s*"([^"]+)"')

# CDK: `new aws_ec2.Instance(...)` or `new Instance(...)` — capture the class
# We also match `new (aws_)?lambda(.Function)?` broadly then classify.
_CDK_NEW_RE = re.compile(
    r'^([+-])\s*.*\bnew\s+(?:aws_[a-zA-Z0-9_]+\.)?([A-Z][a-zA-Z0-9]+)\s*\('
)

# CDK class name → AWS resource type used by the Pricing API mapping.
# Small starter set; expand as we see more.
_CDK_CLASS_TO_RESOURCE = {
    "Function": "aws_lambda_function",
    "Instance": "aws_instance",
    "Bucket": "aws_s3_bucket",
    "Table": "aws_dynamodb_table",
    "Queue": "aws_sqs_queue",
    "Topic": "aws_sns_topic",
    "DatabaseInstance": "aws_rds_instance",
    "DatabaseCluster": "aws_rds_cluster",
    "NatGateway": "aws_nat_gateway",
    "Distribution": "aws_cloudfront_distribution",
    "Cluster": "aws_ecs_cluster",
    "FargateService": "aws_ecs_service",
    "LoadBalancer": "aws_lb",
    "ApplicationLoadBalancer": "aws_lb",
    "NetworkLoadBalancer": "aws_lb",
    "UserPool": "aws_cognito_user_pool",
    "RestApi": "aws_api_gateway_rest_api",
    "HttpApi": "aws_apigatewayv2_api",
    "LogGroup": "aws_cloudwatch_log_group",
    "StateMachine": "aws_sfn_state_machine",
    "Stream": "aws_kinesis_stream",
    "Rule": "aws_cloudwatch_event_rule",
}


# ---------- diff parsing ----------

def _extract_terraform_changes(diff: str) -> list[ResourceChange]:
    adds: dict[tuple[str, str], ResourceChange] = {}
    removes: dict[tuple[str, str], ResourceChange] = {}
    # capture instance_type per resource by scanning nearby lines
    lines = diff.splitlines()
    for i, line in enumerate(lines):
        m = _TF_RESOURCE_RE.match(line)
        if not m:
            continue
        sign, rtype, rname = m.group(1), m.group(2), m.group(3)
        instance_hint = None
        # look ahead ~30 lines within the same resource block for an
        # `instance_type = "..."` — cheap heuristic, no HCL parser needed
        for lookahead in lines[i + 1 : i + 40]:
            if lookahead.startswith(("+", "-")) and lookahead[1:].lstrip().startswith("resource "):
                break
            it = _TF_INSTANCE_TYPE_RE.search(lookahead)
            if it:
                instance_hint = it.group(1)
                break
        rc = ResourceChange(
            kind="terraform", resource_type=rtype, resource_name=rname,
            action="add" if sign == "+" else "remove",
            instance_hint=instance_hint,
        )
        (adds if sign == "+" else removes)[(rtype, rname)] = rc

    out: list[ResourceChange] = []
    for key, add in adds.items():
        if key in removes:
            # same block edited — treat as modify (unless instance type changed;
            # pricing layer can still price a type swap by seeing both hints)
            add.action = "modify"
            out.append(add)
        else:
            out.append(add)
    for key, rem in removes.items():
        if key in adds:
            continue
        out.append(rem)
    return out


def _extract_cdk_changes(diff: str) -> list[ResourceChange]:
    seen: dict[tuple[str, str, str], ResourceChange] = {}
    for line in diff.splitlines():
        m = _CDK_NEW_RE.match(line)
        if not m:
            continue
        sign, cls = m.group(1), m.group(2)
        rtype = _CDK_CLASS_TO_RESOURCE.get(cls)
        if not rtype:
            continue
        # CDK diffs don't give us a logical name easily; use the class + a
        # positional counter as a stand-in so repeated `new Function(...)`
        # calls don't collapse to one row.
        key = (sign, rtype, cls)
        if key in seen:
            continue
        seen[key] = ResourceChange(
            kind="cdk", resource_type=rtype, resource_name=cls,
            action="add" if sign == "+" else "remove",
        )
    return list(seen.values())


def extract_changes(diff: str) -> list[ResourceChange]:
    return _extract_terraform_changes(diff) + _extract_cdk_changes(diff)


# ---------- gh CLI adapters ----------

def _run(args: list[str]) -> str:
    r = subprocess.run(args, capture_output=True, check=False)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"gh failed ({args}): {err}")
    return r.stdout.decode("utf-8", errors="replace")


def list_recent_merged_prs(
    repo: str, base: str, since: date, limit: int = 50,
) -> list[dict]:
    """Return PR summaries (JSON) merged into `base` on or after `since`."""
    query = f"is:merged base:{base} merged:>={since.isoformat()}"
    out = _run([
        "gh", "pr", "list",
        "--repo", repo,
        "--state", "merged",
        "--base", base,
        "--search", query,
        "--limit", str(limit),
        "--json", "number,title,author,mergedAt,url",
    ])
    return json.loads(out or "[]")


def pr_diff(repo: str, number: int) -> str:
    return _run(["gh", "pr", "diff", str(number), "--repo", repo])


def _scan_single_pr(repo: str, pr: dict, keep_empty: bool = False) -> PrRecord | None:
    try:
        diff = pr_diff(repo, pr["number"])
    except RuntimeError:
        return None
    changes = extract_changes(diff)
    if not changes and not keep_empty:
        return None
    return PrRecord(
        number=pr["number"],
        title=pr["title"],
        repo=repo,
        author=(pr.get("author") or {}).get("login", ""),
        merged_at=pr["mergedAt"],
        url=pr["url"],
        changes=changes,
    )


def scan_repo(
    repo: str, base: str, lookback_days: int = 14, limit: int = 50,
    max_workers: int = 8, keep_empty: bool = False,
) -> list[PrRecord]:
    """If keep_empty=True, return every merged PR (even non-IaC) so the caller
    can hand them to an LLM that reads the raw diff."""
    since = date.today() - timedelta(days=lookback_days)
    prs = list_recent_merged_prs(repo, base=base, since=since, limit=limit)
    if not prs:
        return []
    out: list[PrRecord] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_scan_single_pr, repo, pr, keep_empty) for pr in prs]
        for f in as_completed(futures):
            rec = f.result()
            if rec is not None:
                out.append(rec)
    return out


def scan_repos(
    repos: Iterable[str], base: str, lookback_days: int = 14,
    keep_empty: bool = False,
) -> list[PrRecord]:
    all_prs: list[PrRecord] = []
    for repo in repos:
        try:
            all_prs.extend(scan_repo(
                repo, base=base, lookback_days=lookback_days,
                keep_empty=keep_empty,
            ))
        except RuntimeError:
            continue
    return all_prs
