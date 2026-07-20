"""Full-repo scan for cost-relevant signals.

For each selected GitHub repo, extract:
  - Infrastructure config (accounts, regions, envs) via GitHub API
  - Scheduled events (EventBridge cron rules, CloudWatch scheduled invocations)
  - Provisioned Lambda / Fargate declarations that will run continuously
  - Recently opened PRs (leading indicator of cost changes about to land)
  - Growing files or new resource declarations in the last 30 days

Kept fast: everything routed through `gh` CLI, no repo cloning.
"""
from __future__ import annotations

import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache


def _run(args: list[str]) -> str:
    r = subprocess.run(args, capture_output=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode("utf-8", errors="replace").strip())
    return r.stdout.decode("utf-8", errors="replace")


@dataclass
class RepoSweepResult:
    repo: str
    accounts: dict[str, str] = field(default_factory=dict)   # env -> account_id
    open_prs: list[dict] = field(default_factory=list)
    recent_iac_files: list[str] = field(default_factory=list)
    scheduled_rules: list[str] = field(default_factory=list)
    error: str | None = None


@lru_cache(maxsize=64)
def _fetch_config(repo: str) -> dict[str, str]:
    """Read infrastructure/config.json if present. Returns {env: account_id}."""
    for path in ("infrastructure/config.json", "cdk/infrastructure/config.json"):
        try:
            raw = _run(["gh", "api", f"repos/{repo}/contents/{path}"])
        except RuntimeError:
            continue
        try:
            import base64
            content = json.loads(raw).get("content", "")
            text = base64.b64decode(content).decode("utf-8", errors="replace")
            doc = json.loads(text)
        except Exception:  # noqa: BLE001
            continue
        aws_accounts = (doc.get("account") or {}).get("aws") or {}
        return {k: str(v) for k, v in aws_accounts.items() if v}
    return {}


def _open_prs(repo: str, limit: int = 15) -> list[dict]:
    """Recently updated open PRs. These are the leading indicator of what
    might merge and shift cost next."""
    try:
        raw = _run([
            "gh", "pr", "list", "--repo", repo,
            "--state", "open", "--limit", str(limit),
            "--json", "number,title,author,updatedAt,url,additions,deletions",
        ])
        return json.loads(raw)
    except RuntimeError:
        return []


def _recent_iac_files(repo: str, days: int = 30, limit: int = 30) -> list[str]:
    """List infrastructure-y files touched in the last N days. Uses the
    commits API to walk recent changes."""
    since = (date.today() - timedelta(days=days)).isoformat()
    try:
        raw = _run([
            "gh", "api",
            f"repos/{repo}/commits?since={since}T00:00:00Z&per_page=30",
        ])
        commits = json.loads(raw)
    except RuntimeError:
        return []

    # For each commit, grab its files; keep only IaC / infra-relevant paths.
    iac_pat = re.compile(
        r"(\.tf$|\.tf\.|cdk\.json$|serverless\.yml$|"
        r"infrastructure/|cloudformation|template\.ya?ml$|"
        r"config\.json$|\.terraform)"
    )
    seen: set[str] = set()
    for c in commits[:15]:
        sha = c.get("sha")
        if not sha:
            continue
        try:
            det = json.loads(_run(["gh", "api", f"repos/{repo}/commits/{sha}"]))
            for f in det.get("files", []) or []:
                p = f.get("filename", "")
                if iac_pat.search(p):
                    seen.add(p)
                    if len(seen) >= limit:
                        return sorted(seen)
        except RuntimeError:
            continue
    return sorted(seen)


def _scheduled_rules(repo: str) -> list[str]:
    """Search the repo for EventBridge / cron declarations. These commit the
    account to recurring compute."""
    try:
        raw = _run([
            "gh", "api",
            f"search/code?q=repo:{repo}+Schedule.rate+OR+Schedule.cron"
            "+OR+aws_cloudwatch_event_rule&per_page=15",
        ])
        data = json.loads(raw)
    except RuntimeError:
        return []
    return [item.get("path", "") for item in data.get("items", [])][:15]


def sweep_repo(repo: str) -> RepoSweepResult:
    result = RepoSweepResult(repo=repo)
    try:
        result.accounts = dict(_fetch_config(repo))
        result.open_prs = _open_prs(repo)
        result.recent_iac_files = _recent_iac_files(repo)
        result.scheduled_rules = _scheduled_rules(repo)
    except Exception as e:  # noqa: BLE001
        result.error = str(e)
    return result


def sweep_repos(repos: list[str], max_workers: int = 4) -> list[RepoSweepResult]:
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(sweep_repo, repos))


def sweep_to_summary(results: list[RepoSweepResult]) -> dict:
    """Compact summary passed to the LLM as pre-computed context."""
    return {
        "repos": [
            {
                "repo": r.repo,
                "accounts_by_env": r.accounts,
                "open_pr_count": len(r.open_prs),
                "open_pr_titles": [p.get("title", "") for p in r.open_prs[:8]],
                "recent_iac_files": r.recent_iac_files[:15],
                "scheduled_rule_files": r.scheduled_rules[:10],
                "error": r.error,
            }
            for r in results
        ],
    }
