"""Scan OPEN PRs (not yet merged) and pre-price them.

Merged PRs tell you what already happened. Open PRs tell you what's about to
happen — if approved and passing CI, they'll typically merge within 1-7 days.
Pre-pricing them lets the forecast shift before the merge event.

Each open PR gets:
  - Deep LLM analysis (same as merged PRs — extract resources, call CloudWatch)
  - A `merge_probability` (0.0-1.0) inferred from state signals
  - An `expected_daily_delta_usd` = est_daily_delta × merge_probability
  - An `expected_merge_day` (best guess based on PR age + state)
"""
from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Iterable


def _run(args: list[str]) -> str:
    r = subprocess.run(args, capture_output=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode("utf-8", errors="replace").strip())
    return r.stdout.decode("utf-8", errors="replace")


@dataclass
class OpenPr:
    repo: str
    number: int
    title: str
    author: str
    url: str
    is_draft: bool
    created_at: str
    updated_at: str
    review_state: str          # APPROVED | CHANGES_REQUESTED | REVIEW_REQUIRED | ""
    mergeable: str             # MERGEABLE | CONFLICTING | UNKNOWN
    checks_state: str          # SUCCESS | PENDING | FAILURE | ""
    additions: int = 0
    deletions: int = 0
    days_open: int = 0


@dataclass
class PricedOpenPr:
    open_pr: OpenPr
    est_daily_delta_usd: float           # what THIS PR would cost/save if merged as-is
    direction: str                       # increase | decrease | neutral
    llm_summary: str
    merge_probability: float             # 0.0 - 1.0
    expected_merge_day: str              # ISO date
    expected_daily_delta_usd: float      # est × probability
    findings: list[dict] = field(default_factory=list)


def list_open_prs(repo: str, limit: int = 20) -> list[OpenPr]:
    """Fetch open PRs on a repo with the metadata we need to price them."""
    try:
        raw = _run([
            "gh", "pr", "list",
            "--repo", repo,
            "--state", "open",
            "--limit", str(limit),
            "--json",
            "number,title,author,url,isDraft,createdAt,updatedAt,"
            "reviewDecision,mergeable,statusCheckRollup,additions,deletions",
        ])
        data = json.loads(raw)
    except Exception:  # noqa: BLE001
        return []

    now = datetime.now(timezone.utc)
    out: list[OpenPr] = []
    for pr in data:
        # checks rollup: derive one aggregate from the list
        checks = pr.get("statusCheckRollup") or []
        states = {c.get("state") or c.get("conclusion") for c in checks}
        if states & {"FAILURE", "FAILED", "ERROR"}:
            checks_state = "FAILURE"
        elif states & {"PENDING", "IN_PROGRESS", "QUEUED"}:
            checks_state = "PENDING"
        elif states & {"SUCCESS", "PASSED"}:
            checks_state = "SUCCESS"
        else:
            checks_state = ""

        try:
            created = datetime.fromisoformat(pr["createdAt"].replace("Z", "+00:00"))
            days_open = (now - created).days
        except Exception:
            days_open = 0

        out.append(OpenPr(
            repo=repo,
            number=pr["number"],
            title=pr.get("title", ""),
            author=(pr.get("author") or {}).get("login", ""),
            url=pr.get("url", ""),
            is_draft=bool(pr.get("isDraft", False)),
            created_at=pr.get("createdAt", ""),
            updated_at=pr.get("updatedAt", ""),
            review_state=pr.get("reviewDecision") or "",
            mergeable=pr.get("mergeable") or "",
            checks_state=checks_state,
            additions=pr.get("additions") or 0,
            deletions=pr.get("deletions") or 0,
            days_open=days_open,
        ))
    return out


def list_open_prs_many(repos: Iterable[str]) -> list[OpenPr]:
    repos = list(repos)
    with ThreadPoolExecutor(max_workers=min(6, len(repos) or 1)) as pool:
        return [pr for prs in pool.map(list_open_prs, repos) for pr in prs]


def estimate_merge_probability(pr: OpenPr) -> float:
    """Combine review + CI + draft + age signals into a merge probability.

    These are heuristics, not calibrated on real merge data. But they capture
    the obvious signal: draft PRs almost never merge as-is; approved PRs
    passing CI almost always merge within a week.
    """
    if pr.is_draft:
        return 0.10
    if pr.review_state == "CHANGES_REQUESTED":
        return 0.15
    if pr.mergeable == "CONFLICTING":
        return 0.20
    if pr.checks_state == "FAILURE":
        return 0.20

    p = 0.55  # base for a normal open PR
    if pr.review_state == "APPROVED":
        p += 0.30
    if pr.checks_state == "SUCCESS":
        p += 0.10
    if pr.checks_state == "PENDING":
        p -= 0.05
    # PRs that have been open >30d and not merged are probably stalled
    if pr.days_open > 30:
        p -= 0.25
    elif pr.days_open > 14:
        p -= 0.10

    return max(0.05, min(0.98, p))


def expected_merge_date(pr: OpenPr, prob: float) -> date:
    """Best-guess merge day. Approved + passing CI → tomorrow. Stalled → far."""
    today = date.today()
    if prob >= 0.85:
        return today + timedelta(days=1)
    if prob >= 0.60:
        return today + timedelta(days=3)
    if prob >= 0.30:
        return today + timedelta(days=7)
    return today + timedelta(days=14)


def pr_diff(repo: str, number: int) -> str:
    """Fetch the raw diff of an open PR — same as merged PRs."""
    return _run(["gh", "pr", "diff", str(number), "--repo", repo])


def analyze_open_prs(
    open_prs: list[OpenPr],
    profile: str | None = None,
    llm_model: str = "us.anthropic.claude-sonnet-4-6",
) -> list[PricedOpenPr]:
    """Run each open PR through the deep LLM analyzer used for merged PRs.

    Returns priced PRs with merge probability + expected delta. Sorted so
    high-impact PRs land first.
    """
    from src.pr_scanner.llm_analyzer import analyze_pr_diff

    priced: list[PricedOpenPr] = []
    for opr in open_prs:
        try:
            diff = pr_diff(opr.repo, opr.number)
        except Exception:  # noqa: BLE001
            continue
        # Reuse the merged-PR analyzer — same prompt, same tool loop.
        verdict = analyze_pr_diff(
            diff, pr_title=opr.title, profile=profile, model_id=llm_model,
        )
        est = float(verdict.total_daily_delta_usd or 0.0)
        direction = ("increase" if est > 0.01 else
                     "decrease" if est < -0.01 else "neutral")
        prob = estimate_merge_probability(opr)
        merge_day = expected_merge_date(opr, prob)
        priced.append(PricedOpenPr(
            open_pr=opr,
            est_daily_delta_usd=round(est, 4),
            direction=direction,
            llm_summary=verdict.summary,
            merge_probability=round(prob, 3),
            expected_merge_day=merge_day.isoformat(),
            expected_daily_delta_usd=round(est * prob, 4),
            findings=[
                {"resource_type": c.resource_type,
                 "resource_name": c.resource_name,
                 "action": c.action,
                 "est_daily_delta_usd": c.est_daily_delta_usd,
                 "rationale": c.rationale}
                for c in verdict.changes
            ],
        ))
    # Rank by absolute expected impact (already probability-weighted)
    priced.sort(key=lambda p: -abs(p.expected_daily_delta_usd))
    return priced


def to_step_series(
    priced: list[PricedOpenPr], cutoff: date, horizon_days: int = 14,
) -> dict[date, float]:
    """Convert priced open PRs into a daily cumulative expected delta.

    Each PR adds its `expected_daily_delta_usd` starting on its expected
    merge day and continuing forever. Days before any expected merge = 0.
    """
    daily: dict[date, float] = {}
    end = cutoff + timedelta(days=horizon_days)
    for pr in priced:
        try:
            start = date.fromisoformat(pr.expected_merge_day)
        except ValueError:
            continue
        d = max(start, cutoff + timedelta(days=1))
        while d <= end:
            daily[d] = daily.get(d, 0.0) + pr.expected_daily_delta_usd
            d = d + timedelta(days=1)
    # Fill zeros for days not yet touched
    d = cutoff + timedelta(days=1)
    while d <= end:
        daily.setdefault(d, 0.0)
        d = d + timedelta(days=1)
    return daily


def to_dict(priced: list[PricedOpenPr]) -> list[dict]:
    return [
        {
            "repo": p.open_pr.repo,
            "pr_number": p.open_pr.number,
            "pr_title": p.open_pr.title,
            "pr_url": p.open_pr.url,
            "author": p.open_pr.author,
            "is_draft": p.open_pr.is_draft,
            "review_state": p.open_pr.review_state,
            "checks_state": p.open_pr.checks_state,
            "mergeable": p.open_pr.mergeable,
            "days_open": p.open_pr.days_open,
            "est_daily_delta_usd": p.est_daily_delta_usd,
            "direction": p.direction,
            "merge_probability": p.merge_probability,
            "expected_merge_day": p.expected_merge_day,
            "expected_daily_delta_usd": p.expected_daily_delta_usd,
            "llm_summary": p.llm_summary,
            "findings": p.findings,
        }
        for p in priced
    ]
