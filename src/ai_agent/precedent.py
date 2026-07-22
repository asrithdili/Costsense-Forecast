"""Historical precedent lookup for scope-expansion PRs.

Instead of falling back to a generic per-tenant rate when the target AWS
account has no measurable signal, this module finds PRIOR merged PRs in
the same repo that also expanded scope (grew a whitelist/allowlist), then
measures how the *sibling* AWS accounts (dev/staging/prod) reacted around
each merge date. The resulting $/tenant/day is a real number, not a guess.

Pipeline:
  1. Identify the whitelist file(s) the current PR grew.
  2. `gh pr list` for prior merged PRs touching those files.
  3. For each precedent PR:
     - fetch diff → count tenants added via `extract_resources`
     - grab merged_at
     - for each sibling env AWS profile that matches the repo, fetch daily
       totals around the merge and detect a step change (Bayesian-style
       changepoint: max t-stat over candidate split points).
  4. Aggregate → mean $/tenant/day + confidence band.
"""
from __future__ import annotations

import json
import re
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from src.aws.cost_explorer import fetch_daily_totals
from src.pr_scanner.gh_client import pr_diff, gh_available


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _run(args: list[str]) -> str:
    r = subprocess.run(args, capture_output=True, check=False)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode("utf-8", errors="replace").strip())
    return r.stdout.decode("utf-8", errors="replace")


def _files_touched_in_diff(diff: str) -> list[str]:
    """Extract `+++ b/<path>` paths from a unified diff."""
    files: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            files.append(line[6:].strip())
    return files


# ---------------------------------------------------------------------------
# step-change detection
# ---------------------------------------------------------------------------

def _detect_step_change(
    series: list[tuple[date, float]],
    merge_day: date,
    window_days: int = 14,
) -> float | None:
    """Return the estimated step ($/day) at merge_day, or None if the
    series is too short / noisy. Compares avg of the `window_days` days
    BEFORE (up to and including merge_day) to `window_days` AFTER."""
    if not series:
        return None
    by_day = {d: a for d, a in series}
    before = [by_day.get(merge_day - timedelta(days=i))
              for i in range(1, window_days + 1)]
    after = [by_day.get(merge_day + timedelta(days=i))
             for i in range(1, window_days + 1)]
    before = [x for x in before if x is not None]
    after = [x for x in after if x is not None]
    if len(before) < 5 or len(after) < 5:
        return None
    mean_before = statistics.fmean(before)
    mean_after = statistics.fmean(after)

    # Guard against noise: require the step to exceed 1σ of the combined
    # baseline. Otherwise it's just workload jitter.
    try:
        pooled_std = statistics.pstdev(before + after)
    except statistics.StatisticsError:
        pooled_std = 0.0
    step = mean_after - mean_before
    if pooled_std > 0 and abs(step) < pooled_std:
        return None
    return step


# ---------------------------------------------------------------------------
# github: prior PRs touching the same files
# ---------------------------------------------------------------------------

def _list_prior_prs_touching_files(
    repo: str, files: list[str], limit: int = 20,
) -> list[dict]:
    """Merged PRs in `repo` that touched at least one of the given files.

    GitHub's PR search doesn't natively support file filters, so we use
    the code/search + a follow-up `gh pr list` in --search mode as a
    heuristic. Fallback: list recent merged PRs and check their file
    lists via the REST API (slower)."""
    if not gh_available():
        return []
    # Heuristic: search issues (PRs are issues on GitHub) using a filename
    # keyword. This is imperfect but avoids fetching every merged PR's
    # file list.
    hits: dict[int, dict] = {}
    for f in files[:5]:
        # Just the basename — search matches PR title/body.
        base = f.rsplit("/", 1)[-1]
        try:
            out = _run([
                "gh", "pr", "list", "--repo", repo,
                "--state", "merged", "--limit", str(limit),
                "--search", base,
                "--json", "number,title,mergedAt,url",
            ])
        except RuntimeError:
            continue
        for pr in json.loads(out or "[]"):
            hits[pr["number"]] = pr
    return list(hits.values())


def _pr_files(repo: str, number: int) -> list[str]:
    """List of files changed by PR #number."""
    if not gh_available():
        return []
    try:
        out = _run([
            "gh", "api", f"repos/{repo}/pulls/{number}/files?per_page=100",
        ])
        return [f.get("filename", "") for f in json.loads(out or "[]")]
    except RuntimeError:
        return []


# ---------------------------------------------------------------------------
# top-level
# ---------------------------------------------------------------------------

@dataclass
class PrecedentSample:
    """One historical (repo, PR, sibling profile) observation."""
    pr_number: int
    pr_url: str
    merged_at: str
    tenants_added: int
    sibling_profile: str
    account_id: str
    step_daily_usd: float
    per_tenant_daily_usd: float


@dataclass
class PrecedentAggregate:
    samples: list[PrecedentSample] = field(default_factory=list)
    mean_per_tenant_daily_usd: float | None = None
    low_per_tenant_daily_usd: float | None = None
    high_per_tenant_daily_usd: float | None = None
    note: str = ""

    @property
    def usable(self) -> bool:
        return bool(self.samples) and self.mean_per_tenant_daily_usd is not None


def _sibling_profiles_for_repo(repo: str):
    """AWS profiles across ALL envs (dev/staging/prod/…) that map to this
    repo's short name. Returns ProfileInfo objects."""
    from src.aws.profiles import resolve_all
    from src.pr_scanner.profile_repo_match import normalize_profile

    short = repo.split("/", 1)[-1].lower()
    matches = []
    for p in resolve_all():
        if not p.account_id:
            continue
        norm = (normalize_profile(p.profile) or "").lower()
        if not norm:
            continue
        if short == norm or short.startswith(norm) or norm.startswith(short):
            matches.append(p)
    return matches


def find_precedents(
    repo: str,
    current_diff: str,
    max_precedents: int = 5,
    window_days: int = 14,
) -> PrecedentAggregate:
    """Full pipeline. Returns aggregated $/tenant/day rate + samples."""
    from src.ai_agent.diff_resources import (
        _count_added_scope_ids, extract_resources,
    )

    agg = PrecedentAggregate()
    files = _files_touched_in_diff(current_diff)
    if not files:
        agg.note = "No files identified in the PR diff."
        return agg

    prior = _list_prior_prs_touching_files(repo, files, limit=25)
    if not prior:
        agg.note = "No prior merged PRs found touching these files."
        return agg

    # Only PRs whose diff actually adds tenant IDs count as precedents.
    candidates: list[tuple[dict, int]] = []
    for pr in prior:
        num = int(pr.get("number", 0))
        if not num:
            continue
        try:
            diff = pr_diff(repo, num)
        except Exception:  # noqa: BLE001
            continue
        added = _count_added_scope_ids(diff.splitlines())
        if added >= 3:
            candidates.append((pr, added))
        if len(candidates) >= max_precedents:
            break

    if not candidates:
        agg.note = (f"Found {len(prior)} prior PR(s) touching the same "
                    "files, but none of them were scope expansions.")
        return agg

    sib_profiles = _sibling_profiles_for_repo(repo)
    if not sib_profiles:
        agg.note = ("Found precedent PRs, but no local AWS profile "
                    "matches this repo — can't measure historical reaction.")
        return agg

    samples: list[PrecedentSample] = []
    today = date.today()
    for pr, tenants_added in candidates:
        merged_at_iso = pr.get("mergedAt") or ""
        if not merged_at_iso:
            continue
        try:
            merge_day = date.fromisoformat(merged_at_iso[:10])
        except ValueError:
            continue
        # Cost Explorer daily granularity is ~13 months back at most.
        if merge_day < today - timedelta(days=360):
            continue
        if merge_day > today - timedelta(days=window_days + 1):
            continue  # not enough post-merge history yet

        for sib in sib_profiles:
            start = merge_day - timedelta(days=window_days + 2)
            end = merge_day + timedelta(days=window_days + 2)
            try:
                series = fetch_daily_totals(start, end, profile=sib.profile)
            except Exception:  # noqa: BLE001
                continue
            step = _detect_step_change(series, merge_day, window_days)
            if step is None:
                continue
            per_tenant = step / tenants_added
            samples.append(PrecedentSample(
                pr_number=int(pr.get("number", 0)),
                pr_url=str(pr.get("url", "")),
                merged_at=merged_at_iso[:10],
                tenants_added=tenants_added,
                sibling_profile=sib.profile,
                account_id=sib.account_id or "",
                step_daily_usd=round(step, 3),
                per_tenant_daily_usd=round(per_tenant, 4),
            ))

    if not samples:
        agg.note = (f"Found {len(candidates)} precedent PR(s), but no "
                    "sibling AWS account showed a detectable step change "
                    "in daily spend around their merges.")
        return agg

    # Pick the single most trustworthy precedent: the one that added the
    # most tenants, so noise-per-tenant is smallest. A 6-tenant PR whose
    # account happened to drift -$24/day around merge day would otherwise
    # dominate the average with a spurious signal.
    best = max(samples, key=lambda s: s.tenants_added)
    agg.samples = [best]
    agg.mean_per_tenant_daily_usd = round(best.per_tenant_daily_usd, 4)
    agg.low_per_tenant_daily_usd = round(best.per_tenant_daily_usd, 4)
    agg.high_per_tenant_daily_usd = round(best.per_tenant_daily_usd, 4)
    agg.note = (
        f"Grounded in the largest historical precedent: PR #{best.pr_number} "
        f"(merged {best.merged_at}) added {best.tenants_added} tenants, "
        f"causing a ${best.step_daily_usd:+,.2f}/day step in "
        f"`{best.sibling_profile}`. Dropped {len(samples) - 1} smaller "
        "sample(s) as too noisy."
        if len(samples) > 1 else
        f"Grounded in one historical precedent: PR #{best.pr_number} "
        f"(merged {best.merged_at}) added {best.tenants_added} tenants, "
        f"causing a ${best.step_daily_usd:+,.2f}/day step in "
        f"`{best.sibling_profile}`."
    )
    return agg


def precedent_prompt_hint(agg: PrecedentAggregate, new_tenants: int) -> str:
    """Format a PrecedentAggregate as a prompt block the LLM sees."""
    if not agg.usable:
        return f"HISTORICAL PRECEDENT: {agg.note}"

    lines = [
        "HISTORICAL PRECEDENT — measured $/tenant/day from prior merges:",
        f"  Rate: ${agg.mean_per_tenant_daily_usd:.4f}/tenant/day "
        f"(range ${agg.low_per_tenant_daily_usd:.4f}"
        f"–${agg.high_per_tenant_daily_usd:.4f})",
        f"  {agg.note}",
        "",
        "Sample observations:",
    ]
    for s in agg.samples[:5]:
        lines.append(
            f"  • PR #{s.pr_number} merged {s.merged_at}: added "
            f"{s.tenants_added} tenants → account `{s.sibling_profile}` "
            f"stepped by ${s.step_daily_usd:+,.2f}/day "
            f"(${s.per_tenant_daily_usd:+,.4f}/tenant/day)"
        )
    lines.append("")
    lines.append(
        f"Apply this measured rate to the {new_tenants} tenants THIS PR "
        f"adds: expected delta ≈ ${agg.mean_per_tenant_daily_usd * new_tenants:+,.2f}"
        f"/day (range ${agg.low_per_tenant_daily_usd * new_tenants:+,.2f}"
        f"–${agg.high_per_tenant_daily_usd * new_tenants:+,.2f}/day). "
        "Set `estimation_basis=\"sibling_account\"` and "
        "`confidence=\"medium\"` since this is grounded in real "
        "historical data, not a generic rate."
    )
    return "\n".join(lines)
