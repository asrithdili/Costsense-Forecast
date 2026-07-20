"""Enumerate GitHub repos the current `gh` user has interacted with.

Two lenses are useful for the UI:
  - repos where the user has authored PRs (best signal of "repos I work on")
  - a free-form `org/repo` textbox to add anything the search missed
"""
from __future__ import annotations

import json
import subprocess
from functools import lru_cache


def _run(args: list[str]) -> str:
    r = subprocess.run(args, capture_output=True, check=False)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"gh failed ({args}): {err}")
    return r.stdout.decode("utf-8", errors="replace")


@lru_cache(maxsize=8)
def repos_with_user_prs(org: str, limit: int = 100) -> tuple[str, ...]:
    """Repos in `org` where the authenticated user has authored PRs."""
    login = json.loads(_run(["gh", "api", "user"]))["login"]
    resp = _run([
        "gh", "api",
        f"search/issues?q=author:{login}+org:{org}+is:pr&per_page={limit}",
    ])
    data = json.loads(resp)
    repos: list[str] = []
    seen: set[str] = set()
    for item in data.get("items", []):
        url = item.get("repository_url", "")
        if not url:
            continue
        parts = url.split("/repos/", 1)
        if len(parts) != 2:
            continue
        slug = parts[1]
        if slug in seen:
            continue
        seen.add(slug)
        repos.append(slug)
    return tuple(sorted(repos))


@lru_cache(maxsize=1)
def gh_login() -> str:
    return json.loads(_run(["gh", "api", "user"]))["login"]


@lru_cache(maxsize=1)
def gh_orgs() -> tuple[str, ...]:
    """Orgs the authenticated user is a member of."""
    resp = _run(["gh", "api", "user/orgs"])
    return tuple(o["login"] for o in json.loads(resp))


@lru_cache(maxsize=64)
def repo_default_branch(repo: str) -> str:
    """Default branch of `org/repo` (e.g. main, master, develop)."""
    resp = _run(["gh", "api", f"repos/{repo}"])
    return json.loads(resp).get("default_branch", "main")


@lru_cache(maxsize=64)
def recent_base_branches(repo: str, limit: int = 30) -> tuple[str, ...]:
    """Base branches that have received merged PRs recently — ranked by count."""
    resp = _run([
        "gh", "pr", "list",
        "--repo", repo, "--state", "merged", "--limit", str(limit),
        "--json", "baseRefName",
    ])
    prs = json.loads(resp)
    counts: dict[str, int] = {}
    for p in prs:
        b = p.get("baseRefName") or ""
        if b:
            counts[b] = counts.get(b, 0) + 1
    return tuple(sorted(counts, key=lambda b: -counts[b]))
