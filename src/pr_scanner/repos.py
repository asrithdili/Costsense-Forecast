"""Enumerate GitHub repos the current token user has interacted with.

Two lenses are useful for the UI:
  - repos where the user has authored PRs (best signal of "repos I work on")
  - a free-form `org/repo` textbox to add anything the search missed
"""
from __future__ import annotations

from functools import lru_cache

from src.pr_scanner.gh_client import api_get, list_pull_requests


@lru_cache(maxsize=8)
def repos_with_user_prs(org: str, limit: int = 100) -> tuple[str, ...]:
    """Repos in `org` where the authenticated user has authored PRs."""
    login = (api_get("user") or {}).get("login", "")
    if not login:
        return ()
    data = api_get("search/issues", {
        "q": f"author:{login} org:{org} is:pr",
        "per_page": str(limit),
    }) or {}
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
    return (api_get("user") or {}).get("login", "?")


@lru_cache(maxsize=1)
def gh_orgs() -> tuple[str, ...]:
    """Orgs the authenticated user is a member of."""
    data = api_get("user/orgs") or []
    return tuple(o["login"] for o in data if o.get("login"))


@lru_cache(maxsize=64)
def repo_default_branch(repo: str) -> str:
    """Default branch of `org/repo` (e.g. main, master, develop)."""
    data = api_get(f"repos/{repo}") or {}
    return data.get("default_branch", "main")


@lru_cache(maxsize=64)
def recent_base_branches(repo: str, limit: int = 30) -> tuple[str, ...]:
    """Base branches that have received merged PRs recently — ranked by count."""
    prs = list_pull_requests(repo, state="closed", limit=limit)
    counts: dict[str, int] = {}
    for p in prs:
        b = p.get("baseRefName") or ""
        if b:
            counts[b] = counts.get(b, 0) + 1
    return tuple(sorted(counts, key=lambda b: -counts[b]))
