"""Read-only GitHub tools for the chat bot.

Lets CostSense AI answer questions about a repo directly — find it by name,
list its files, read a file, and list/diff pull requests — instead of
saying it has no GitHub access.

Uses the GitHub REST API via `GITHUB_TOKEN`/`GH_TOKEN` (see `src.pr_scanner.gh_client`).
Falls back to the `gh` CLI only when no token env var is set.
Every result still passes through the chat agent's secret scrubber before
Claude sees it.
"""
from __future__ import annotations

from src.pr_scanner.gh_client import (
    get_file,
    list_dir,
    list_pull_requests,
    pr_diff,
    repo_info,
    search_code,
    search_repositories,
)


def _safe(fn):
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}
    wrapped.__name__ = fn.__name__
    return wrapped


_MAX_FILE_CHARS = 8000

# Default org to search first — most repos the user refers to belong here.
# Falls back to an unscoped search for repos outside this org (e.g. the
# user's own fork/clone of this very project).
DEFAULT_ORG = "DiligentCorp"


@_safe
def github_search_repositories(query: str, limit: int = 10) -> dict:
    repos = search_repositories(query, limit=limit, org=DEFAULT_ORG)
    searched_org = DEFAULT_ORG
    if not repos:
        repos = search_repositories(query, limit=limit)
        searched_org = None
    return {"repos": repos, "count": len(repos), "org_scoped_to": searched_org}


GITHUB_SEARCH_REPOS_SPEC = {
    "name": "github_search_repositories",
    "description": (
        "Search GitHub for repositories by name/keywords (e.g. the user "
        "says 'the data platform repo' but not the exact org/repo slug). "
        f"Searches the '{DEFAULT_ORG}' org first; if nothing matches there, "
        "falls back to a global search (covers repos outside that org, "
        "e.g. this very CostSense project). Returns full_name (org/repo), "
        "description, language, default branch — use the matched full_name "
        "in the other github_* tools."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "Keywords, e.g. 'data platform'"},
            "limit": {"type": "integer", "default": 10, "maximum": 30},
        },
        "required": ["query"],
    },
}


@_safe
def github_repo_info(repo: str) -> dict:
    return repo_info(repo)


GITHUB_REPO_INFO_SPEC = {
    "name": "github_repo_info",
    "description": ("Get metadata for a GitHub repo: description, default "
                    "branch, language, stars, open issues, topics."),
    "input_schema": {
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "org/repo, e.g. DiligentCorp/data-platform"},
        },
        "required": ["repo"],
    },
}


@_safe
def github_list_dir(repo: str, path: str = "", ref: str | None = None) -> dict:
    entries = list_dir(repo, path=path, ref=ref)
    return {"repo": repo, "path": path or "/", "entries": entries,
            "count": len(entries)}


GITHUB_LIST_DIR_SPEC = {
    "name": "github_list_dir",
    "description": ("List files/folders at a path in a GitHub repo (default "
                    "branch unless `ref` is given). Use path='' for the "
                    "repo root."),
    "input_schema": {
        "type": "object",
        "properties": {
            "repo": {"type": "string"},
            "path": {"type": "string", "default": ""},
            "ref": {"type": "string",
                   "description": "branch, tag, or commit SHA (optional)"},
        },
        "required": ["repo"],
    },
}


@_safe
def github_get_file(repo: str, path: str, ref: str | None = None) -> dict:
    content = get_file(repo, path, ref=ref)
    truncated = len(content) > _MAX_FILE_CHARS
    if truncated:
        content = content[:_MAX_FILE_CHARS]
    return {"repo": repo, "path": path, "content": content,
            "truncated": truncated}


GITHUB_GET_FILE_SPEC = {
    "name": "github_get_file",
    "description": ("Read a text file's contents from a GitHub repo (e.g. "
                    "README.md, a Terraform/CDK file, package.json). Large "
                    f"files are truncated to {_MAX_FILE_CHARS} chars."),
    "input_schema": {
        "type": "object",
        "properties": {
            "repo": {"type": "string"},
            "path": {"type": "string", "description": "file path within the repo"},
            "ref": {"type": "string",
                   "description": "branch, tag, or commit SHA (optional)"},
        },
        "required": ["repo", "path"],
    },
}


@_safe
def github_search_code(repo: str, query: str, limit: int = 10) -> dict:
    results = search_code(repo, query, limit=limit)
    return {"repo": repo, "results": results, "count": len(results)}


GITHUB_SEARCH_CODE_SPEC = {
    "name": "github_search_code",
    "description": ("Search for code within one GitHub repo by keyword "
                    "(e.g. a resource name, function name, config key). "
                    "Returns matching file paths."),
    "input_schema": {
        "type": "object",
        "properties": {
            "repo": {"type": "string"},
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 10, "maximum": 30},
        },
        "required": ["repo", "query"],
    },
}


@_safe
def github_list_pull_requests(repo: str, state: str = "open",
                              limit: int = 20) -> dict:
    prs = list_pull_requests(repo, state=state, limit=limit)
    return {"repo": repo, "state": state, "pull_requests": prs,
            "count": len(prs)}


GITHUB_LIST_PRS_SPEC = {
    "name": "github_list_pull_requests",
    "description": ("List pull requests on a GitHub repo, filtered by state "
                    "(open/closed/merged/all)."),
    "input_schema": {
        "type": "object",
        "properties": {
            "repo": {"type": "string"},
            "state": {"type": "string",
                     "enum": ["open", "closed", "merged", "all"],
                     "default": "open"},
            "limit": {"type": "integer", "default": 20, "maximum": 50},
        },
        "required": ["repo"],
    },
}


@_safe
def github_pr_diff(repo: str, number: int) -> dict:
    diff = pr_diff(repo, number)
    truncated = len(diff) > _MAX_FILE_CHARS
    if truncated:
        diff = diff[:_MAX_FILE_CHARS]
    return {"repo": repo, "number": number, "diff": diff, "truncated": truncated}


GITHUB_PR_DIFF_SPEC = {
    "name": "github_pr_diff",
    "description": ("Get the unified diff for a specific pull request "
                    "number on a GitHub repo."),
    "input_schema": {
        "type": "object",
        "properties": {
            "repo": {"type": "string"},
            "number": {"type": "integer"},
        },
        "required": ["repo", "number"],
    },
}


GITHUB_TOOLS: dict[str, tuple[callable, dict]] = {
    "github_search_repositories": (github_search_repositories, GITHUB_SEARCH_REPOS_SPEC),
    "github_repo_info": (github_repo_info, GITHUB_REPO_INFO_SPEC),
    "github_list_dir": (github_list_dir, GITHUB_LIST_DIR_SPEC),
    "github_get_file": (github_get_file, GITHUB_GET_FILE_SPEC),
    "github_search_code": (github_search_code, GITHUB_SEARCH_CODE_SPEC),
    "github_list_pull_requests": (github_list_pull_requests, GITHUB_LIST_PRS_SPEC),
    "github_pr_diff": (github_pr_diff, GITHUB_PR_DIFF_SPEC),
}


def all_github_specs() -> list[dict]:
    return [spec for _, spec in GITHUB_TOOLS.values()]
