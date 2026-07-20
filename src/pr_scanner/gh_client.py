"""GitHub access — prefers `gh` CLI when installed, falls back to REST API."""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request


def gh_available() -> bool:
    return shutil.which("gh") is not None


def _run_cli(args: list[str]) -> str:
    r = subprocess.run(args, capture_output=True, check=False)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"gh failed ({args}): {err}")
    return r.stdout.decode("utf-8", errors="replace")


def _github_token() -> str | None:
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def _api_request(path: str, *, accept: str = "application/vnd.github+json") -> bytes:
    headers = {
        "Accept": accept,
        "User-Agent": "costsense-forecast",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = _github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(f"https://api.github.com{path}", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        if e.code in (401, 403):
            raise RuntimeError(
                "GitHub API auth failed. Set GITHUB_TOKEN (or GH_TOKEN) "
                "with repo read access, or install the GitHub CLI (`gh`) "
                "and run `gh auth login`."
            ) from e
        if e.code == 404:
            raise RuntimeError(
                f"GitHub resource not found or not accessible: {path}. "
                "Check the PR URL and token permissions."
            ) from e
        raise RuntimeError(f"GitHub API error {e.code}: {body[:500]}") from e


def pr_view_json(repo: str, number: int) -> dict:
    if gh_available():
        return json.loads(_run_cli([
            "gh", "pr", "view", str(number),
            "--repo", repo, "--json", "title",
        ]))
    data = json.loads(_api_request(f"/repos/{repo}/pulls/{number}"))
    return {"title": data.get("title", "")}


def pr_diff(repo: str, number: int) -> str:
    if gh_available():
        return _run_cli(["gh", "pr", "diff", str(number), "--repo", repo])
    return _api_request(
        f"/repos/{repo}/pulls/{number}",
        accept="application/vnd.github.diff",
    ).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Generic repo browsing — used by the chat agent's GitHub tools
# ---------------------------------------------------------------------------

def api_get(path: str, params: dict | None = None,
           accept: str = "application/vnd.github+json"):
    """GET a GitHub REST API path (no leading slash), via `gh api` when
    available, else the raw REST API. Returns parsed JSON."""
    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    if gh_available():
        args = ["gh", "api", f"{path}{query}"]
        if accept != "application/vnd.github+json":
            args += ["-H", f"Accept: {accept}"]
        out = _run_cli(args)
        return json.loads(out) if out.strip() else None
    data = _api_request(f"/{path}{query}", accept=accept)
    return json.loads(data) if data.strip() else None


def search_repositories(query: str, limit: int = 10,
                        org: str | None = None) -> list[dict]:
    q = f"{query} org:{org}" if org else query
    data = api_get("search/repositories", {"q": q, "per_page": limit}) or {}
    items = data.get("items", []) if isinstance(data, dict) else []
    return [{
        "full_name": it.get("full_name"),
        "description": it.get("description"),
        "private": it.get("private"),
        "default_branch": it.get("default_branch"),
        "language": it.get("language"),
        "updated_at": it.get("updated_at"),
        "url": it.get("html_url"),
    } for it in items]


def repo_info(repo: str) -> dict:
    data = api_get(f"repos/{repo}") or {}
    return {
        "full_name": data.get("full_name"),
        "description": data.get("description"),
        "private": data.get("private"),
        "default_branch": data.get("default_branch"),
        "language": data.get("language"),
        "stars": data.get("stargazers_count"),
        "open_issues": data.get("open_issues_count"),
        "topics": data.get("topics"),
        "updated_at": data.get("updated_at"),
        "url": data.get("html_url"),
    }


def list_dir(repo: str, path: str = "", ref: str | None = None) -> list[dict]:
    params = {"ref": ref} if ref else None
    data = api_get(f"repos/{repo}/contents/{path}".rstrip("/"), params)
    if isinstance(data, dict):
        raise RuntimeError(
            f"'{path}' is a file, not a directory. Use get_file instead."
        )
    return [{"name": it.get("name"), "path": it.get("path"),
             "type": it.get("type"), "size": it.get("size")}
            for it in (data or [])]


def get_file(repo: str, path: str, ref: str | None = None) -> str:
    params = {"ref": ref} if ref else None
    data = api_get(f"repos/{repo}/contents/{path}", params)
    if isinstance(data, list):
        raise RuntimeError(f"'{path}' is a directory. Use list_dir instead.")
    if not data or data.get("encoding") != "base64":
        raise RuntimeError(f"unsupported or empty content for {path!r}")
    return base64.b64decode(data.get("content", "")).decode(
        "utf-8", errors="replace")


def list_pull_requests(repo: str, state: str = "open", limit: int = 20) -> list[dict]:
    if gh_available():
        out = _run_cli([
            "gh", "pr", "list", "--repo", repo, "--state", state,
            "--limit", str(limit), "--json",
            "number,title,author,createdAt,url,baseRefName",
        ])
        return json.loads(out or "[]")
    data = api_get(f"repos/{repo}/pulls", {"state": state, "per_page": limit}) or []
    return [{
        "number": p.get("number"),
        "title": p.get("title"),
        "author": (p.get("user") or {}).get("login"),
        "createdAt": p.get("created_at"),
        "url": p.get("html_url"),
        "baseRefName": (p.get("base") or {}).get("ref"),
    } for p in data]


def search_code(repo: str, query: str, limit: int = 10) -> list[dict]:
    data = api_get("search/code", {"q": f"{query} repo:{repo}",
                                   "per_page": limit}) or {}
    items = data.get("items", []) if isinstance(data, dict) else []
    return [{"path": it.get("path"), "name": it.get("name"),
             "url": it.get("html_url")} for it in items]
