"""GitHub access — uses GITHUB_TOKEN / GH_TOKEN when set, else `gh` CLI."""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request

from src.config import get_str
from src.env import load_env

load_env()


# Common Windows install locations — checked as a fallback when `gh` isn't
# on PATH. A Streamlit server started from a terminal that predates a fresh
# `gh` install (or a fresh PATH update) won't see it via shutil.which alone,
# silently falling back to an unauthenticated REST call that 404s on private
# repos instead of a clear auth error.
_GH_FALLBACK_PATHS = [
    r"C:\Program Files\GitHub CLI\gh.exe",
    r"C:\Program Files (x86)\GitHub CLI\gh.exe",
]

_gh_path_cache: str | None = None


def _resolve_gh() -> str | None:
    global _gh_path_cache
    if _gh_path_cache is not None:
        return _gh_path_cache or None
    found = shutil.which("gh")
    if not found:
        for candidate in _GH_FALLBACK_PATHS:
            if os.path.isfile(candidate):
                found = candidate
                break
    _gh_path_cache = found or ""
    return found or None


def gh_available() -> bool:
    return _resolve_gh() is not None


def _github_token() -> str | None:
    token = get_str("github.token")
    if token:
        return token
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def token_configured() -> bool:
    """True when GITHUB_TOKEN or GH_TOKEN is set in the environment."""
    return bool(_github_token())


def _run_cli(args: list[str]) -> str:
    gh_path = _resolve_gh() or "gh"
    resolved_args = [gh_path, *args[1:]] if args and args[0] == "gh" else args
    r = subprocess.run(resolved_args, capture_output=True, check=False)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"gh failed ({args}): {err}")
    return r.stdout.decode("utf-8", errors="replace")


def _api_request(
    path: str,
    *,
    accept: str = "application/vnd.github+json",
    method: str = "GET",
    body: bytes | None = None,
) -> bytes:
    headers = {
        "Accept": accept,
        "User-Agent": "costsense-forecast",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = _github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif not gh_available():
        raise RuntimeError(
            "GitHub auth not configured. Set GITHUB_TOKEN (or GH_TOKEN) "
            "in your environment, or install the GitHub CLI (`gh`) and run "
            "`gh auth login`."
        )
    if body is not None:
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(
        f"https://api.github.com{path}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        if e.code in (401, 403):
            raise RuntimeError(
                "GitHub API auth failed. Set GITHUB_TOKEN (or GH_TOKEN) "
                "with repo access in your environment."
            ) from e
        if e.code == 404:
            raise RuntimeError(
                f"GitHub resource not found or not accessible: {path}. "
                "Check the PR URL and token permissions."
            ) from e
        raise RuntimeError(f"GitHub API error {e.code}: {err_body[:500]}") from e


def api_get(path: str, params: dict | None = None,
           accept: str = "application/vnd.github+json"):
    """GET a GitHub REST API path (no leading slash). Returns parsed JSON."""
    query = f"?{urllib.parse.urlencode(params)}" if params else ""
    if token_configured():
        data = _api_request(f"/{path}{query}", accept=accept)
        return json.loads(data) if data.strip() else None
    if gh_available():
        args = ["gh", "api", f"{path}{query}"]
        if accept != "application/vnd.github+json":
            args += ["-H", f"Accept: {accept}"]
        out = _run_cli(args)
        return json.loads(out) if out.strip() else None
    data = _api_request(f"/{path}{query}", accept=accept)
    return json.loads(data) if data.strip() else None


def api_post(path: str, payload: dict) -> dict:
    """POST to a GitHub REST API path (no leading slash)."""
    if token_configured():
        data = _api_request(
            f"/{path}", method="POST", body=json.dumps(payload).encode(),
        )
        return json.loads(data) if data.strip() else {}
    if gh_available():
        import subprocess
        gh_path = _resolve_gh() or "gh"
        r = subprocess.run(
            [gh_path, "api", path, "-X", "POST", "--input", "-"],
            input=json.dumps(payload).encode(),
            capture_output=True,
            check=False,
        )
        if r.returncode != 0:
            err = r.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"gh api POST failed: {err}")
        out = r.stdout.decode("utf-8", errors="replace")
        return json.loads(out) if out.strip() else {}
    data = _api_request(
        f"/{path}", method="POST", body=json.dumps(payload).encode(),
    )
    return json.loads(data) if data.strip() else {}


def api_put(path: str, payload: dict) -> dict:
    """PUT to a GitHub REST API path (no leading slash)."""
    if token_configured():
        data = _api_request(
            f"/{path}", method="PUT", body=json.dumps(payload).encode(),
        )
        return json.loads(data) if data.strip() else {}
    if gh_available():
        import subprocess
        gh_path = _resolve_gh() or "gh"
        r = subprocess.run(
            [gh_path, "api", path, "-X", "PUT", "--input", "-"],
            input=json.dumps(payload).encode(),
            capture_output=True,
            check=False,
        )
        if r.returncode != 0:
            err = r.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"gh api PUT failed: {err}")
        out = r.stdout.decode("utf-8", errors="replace")
        return json.loads(out) if out.strip() else {}
    data = _api_request(
        f"/{path}", method="PUT", body=json.dumps(payload).encode(),
    )
    return json.loads(data) if data.strip() else {}


def pr_view_json(repo: str, number: int) -> dict:
    if token_configured():
        data = json.loads(_api_request(f"/repos/{repo}/pulls/{number}"))
        return {"title": data.get("title", "")}
    if gh_available():
        return json.loads(_run_cli([
            "gh", "pr", "view", str(number),
            "--repo", repo, "--json", "title",
        ]))
    data = json.loads(_api_request(f"/repos/{repo}/pulls/{number}"))
    return {"title": data.get("title", "")}


def pr_diff(repo: str, number: int) -> str:
    if token_configured():
        return _api_request(
            f"/repos/{repo}/pulls/{number}",
            accept="application/vnd.github.diff",
        ).decode("utf-8", errors="replace")
    if gh_available():
        return _run_cli(["gh", "pr", "diff", str(number), "--repo", repo])
    return _api_request(
        f"/repos/{repo}/pulls/{number}",
        accept="application/vnd.github.diff",
    ).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Generic repo browsing — used by the chat agent's GitHub tools
# ---------------------------------------------------------------------------


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
    if token_configured():
        data = api_get(f"repos/{repo}/pulls", {"state": state, "per_page": limit}) or []
        return [{
            "number": p.get("number"),
            "title": p.get("title"),
            "author": (p.get("user") or {}).get("login"),
            "createdAt": p.get("created_at"),
            "updatedAt": p.get("updated_at"),
            "url": p.get("html_url"),
            "baseRefName": (p.get("base") or {}).get("ref"),
            "additions": p.get("additions"),
            "deletions": p.get("deletions"),
        } for p in data]
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
