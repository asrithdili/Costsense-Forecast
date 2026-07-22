"""GitHub write helpers — create branches, commit files, open draft PRs.

Authentication: set **GITHUB_TOKEN** or **GH_TOKEN** in your environment
(classic `repo` scope, or fine-grained with contents + pull_requests write).
Never commit tokens to the repo.
"""
from __future__ import annotations

import base64
import difflib
import json
import re
import urllib.parse

from src.pr_scanner.gh_client import (
    _api_request,
    api_get,
    api_post,
    api_put,
    get_file,
    token_configured,
)
from src.pr_scanner.repos import repo_default_branch


def github_write_auth_status() -> tuple[bool, str]:
    """Return (ready, human-readable status) for opening draft PRs."""
    if not token_configured():
        return (
            False,
            "Set **GITHUB_TOKEN** or **GH_TOKEN** in your environment "
            "(repo scope, or contents:write + pull_requests:write).",
        )
    try:
        api_get("user")
        return True, "GitHub write ready (GITHUB_TOKEN / GH_TOKEN)"
    except Exception as e:  # noqa: BLE001
        return False, f"Token check failed: {e}"


def _branch_ref_sha(repo: str, branch: str) -> str:
    data = json.loads(_api_request(
        f"/repos/{repo}/git/ref/heads/{urllib.parse.quote(branch, safe='')}",
    ))
    return data["object"]["sha"]


def create_branch(repo: str, branch: str, base: str | None = None) -> None:
    """Create `branch` pointing at the tip of `base` (default branch if omitted)."""
    base = base or repo_default_branch(repo)
    sha = _branch_ref_sha(repo, base)
    try:
        api_post(f"repos/{repo}/git/refs", {
            "ref": f"refs/heads/{branch}",
            "sha": sha,
        })
    except RuntimeError as e:
        if "Reference already exists" not in str(e):
            raise


def _file_sha(repo: str, path: str, ref: str) -> str | None:
    try:
        data = api_get(f"repos/{repo}/contents/{path}", {"ref": ref})
        if isinstance(data, dict):
            return data.get("sha")
    except Exception:  # noqa: BLE001
        pass
    return None


def upsert_file(
    repo: str,
    path: str,
    content: str,
    branch: str,
    message: str,
) -> None:
    """Create or update a single file on `branch`."""
    if not token_configured():
        raise RuntimeError(
            "GITHUB_TOKEN (or GH_TOKEN) is required to commit files."
        )
    payload: dict = {
        "message": message,
        "content": base64.b64encode(content.encode()).decode(),
        "branch": branch,
    }
    sha = _file_sha(repo, path, branch)
    if not sha:
        sha = _file_sha(repo, path, repo_default_branch(repo))
    if sha:
        payload["sha"] = sha
    api_put(f"repos/{repo}/contents/{path}", payload)


def create_draft_pr(
    repo: str,
    title: str,
    body: str,
    head: str,
    base: str | None = None,
) -> str:
    """Open a draft PR; returns the PR URL."""
    if not token_configured():
        raise RuntimeError(
            "GITHUB_TOKEN (or GH_TOKEN) is required to open draft PRs."
        )
    base = base or repo_default_branch(repo)
    data = api_post(f"repos/{repo}/pulls", {
        "title": title,
        "body": body,
        "head": head,
        "base": base,
        "draft": True,
    })
    return data.get("html_url", "")


def unified_diff(path: str, old: str, new: str) -> str:
    lines = list(difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    ))
    return "".join(lines) if lines else f"--- a/{path}\n+++ b/{path}\n(new file)\n"


def build_diff_preview(
    repo: str,
    files: list[dict],
    base: str | None = None,
) -> str:
    """Unified diff for each planned file against the default branch."""
    base = base or repo_default_branch(repo)
    parts: list[str] = []
    for f in files:
        path = f["path"]
        new_content = f["content"]
        try:
            old_content = get_file(repo, path, ref=base)
        except Exception:  # noqa: BLE001
            old_content = ""
        parts.append(unified_diff(path, old_content, new_content))
    return "\n".join(parts)


def slugify_branch(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len] or "fix"


def apply_pr_plan(
    repo: str,
    branch: str,
    title: str,
    body: str,
    files: list[dict],
) -> str:
    """Create branch, commit all files, open draft PR. Returns PR URL."""
    create_branch(repo, branch)
    for f in files:
        upsert_file(
            repo, f["path"], f["content"], branch,
            message=f"CostSense: {title}",
        )
    return create_draft_pr(repo, title, body, head=branch)
