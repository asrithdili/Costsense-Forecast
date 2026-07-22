"""Plan a draft PR for an Anomalies recommendation.

Second-pass Bedrock agent with read-only GitHub tools. Locates the target
repo/file(s), applies the suggested fix, and returns structured JSON for
preview + `gh_write.apply_pr_plan`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from src.ai_agent.anomaly_agent import Action, Approach
from src.ai_agent.aws_tools_broad import scrub
from src.ai_agent.bedrock_client import make_client
from src.ai_agent.github_tools import GITHUB_TOOLS, all_github_specs
from src.pr_scanner.gh_write import slugify_branch


BEDROCK_REGION = "us-west-2"
DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"
MAX_TOOL_TURNS = 10
MAX_TOKENS = 8000

_PR_FIX_TOOL_NAMES = frozenset({
    "github_repo_info",
    "github_list_dir",
    "github_get_file",
    "github_search_code",
})

SYSTEM = """You are a senior infra engineer. CostSense flagged a cost issue \
and suggested a code fix. Your job:

  1. Pick ONE target repo from ALLOWED_REPOS (never invent a repo).
  2. Use github_search_code / github_get_file to find the real file(s) to edit.
  3. Return the FULL post-fix file content for each changed file — not a \
partial snippet. Apply the suggested fix to the actual file you read.
  4. Propose a branch name starting with `costsense/` (e.g. \
`costsense/reduce-lambda-memory`).
  5. Write a concise PR title and a markdown body that cites the issue, \
reason, and estimated savings.

Rules:
- ONLY edit repos in ALLOWED_REPOS.
- If you cannot find a concrete file to change, set `"error"` explaining why.
- Do not guess file paths — read them via tools first.
- At most 3 files changed.
- Return ONLY JSON (no prose, no code fences):

{
  "repo": "org/repo",
  "branch": "costsense/short-slug",
  "title": "fix: ...",
  "body": "markdown PR description",
  "files": [{"path": "relative/path", "content": "full file text"}],
  "error": null
}"""


@dataclass
class FileChange:
    path: str
    content: str


@dataclass
class PrFixPlan:
    repo: str = ""
    branch: str = ""
    title: str = ""
    body: str = ""
    files: list[FileChange] = field(default_factory=list)
    error: str | None = None
    tool_calls: int = 0
    model_id: str = ""
    raw_text: str = ""


def _pr_fix_specs() -> list[dict]:
    return [s for s in all_github_specs() if s["name"] in _PR_FIX_TOOL_NAMES]


def _run_github_tool(name: str, args: dict) -> dict:
    if name not in GITHUB_TOOLS:
        return {"error": f"unknown tool: {name}"}
    fn, _ = GITHUB_TOOLS[name]
    try:
        result = fn(**(args or {}))
    except Exception as e:  # noqa: BLE001
        return {"error": f"tool crashed: {e}"}
    return scrub(result)


def _extract_json(text: str) -> dict:
    t = text.strip()
    for fence in ("```json", "```"):
        idx = t.find(fence)
        if idx >= 0:
            t = t[idx + len(fence):]
            break
    if "```" in t:
        t = t.split("```", 1)[0]
    try:
        return json.loads(t.strip())
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(t[start:i + 1])
        start = t.find("{", start + 1)
    raise json.JSONDecodeError("no JSON object", text, 0)


def _parse_plan(text: str, model_id: str, tool_calls: int) -> PrFixPlan:
    try:
        parsed = _extract_json(text)
    except json.JSONDecodeError as e:
        return PrFixPlan(
            model_id=model_id, tool_calls=tool_calls,
            error=f"no JSON in response: {e}", raw_text=text[:500],
        )

    err = parsed.get("error")
    if err:
        return PrFixPlan(
            model_id=model_id, tool_calls=tool_calls,
            error=str(err), raw_text=text[:500],
        )

    files: list[FileChange] = []
    for f in (parsed.get("files") or []):
        if not isinstance(f, dict):
            continue
        path = str(f.get("path", "")).strip()
        content = str(f.get("content", ""))
        if path and content:
            files.append(FileChange(path=path, content=content))

    repo = str(parsed.get("repo", "")).strip()
    branch = str(parsed.get("branch", "")).strip()
    if not branch.startswith("costsense/"):
        slug = slugify_branch(parsed.get("title") or repo or "fix")
        branch = f"costsense/{slug}"

    if not repo or not files:
        return PrFixPlan(
            model_id=model_id, tool_calls=tool_calls,
            error="Could not resolve target repo and file(s).",
            raw_text=text[:500],
        )

    return PrFixPlan(
        repo=repo,
        branch=branch,
        title=str(parsed.get("title", "CostSense cost fix")).strip(),
        body=str(parsed.get("body", "")).strip(),
        files=files,
        tool_calls=tool_calls,
        model_id=model_id,
        raw_text=text,
    )


def plan_pr_fix(
    action: Action,
    approach: Approach,
    allowed_repos: list[str],
    model_id: str = DEFAULT_MODEL,
) -> PrFixPlan:
    """Run the GitHub-only agent to produce a previewable PR plan."""
    if not allowed_repos:
        return PrFixPlan(error="No repos were scanned — pick repos and Analyze first.")

    client = make_client(None, region=BEDROCK_REGION)
    user_msg = (
        f"ALLOWED_REPOS: {json.dumps(allowed_repos)}\n\n"
        f"ISSUE: {action.issue}\n"
        f"REASON: {action.reason}\n"
        f"RECOMMENDATION: {action.recommendation}\n"
        f"CATEGORY: {action.category}\n"
        f"EST_DAILY_SAVINGS_USD: {action.est_daily_savings_usd}\n"
        f"SOURCE: {action.source}\n\n"
        f"SUGGESTED FIX TITLE: {approach.title}\n"
        f"SUGGESTED FIX DESCRIPTION: {approach.description}\n"
        f"SUGGESTED CODE ({approach.language}):\n{approach.code}\n\n"
        "Locate the real file(s), apply this fix, and return the JSON plan."
    )
    messages: list = [{"role": "user", "content": user_msg}]
    tool_calls = 0
    specs = _pr_fix_specs()

    for _ in range(MAX_TOOL_TURNS + 1):
        try:
            resp = client.invoke_model(
                modelId=model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.0,
                    "system": SYSTEM,
                    "tools": specs,
                    "messages": messages,
                }),
            )
            payload = json.loads(resp["body"].read())
        except Exception as e:  # noqa: BLE001
            return PrFixPlan(error=f"bedrock invoke failed: {e}")

        content = payload.get("content", [])
        tool_use = [b for b in content if b.get("type") == "tool_use"]

        if tool_use:
            messages.append({"role": "assistant", "content": content})
            results = []
            for blk in tool_use:
                tool_calls += 1
                out = _run_github_tool(blk["name"], blk.get("input") or {})
                results.append({
                    "type": "tool_result",
                    "tool_use_id": blk["id"],
                    "content": json.dumps(out, default=str)[:8000],
                })
            messages.append({"role": "user", "content": results})
            continue

        text_block = next((b for b in content if b.get("type") == "text"), None)
        if not text_block:
            return PrFixPlan(
                model_id=model_id, tool_calls=tool_calls,
                error="no text in final response",
            )
        plan = _parse_plan(text_block.get("text", ""), model_id, tool_calls)
        if plan.repo and plan.repo not in allowed_repos:
            return PrFixPlan(
                model_id=model_id, tool_calls=tool_calls,
                error=f"Agent picked {plan.repo!r} which is not in ALLOWED_REPOS.",
            )
        return plan

    return PrFixPlan(
        model_id=model_id, tool_calls=tool_calls,
        error=f"exceeded {MAX_TOOL_TURNS} tool turns",
    )
