"""Conversational FinOps bot with read-only AWS + GitHub access.

Maintains a message history across turns so the user can ask follow-ups.
Every tool call goes through `scrub()` (from aws_tools_broad) which recursively
strips anything that looks like a secret, IAM policy document, JWT, private
key, or AWS access key ID — before Claude sees the tool output.

Hallucination hardening (added 2026-07):
    LLMs will fabricate dollar figures when a tool errors, or pivot to the
    connected account when the user asked about a different one, even with
    a system prompt telling them not to. Four layers of defense:
      1. Every tool_result whose payload contains an ``error`` field is sent
         back to Claude with Anthropic's ``is_error: true`` flag.
      2. The system prompt injects the ACTIVE profile + account id + a
         signal for whether GitHub read is configured, so the model knows
         exactly what it can and cannot reach.
      3. Denial-guard: if any tool_result was is_error=no_access AND the
         reply contains a $-figure, override the reply with an honest "I
         don't have access" message.
      4. Substitution-guard: if the reply admits it couldn't answer the
         user's target (contains phrases like "I don't have access",
         "different account", "connected account") but ALSO contains
         $-figures or a services table sourced from the CURRENT account,
         override the reply — the model tried to be helpful by handing
         over the connected account's data as a consolation, which is
         the exact behavior we don't want.

Public API:
    chat_step(profile, model_id, history, user_msg, account_id,
              github_read_available) -> ChatTurn

`history` is the running list of {role, content} messages Claude will see.
`ChatTurn` bundles the assistant reply text, tool-call transcript, and the
updated history to store back in session_state.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import boto3

from src.ai_agent.aws_tools import TOOLS as BASE_TOOLS
from src.ai_agent.aws_tools_broad import BROAD_TOOLS, all_broad_specs, scrub
from src.ai_agent.github_tools import GITHUB_TOOLS, all_github_specs


BEDROCK_REGION = "us-west-2"
DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"
MAX_TOOL_TURNS_PER_QUESTION = 12
MAX_TOKENS = 3000


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------
#
# Every tool in aws_tools / aws_tools_broad / github_tools wraps its
# implementation in a `_safe` decorator that returns
# ``{"error": "<ExceptionClassName>: <message>"}`` when boto3 or the GitHub
# API raises. We inspect that string to bucket failures — the buckets drive
# both the is_error flag we send to Claude and the post-loop guard.

_NO_ACCESS_MARKERS = (
    "AccessDenied",
    "UnauthorizedOperation",
    "AuthFailure",
    "InvalidClientTokenId",
    "SignatureDoesNotMatch",
    "NoCredentials",
    "TokenRefreshRequired",
    "ExpiredToken",
    "SSOTokenLoadError",
    "UnrecognizedClientException",
    "not authorized",
    "is not authorized to perform",
    "403",
)

_TRANSIENT_MARKERS = (
    "Throttling",
    "ThrottlingException",
    "RequestLimitExceeded",
    "TooManyRequests",
    "ServiceUnavailable",
    "InternalServerError",
    "500",
    "502",
    "503",
    "504",
    "EndpointConnectionError",
    "ReadTimeoutError",
    "ConnectTimeoutError",
)

_NOT_FOUND_MARKERS = (
    "NoSuchEntity",
    "NotFound",
    "ResourceNotFoundException",
    "NoSuchBucket",
    "InvalidParameterValueException",
    "404",
)


def _classify_error(error_text: str) -> str:
    """Bucket a raw error string into a coarse kind. The buckets are
    deliberately coarse: the goal is to detect *access denial* reliably so
    the post-loop guard can catch fabrications; other kinds are informational.
    """
    if not error_text:
        return "unknown"
    lowered = str(error_text)
    for marker in _NO_ACCESS_MARKERS:
        if marker in lowered:
            return "no_access"
    for marker in _TRANSIENT_MARKERS:
        if marker in lowered:
            return "transient"
    for marker in _NOT_FOUND_MARKERS:
        if marker in lowered:
            return "not_found"
    return "unknown"


def _tool_result_meta(result) -> tuple[bool, str | None, str | None]:
    """Return ``(is_error, error_kind, error_text)`` for a tool result.

    Success case returns ``(False, None, None)``. Any dict with a truthy
    ``error`` key is treated as a failure.
    """
    if isinstance(result, dict) and result.get("error"):
        error_text = str(result["error"])
        return True, _classify_error(error_text), error_text
    return False, None, None


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_BASE = """You are CostSense — a senior AWS FinOps analyst embedded in a \
chat interface. The user will ask questions about their AWS account and you \
will ground every answer in real data by calling the read-only tools you \
have available.

Your AWS tools cover: Cost Explorer, CloudWatch metrics, CloudTrail events, \
resource inventory (Lambda, RDS, EC2, NAT, EBS, S3, DynamoDB), AWS Compute \
Optimizer, AWS Budgets, Service Quotas, S3 lifecycle policies, and rightsizing \
recommendations. These tools ONLY work against the currently connected AWS \
account (see ACTIVE SCOPE above). If the user asks about a different \
account, you have NO way to reach it — say so and stop.

GitHub tools: read-only access to repos (search, list files, get file \
contents, search code, list/diff pull requests). Their availability is \
signalled in the ACTIVE SCOPE block above. If GitHub read is available you \
CAN and SHOULD use these tools directly; if it's marked unavailable, tell \
the user GitHub isn't reachable — do NOT attempt the tools anyway.

Rules:
- NEVER guess. When the user asks "what's my biggest cost driver," call \
`cost_by_service` and report the actual top service. When they ask "is my \
Lambda oversized," call `compute_optimizer_lambda`.
- If the user references a repo by name only (not "org/repo"), ALWAYS call \
`github_search_repositories` first to resolve it — it checks the \
DiligentCorp org first, then falls back to a global search — then use the \
matched `full_name` in the other `github_*` tools. Only tell the user you \
couldn't find a repo after this search comes back empty.
- To analyze a repo's cost impact (e.g. "analyse the cost of X repo"), read \
its IaC/config files (`github_get_file` on Terraform/CDK/Dockerfiles, etc.) \
and cross-reference the resources you find with AWS tools (cost_by_service, \
CloudWatch, resource inventory) to ground the analysis in real spend.
- Answer the question they asked. Don't hallucinate follow-up asks the user \
didn't make. Keep replies short and direct.
- When you cite a number or fact, say WHICH tool call it came from ("via \
cost_by_service") so the user can trust it.
- If a tool errors or returns empty, say so plainly. Don't pretend a value \
you fabricated came from a tool.
- Assume you cannot see secrets. Fields that come back as \
"[REDACTED-BY-COSTSENSE]" have been stripped by policy — don't demand them.
- If the user asks something truly out of scope (e.g. "write me a poem"), \
politely redirect to FinOps or the connected GitHub repos.

CRITICAL ANTI-HALLUCINATION RULES:
- If ANY tool_result comes back with is_error=true, treat that call as \
having produced NO DATA. You may not use its content as a source of numbers.
- If an is_error tool_result mentions access denial (AccessDenied, not \
authorized, NoCredentials, ExpiredToken), tell the user their profile lacks \
access to that resource — do NOT retry the same call and do NOT substitute \
a plausible number in its place.
- If ALL tools relevant to the user's question failed, your reply MUST NOT \
contain any dollar figure. Say plainly that you can't answer without \
access and suggest switching profiles or asking the account owner.
- Numbers you cannot ground in a successful tool_result MUST be omitted. \
Prose ("Lambda is your biggest driver") is fine when supported by a tool; \
figures ($400/day, ~$65K/month) are ONLY fine when a tool call actually \
returned them.

CRITICAL SCOPE RULES:
- The ACTIVE SCOPE block above names EXACTLY ONE AWS account you can \
reach. Your AWS tools return data from THAT account only. You have NO \
mechanism to see any other account.
- If the user asks about a DIFFERENT account (by name, id, or nickname \
like "policy manager", "acl-prod", "shared services"), you MUST refuse \
briefly and STOP. Reply is ONE short paragraph: "I don't have access to \
that account — my active profile is X (account Y). Switch profiles in \
the sidebar or ask the account's owner."
- DO NOT offer the connected account's data as a "here's what I can see \
instead" consolation. That is the specific failure mode this rule \
prevents. No table, no numbers, no "meanwhile", no "however here is". \
Full stop after the refusal.
- The refusal-and-stop rule applies even if the connected account's data \
would be interesting or useful. The user did not ask for it.

CHARTS — when to emit and how:
- If the user asks for a "graph", "chart", "bar chart", "trend", "plot", \
"visualise", or clearly wants a picture rather than a number, INCLUDE ONE \
chart in your reply. Do not emit a chart when the user only asked a \
question that resolves to a single number ("what's my current $/day") — \
answer with prose in that case.
- Chart format: a single fenced JSON block with the language tag `chart`. \
Example (do NOT copy the numbers — use YOUR OWN tool_result values):

```chart
{
  "type": "line",
  "title": "Daily spend by service, last 14 days",
  "x_title": "Day",
  "y_title": "USD",
  "series": [
    {"name": "AWS Lambda", "x": ["2026-07-10", "2026-07-11", "2026-07-12"],
     "y": [12.50, 14.20, 13.80]},
    {"name": "Amazon S3",  "x": ["2026-07-10", "2026-07-11", "2026-07-12"],
     "y": [8.10, 8.30, 8.20]}
  ],
  "source_tool": "cost_by_service"
}
```

- Supported types: "line" and "bar" only. Use "line" for time series, \
"bar" for categorical comparisons ("top services", "top instances").
- Every "y" value MUST come from a tool_result you received earlier in \
THIS turn. A post-loop guard cross-checks every point against the tool \
output; unverified values are stripped and the chart is REPLACED with a \
warning banner. Do not fabricate, average, extrapolate, or extend beyond \
the values a tool actually returned.
- `source_tool` must name the tool that produced the numbers (e.g. \
"cost_by_service", "fetch_daily_totals", "cloudwatch_metric"). This is \
what the user will see under the chart.
- Include a one-sentence prose summary above or below the chart so the \
user knows what to look at. The chart block itself renders separately.
- If you have no successful tool_result to plot, say so plainly and do \
NOT emit a chart block. An empty or fabricated chart is worse than no \
chart.

EVENT PREDICTIONS — customer/org onboarding, offboarding, migration ramp, \
backfill pulse, seasonal multiplier:

When the user asks a "what if" cost question about one of these events \
("if we onboard 50 orgs", "cost of backfilling 90 days", "seasonal peak \
impact", "if we migrate service X"), follow this workflow:

  1. Call `cost_by_service` or `fetch_daily_totals` for at least 90 days \
of history so you have enough data to detect prior events.
  2. LOOK for historical precedent in that data:
       * Onboarding / offboarding: a persistent step change in daily total \
around a known date, sustained for ≥ 14 days after the step.
       * Migration ramp: a gradual monotonic climb or fall in one service \
over 30–60 days.
       * Backfill pulse: a bounded spike (typically Athena, EMR, Redshift, \
Glue) that returned to baseline within 1–14 days.
       * Seasonal multiplier: a recurring % lift in the same weeks year- \
over-year (needs ≥ 60 days of history to even detect).
  3. IF you find precedent that matches the pattern:
       * Compute a per-unit rate from that precedent. Example: if adding \
orgs in April caused daily spend to move from $150 → $180 sustained \
for 20 days and the CRM shows 6 orgs onboarded in that window, the \
rate is $5/org/day.
       * State the rate, name the dates it came from, and give a number \
with a stated confidence.
       * ALSO cite the specific historical spend values you used (e.g. \
"$150/day pre-step, $180/day post-step, both from cost_by_service on \
2026-04-01 → 2026-04-30").
  4. IF you find NO precedent (the event type has not happened in the \
last 90 days of history):
       * Say verbatim: "There is no prior <event type> event in the last \
90 days of history for this account, so I can't predict from precedent."
       * Then offer: "If you want, I can give you a rough estimate based \
on current spend and typical assumptions, but it will not be accurate."
       * Do NOT fabricate a rate. Do NOT emit a chart.
       * Only after the user asks for the rough estimate, provide one \
and label every number in it as ASSUMED not measured.

CHARTS FOR EVENT PREDICTIONS — only when the user explicitly asks:

- The user must include a chart word ("chart", "graph", "bar chart", \
"visualise") in the question. If they only ask for a number, answer with \
prose — no chart.
- When you DO emit a prediction chart, use this exact 3-bar shape:

```chart
{
  "type": "bar",
  "title": "Impact of onboarding 50 orgs on daily spend",
  "x_title": "Scenario",
  "y_title": "USD / day",
  "series": [
    {"name": "Daily spend",
     "x": ["Current", "Change", "Projected"],
     "y": [180.11, 50.00, 230.11]}
  ],
  "source_tool": "cost_by_service",
  "prediction_basis": {
    "current_grounding": [180.11],
    "rate_grounding": [150.00, 180.00],
    "note": "Per-org rate $5.00/day from step-change on 2026-04-15 (6 orgs onboarded, $150 -> $180 sustained for 20 days). Applied to 50 orgs = +$50/day. Confidence: medium."
  }
}
```

- `prediction_basis.current_grounding` MUST list the tool_result values \
you used to establish "Current" (e.g. the current daily spend readings).
- `prediction_basis.rate_grounding` MUST list the tool_result values you \
used to establish the per-unit rate (typically the pre- and post-step \
daily spend from history).
- `prediction_basis.note` MUST name specific dates and dollar values from \
the tool_results, and state a confidence (low/medium/high).
- The 3rd bar (Projected) MUST equal Current + Change arithmetically \
(within $0.50 or 1%). A post-loop guard rejects the chart if the values \
don't match a tool output or the arithmetic is off — it will replace \
the chart with a red banner naming exactly what mismatched.
- No historical precedent found = NO chart. Emit only the honest refusal \
prose described above.

Response format: plain markdown. Use bullet points and short paragraphs. \
No JSON unless the user explicitly asks for JSON or the reply includes \
a `chart` fenced block per the rules above. Always write dollar figures \
with a leading "$" (e.g. "$400–800", "$65K/year", never a bare "400–800" \
or "65K/year") since these are all USD amounts."""


def _infer_repo_from_profile(profile: str | None) -> str | None:
    """Given an AWS SSO profile name like ``dil-data-platform-dev``, return
    the likely GitHub repo (``DiligentCorp/data-platform``) using the same
    ``normalize_profile`` helper that every other CostSense page uses.

    Returns None for shared/team profiles (``dil-team-*``) and unknown
    conventions. The bot is told this is a HINT — it should verify with
    ``github_search_repositories`` before treating the repo as confirmed.
    """
    try:
        from src.pr_scanner.profile_repo_match import normalize_profile
    except Exception:  # noqa: BLE001
        return None
    normalized = normalize_profile(profile or "")
    if not normalized:
        return None
    return f"DiligentCorp/{normalized}"


def _build_system(
    profile: str | None,
    account_id: str | None,
    github_read_available: bool = False,
) -> str:
    """Prepend the active-scope block. Front-loading the profile + account
    + GitHub-availability flag + inferred repo before the rules block
    anchors the model to the correct scope earlier in the context window."""
    scope_lines = ["ACTIVE SCOPE:"]
    if profile:
        scope_lines.append(f"- AWS profile: {profile}")
    else:
        scope_lines.append("- AWS profile: (none — tools will fail)")
    if account_id:
        scope_lines.append(f"- AWS account id: {account_id}")
    else:
        scope_lines.append(
            "- AWS account id: (unknown — the profile could not resolve "
            "get-caller-identity; assume you have no AWS access)"
        )
    # Inferred repo hint: `dil-data-platform-dev` -> `DiligentCorp/data-platform`.
    # This is the same convention every other page (Anomalies, PR Predictor,
    # Close the Loop) uses via `normalize_profile`. Adding it here so the
    # chatbot doesn't ask "which repo?" for questions where the answer is
    # already implied by the active profile.
    inferred_repo = _infer_repo_from_profile(profile) if github_read_available else None
    if inferred_repo:
        scope_lines.append(
            f"- Inferred repo (from profile-name convention): {inferred_repo}. "
            f"Treat this as a HINT: use it as the default when the user "
            f"asks about 'the repo', 'this account's code', or an event "
            f"prediction question ('if we onboard N orgs', 'cost of "
            f"migration X'). Verify with `github_search_repositories` "
            f"before quoting file contents or PR numbers, but do NOT ask "
            f"the user which repo it is unless the search comes back "
            f"empty or you have specific reason to think the inference "
            f"is wrong."
        )
    if github_read_available:
        scope_lines.append(
            "- GitHub read: AVAILABLE (github_* tools will succeed)"
        )
    else:
        scope_lines.append(
            "- GitHub read: UNAVAILABLE (no GITHUB_TOKEN configured and "
            "no gh CLI — do NOT call any github_* tool; if asked about a "
            "repo, tell the user GitHub isn't reachable)"
        )
    scope_lines.append(
        "You MUST NOT report figures for any OTHER account. If the user "
        "asks about a different account id, tell them to switch profiles "
        "in the sidebar — do not attempt to answer."
    )
    scope_lines.append(
        "Tool names: the tool list at the top of this turn is exhaustive. "
        "Do NOT invent tool names ('precedent_lookup', 'cost_forecast', "
        "'repo_analyzer', etc.) — if a capability you want isn't in the "
        "tool list, say so plainly rather than pretending you tried to "
        "call something that doesn't exist."
    )
    return "\n".join(scope_lines) + "\n\n" + _SYSTEM_BASE


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    name: str
    input: dict
    output_summary: str        # first ~200 chars of the (scrubbed) result
    is_error: bool = False
    error_kind: str | None = None
    error_text: str | None = None


@dataclass
class ChatTurn:
    reply: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    updated_history: list[dict] = field(default_factory=list)
    error: str | None = None
    model_id: str = ""
    guard_triggered: bool = False   # True when post-loop guard rewrote reply
    guard_reason: str | None = None


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def _merged_tool_specs() -> list[dict]:
    """All tools available to the chat agent: AWS base + broad + GitHub."""
    from src.ai_agent.aws_tools import tool_specs as base_specs
    return base_specs() + all_broad_specs() + all_github_specs()


def _run_tool(name: str, args: dict, profile: str | None) -> dict:
    """Route to whichever registry has the tool, then scrub secrets."""
    kwargs = dict(args or {})
    if name in BASE_TOOLS:
        fn, _ = BASE_TOOLS[name]
        kwargs["profile"] = profile
    elif name in BROAD_TOOLS:
        fn, _ = BROAD_TOOLS[name]
        kwargs["profile"] = profile
    elif name in GITHUB_TOOLS:
        fn, _ = GITHUB_TOOLS[name]  # GitHub tools auth via gh CLI / GITHUB_TOKEN
    else:
        return {"error": f"unknown tool: {name}"}
    try:
        result = fn(**kwargs)
    except Exception as e:  # noqa: BLE001
        return {"error": f"tool crashed: {e}"}
    return scrub(result)


def _summarize_output(obj) -> str:
    s = json.dumps(obj, default=str)
    return s[:220] + ("…" if len(s) > 220 else "")


# ---------------------------------------------------------------------------
# Post-loop hallucination guard
# ---------------------------------------------------------------------------

# Matches "$400", "$1,200.50", "$65K", "$65k", "$1.2M", "$400/day", etc.
_DOLLAR_FIGURE_RE = re.compile(r"\$\s?[\d][\d,\.]*\s?[KkMmBb]?")

# Phrases the model uses when it's admitting it couldn't answer the user's
# original target. Detected case-insensitively. The list is deliberately
# conservative — false positives (guard firing on a legitimate reply) mean
# the user sees the honest fallback instead of a real answer, and that is
# strictly less bad than the substitution behavior we're stopping.
_SCOPE_REFUSAL_PHRASES = (
    "i don't have access",
    "i do not have access",
    "i don't have visibility",
    "i do not have visibility",
    "different account",
    "another account",
    "active aws profile",
    "active profile is",
    "currently connected",
    "connected account",
    "connected to",
    "switch profiles",
    "asked about",
    "asked about the",
    "scope notice",
    "account scope",
)


def _reply_has_refusal_language(reply: str) -> bool:
    """True when the reply text contains a phrase indicating the model
    admitted a scope mismatch — even if it then handed over consolation
    data anyway."""
    if not reply:
        return False
    lowered = reply.lower()
    return any(phrase in lowered for phrase in _SCOPE_REFUSAL_PHRASES)


def _apply_denial_guard(
    reply: str,
    tool_calls: list[ToolCall],
    profile: str | None,
    account_id: str | None,
    github_read_available: bool,
) -> tuple[str, bool, str | None]:
    """If any tool call failed with ``no_access`` AND the reply contains a
    dollar figure, replace the reply with an honest denial. This is the
    deterministic backstop for the anti-hallucination rules in the system
    prompt — it catches cases where the model produces $-figures anyway.

    Returns ``(new_reply, guard_triggered, reason)``.
    """
    denied_tools = [tc for tc in tool_calls
                    if tc.is_error and tc.error_kind == "no_access"]
    if not denied_tools:
        return reply, False, None

    figures = _DOLLAR_FIGURE_RE.findall(reply or "")
    if not figures:
        return reply, False, None

    tool_names = sorted({tc.name for tc in denied_tools})
    honest_reply = _honest_denial_message(
        profile, account_id, tool_names, github_read_available,
    )
    reason = (
        f"denial guard: reply contained {len(figures)} $-figure(s) but "
        f"{len(denied_tools)} tool call(s) were denied: "
        f"{', '.join(tool_names)}"
    )
    return honest_reply, True, reason


def _apply_substitution_guard(
    reply: str,
    tool_calls: list[ToolCall],
    profile: str | None,
    account_id: str | None,
    github_read_available: bool,
) -> tuple[str, bool, str | None]:
    """When the reply admits "I don't have access to X" but ALSO includes
    $-figures from a successful tool call on the currently-connected
    account, the model has substituted the connected account's data as a
    consolation. Rewrite to a clean refusal.

    Returns ``(new_reply, guard_triggered, reason)``.
    """
    if not _reply_has_refusal_language(reply):
        return reply, False, None

    if not _DOLLAR_FIGURE_RE.search(reply or ""):
        # Model correctly refused with no numbers. Nothing to do.
        return reply, False, None

    honest_reply = _honest_scope_message(
        profile, account_id, github_read_available,
    )
    reason = (
        "substitution guard: reply used scope-refusal language "
        "(\"I don't have access\" / \"different account\" / \"connected "
        "account\") but ALSO contained a dollar figure. The model refused "
        "the user's target and then handed over the current account's "
        "data as a consolation — that is the exact behavior this guard "
        "stops."
    )
    return honest_reply, True, reason


def _honest_denial_message(
    profile: str | None,
    account_id: str | None,
    denied_tool_names: list[str],
    github_read_available: bool,
) -> str:
    """The fallback message when AWS explicitly denied a tool call."""
    scope = (f"profile `{profile}` (account {account_id})"
             if account_id else f"profile `{profile or '(none)'}`")
    gh_line = (
        "\n\nGitHub read IS available on this session — if the question "
        "can be answered from repo contents, ask me to check the code "
        "and I can help there."
        if github_read_available else
        "\n\nGitHub read is also not configured on this session, so I "
        "can't fall back to code inspection either."
    )
    return (
        f"**I don't have AWS access for that.** {scope} was denied by "
        f"IAM on: {', '.join(denied_tool_names)}.\n\n"
        f"Any dollar figures for this question would be a guess, not real "
        f"data. Switch to a profile that has read access to this account, "
        f"or ask the account's owner to run the question — I'll ground "
        f"the answer in the actual API response."
        f"{gh_line}"
    )


def _honest_scope_message(
    profile: str | None,
    account_id: str | None,
    github_read_available: bool,
) -> str:
    """The fallback message when the user asked about a DIFFERENT account
    (or one my active profile can't reach)."""
    scope = (f"`{profile}` (account {account_id})"
             if account_id else f"`{profile or '(none)'}`")
    gh_line = (
        "GitHub read IS available on this session, so I can help with "
        "questions answerable from repo contents (code, IaC, PRs) even "
        "when the AWS account is out of reach."
        if github_read_available else
        "GitHub read is also not configured on this session — I can't "
        "fall back to code inspection either."
    )
    return (
        f"**I don't have access to that account.** My active AWS profile "
        f"is {scope} and my tools only reach that one account. I have no "
        f"way to see cost, resources, or activity in a different account.\n\n"
        f"To answer this question:\n"
        f"- Switch to the profile that owns the target account (Account "
        f"selector in the sidebar), or\n"
        f"- Ask the account's owner to run the question on their profile.\n\n"
        f"{gh_line}"
    )


def _apply_hallucination_guard(
    reply: str,
    tool_calls: list[ToolCall],
    profile: str | None,
    account_id: str | None,
    github_read_available: bool = False,
) -> tuple[str, bool, str | None]:
    """Compose both guards. Denial guard runs first (a denied tool call is
    stronger evidence than refusal language), then substitution guard.

    Returns ``(new_reply, guard_triggered, reason)``.
    """
    reply, triggered, reason = _apply_denial_guard(
        reply, tool_calls, profile, account_id, github_read_available,
    )
    if triggered:
        return reply, triggered, reason
    return _apply_substitution_guard(
        reply, tool_calls, profile, account_id, github_read_available,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _detect_github_read_available() -> bool:
    """True when the chat page can plausibly reach GitHub — either a
    GITHUB_TOKEN/GH_TOKEN is set OR the `gh` CLI is on PATH. This mirrors
    what the github_tools actually depend on. Signalling this into the
    system prompt lets Claude honestly claim / disclaim GitHub access."""
    try:
        from src.pr_scanner.gh_client import gh_available, token_configured
        return bool(token_configured() or gh_available())
    except Exception:  # noqa: BLE001
        return False


def chat_step(
    profile: str | None,
    model_id: str,
    history: list[dict],
    user_msg: str,
    account_id: str | None = None,
    github_read_available: bool | None = None,
) -> ChatTurn:
    """Advance one user turn. Runs the tool-use loop for this question and
    returns the assistant reply plus updated history.

    ``account_id`` is the id resolved from ``profile`` via STS
    get-caller-identity. It's used to scope the system prompt and to phrase
    the denial message when the post-loop guard fires. Callers that don't
    have it can pass ``None`` — the guard still works, it just can't name
    the account in the denial.

    ``github_read_available`` controls the GitHub-availability signal in
    the system prompt. If ``None`` (the default), we probe locally with
    ``_detect_github_read_available()``. Callers that already know the
    answer (e.g. a UI that renders a "GitHub connected" indicator) can
    pass it in directly to avoid re-probing.
    """
    from src.ai_agent.bedrock_client import make_client
    client = make_client(profile, region=BEDROCK_REGION)

    if github_read_available is None:
        github_read_available = _detect_github_read_available()
    system_prompt = _build_system(profile, account_id, github_read_available)
    messages = list(history) + [{"role": "user", "content": user_msg}]
    tool_calls: list[ToolCall] = []

    for _ in range(MAX_TOOL_TURNS_PER_QUESTION + 1):
        try:
            resp = client.invoke_model(
                modelId=model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.2,
                    "system": system_prompt,
                    "tools": _merged_tool_specs(),
                    "messages": messages,
                }),
            )
            payload = json.loads(resp["body"].read())
        except Exception as e:  # noqa: BLE001
            return ChatTurn(reply="", error=f"bedrock invoke failed: {e}",
                            model_id=model_id,
                            updated_history=history + [
                                {"role": "user", "content": user_msg}])

        content = payload.get("content", [])
        tool_use_blocks = [b for b in content if b.get("type") == "tool_use"]

        if tool_use_blocks:
            messages.append({"role": "assistant", "content": content})
            results = []
            for blk in tool_use_blocks:
                out = _run_tool(blk["name"], blk.get("input") or {}, profile)
                is_err, err_kind, err_text = _tool_result_meta(out)
                tool_calls.append(ToolCall(
                    name=blk["name"],
                    input=blk.get("input") or {},
                    output_summary=_summarize_output(out),
                    is_error=is_err,
                    error_kind=err_kind,
                    error_text=err_text,
                ))
                # Cap payload sent back to Claude
                serialized = json.dumps(out, default=str)
                if len(serialized) > 8000:
                    if isinstance(out, dict):
                        trimmed = {k: (f"<large list, len={len(v)}>"
                                       if isinstance(v, list) and len(v) > 30
                                       else v)
                                   for k, v in out.items()}
                        serialized = json.dumps(trimmed, default=str)[:8000]
                    else:
                        serialized = serialized[:8000] + "…[truncated]"
                result_block = {
                    "type": "tool_result",
                    "tool_use_id": blk["id"],
                    "content": serialized,
                }
                # Anthropic Messages API: setting is_error=true on the
                # tool_result changes how the model treats the payload —
                # it's much less likely to hallucinate a value from an
                # explicitly-erroring call than from a JSON blob that
                # happens to contain the word "error".
                if is_err:
                    result_block["is_error"] = True
                results.append(result_block)
            messages.append({"role": "user", "content": results})
            continue

        # Terminal — no more tool calls, model wants to reply.
        text_block = next((b for b in content if b.get("type") == "text"), None)
        reply = text_block.get("text", "") if text_block else "(no reply)"

        # Post-loop hallucination guard: two rules, both deterministic.
        # (1) Denial guard: denied tool + $-figure in reply -> honest denial.
        # (2) Substitution guard: refusal language + $-figure in reply ->
        #     honest scope message (catches the "here's the connected
        #     account's data instead" pattern).
        # No LLM in either guard, so they can't be talked around.
        reply, guard_triggered, guard_reason = _apply_hallucination_guard(
            reply, tool_calls, profile, account_id, github_read_available,
        )

        # Persist the final assistant turn in the history so follow-ups see
        # the OVERRIDDEN reply, not the model's original fabrication.
        if guard_triggered:
            # Replace the text block in the assistant message before saving
            # to history, so subsequent turns don't see the fabricated one.
            content = [
                {"type": "text", "text": reply}
                if b.get("type") == "text" else b
                for b in content
            ]
            if not any(b.get("type") == "text" for b in content):
                content.append({"type": "text", "text": reply})
        messages.append({"role": "assistant", "content": content})
        return ChatTurn(
            reply=reply,
            tool_calls=tool_calls,
            updated_history=messages,
            model_id=model_id,
            guard_triggered=guard_triggered,
            guard_reason=guard_reason,
        )

    return ChatTurn(
        reply="",
        tool_calls=tool_calls,
        updated_history=messages,
        error=f"exceeded {MAX_TOOL_TURNS_PER_QUESTION} tool turns without "
              "a final reply",
        model_id=model_id,
    )
