"""Deep-AWS Bedrock agent.

Runs a tool-use loop over the AWS read-only tools in `aws_tools`. Two entry
points that share the same loop:

  - `analyze_pr(pr_url)` — Fetches the PR diff, asks the agent to predict
    the cost impact and suggest changes that would reduce it.
  - `recommend_account(...)` — Free-form "audit this account for
    cost-reduction opportunities."
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

import boto3

from src.ai_agent.aws_tools import call_tool, tool_specs
from src.pr_scanner.gh_client import pr_diff, pr_view_json


DEFAULT_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"
BEDROCK_REGION = "us-west-2"
MAX_TOOL_TURNS = 16
MAX_TOKENS = 3500
# 3 was too low: it lets Haiku call 3 shallow tools and declare "neutral".
# 5 forces a real exploration path — at minimum: cost_by_service + one
# CloudWatch call per major resource + verification of at least one metric.
MIN_TOOL_CALLS_FOR_VERDICT = 5
# An extra bar for "neutral" verdicts: they're the model's escape hatch
# when it doesn't want to compute. Force MORE evidence for those.
MIN_TOOL_CALLS_FOR_NEUTRAL_VERDICT = 7


@dataclass
class Finding:
    resource: str
    action: str                  # "reduce" | "remove" | "resize" | "review"
    est_daily_delta_usd: float   # negative for savings
    rationale: str
    confidence: str = "medium"   # low | medium | high
    # Populated for recommendations only — the concrete before/after code and
    # the trade-offs, so the PR author can decide without guessing.
    current_code: str | None = None
    recommended_code: str | None = None
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)


@dataclass
class AgentVerdict:
    verdict: str = ""            # short one-line answer
    detail: str = ""             # longer explanation
    est_daily_delta_usd: float = 0.0
    est_daily_delta_low_usd: float | None = None
    est_daily_delta_high_usd: float | None = None
    direction: str = "unknown"   # "increase" | "decrease" | "neutral"
    confidence: str = "medium"   # "low" | "medium" | "high"
    measured: bool = True        # False when the number is a generic-rate estimate
    estimation_basis: str = "measured"  # measured | sibling_account | generic_rate | unknown
    findings: list[Finding] = field(default_factory=list)
    recommendations: list[Finding] = field(default_factory=list)
    tool_calls: int = 0
    model_id: str = ""
    error: str | None = None
    raw_text: str = ""


@lru_cache(maxsize=1)
def _client(profile: str | None):
    from src.ai_agent.bedrock_client import make_client
    return make_client(profile, region=BEDROCK_REGION)


def fetch_pr_diff(pr_url: str) -> tuple[str, str, str, str]:
    """Return (repo, pr_number, diff_text, title). Uses `gh` or GitHub API."""
    repo, num = parse_pr_url(pr_url)
    title = ""
    try:
        info = pr_view_json(repo, num)
        title = info.get("title", "")
    except Exception:  # noqa: BLE001
        pass
    diff = pr_diff(repo, num)
    return repo, str(num), diff, title


def parse_pr_url(url: str) -> tuple[str, int]:
    """Extract (org/repo, pr_number) from a GitHub PR URL."""
    m = re.search(r"github\.com/([^/]+/[^/]+)/pull/(\d+)", url.strip())
    if not m:
        raise ValueError(f"not a GitHub PR URL: {url!r}")
    return m.group(1), int(m.group(2))


SYSTEM_PR = """You are a senior AWS FinOps engineer reviewing a pull request \
before it merges. Your job is to answer three questions with EVIDENCE, not \
priors:
  1. Will this PR increase or decrease AWS cost?
  2. By roughly how much per day?
  3. What SPECIFIC changes to the PR would reduce cost further?

You have read-only tools to inspect the AWS account this repo deploys to:
CloudWatch metrics (Invocations, Duration, CPUUtilization, request counts), \
CloudTrail events, resource inventory (Lambda / RDS / EC2 / NAT / EBS), \
Cost Explorer by service, and AWS Rightsizing recommendations.

MANDATORY WORKFLOW — do NOT skip steps:

  Step 1 (ALWAYS): For every AWS resource named or referenced in the diff \
(Lambda function, RDS instance, ECS service, S3 bucket, DynamoDB table, \
CloudWatch metric emissions, log retention, etc.), call \
`get_cloudwatch_metric` to fetch its recent usage. Examples:
    - Lambda tweak → Invocations (Sum) AND Duration (Average) for the last 7d
    - RDS tweak → CPUUtilization, DatabaseConnections
    - CloudWatch metric-emission change → PutMetricData count via CloudTrail
    - Log retention change → IncomingLogEvents on the affected log group
  If the tool returns empty for a name, TRY VARIATIONS (with/without env \
prefix, with/without stage suffix) before giving up.

  Step 2: Call `cost_by_service` to see which services on this account \
currently spend money. This tells you which parts of the diff actually \
matter for the bill.

  Step 3: Only after Steps 1 and 2, compute the $ delta using the pricing \
formulas below, and write your final JSON.

DO NOT claim "very low invocation volume" or "cost impact is negligible" \
based on the resource NAME. You must have the CloudWatch numbers in hand \
before making that claim. Naming is not evidence.

Minimum tool-call budget for a substantive verdict: 3 tool calls. If you \
finish with fewer than 3 tool calls, add more findings/recommendations \
gathered via additional tool exploration — search for related resources, \
check other services touched, look at metric emissions patterns.

Pricing formulas to use once you have the CloudWatch data:
  - Lambda memory delta: invocations/day × avg_duration_sec × \
(new_gb − old_gb) × 0.0000166667 $/GB-s
  - Provisioned concurrency delta: (new_pc − old_pc) × 0.000004097 × 86400
  - CloudWatch custom metric emission: 0.30 per million PutMetricData, so \
batching N metrics saves (baseline_puts − new_puts) × 0.30 / 1e6 per day
  - Log retention change: (old_days − new_days) × avg_daily_ingest_gb × 0.03
  - Instance-type change: use rightsizing_recommendations, or compute new \
vs old hourly × 24

CLASSIFY THE PR FIRST. Read the diff carefully before you assume its \
cost shape. Common categories:
  - resource change: a Lambda/RDS/EC2/S3/... resource is created, resized, \
    or removed → use the pricing formulas below.
  - logging change: log-level / retention / log-statement counts change → \
    see LOGGING PRs.
  - refactor / bug fix / rename: code moves around but no new resources \
    are created and no more tenants/customers get processed → cost delta \
    is typically zero. Set `direction="neutral"`, \
    `est_daily_delta_usd=0`, `confidence="medium"`, and briefly explain.
  - scope expansion: the diff adds new entries (org IDs, tenant IDs, \
    customer IDs, region codes, feature flags) to a WHITELIST / \
    ALLOWLIST / CONFIG COLLECTION so that an existing service now \
    processes more work. Read the diff literally: a run of `+` lines \
    that are just comma-separated values inside a Python/TS/JSON \
    ARRAY LITERAL is scope expansion. `+` lines that are function \
    arguments, imports, or logic are NOT scope expansion — even if \
    they look list-like.

FOR SCOPE-EXPANSION PRs SPECIFICALLY:
  1. Count the new entries yourself from the diff.
  2. Call the `precedent_lookup` tool. It re-scans the repo for prior \
     merged PRs that grew the same file(s), and measures the step change \
     in daily spend that each caused in sibling AWS accounts (dev/ \
     staging/prod of the same repo) that ARE locally reachable.
  3. If precedent_lookup returns `usable: true`, multiply its \
     `per_entry_daily_usd` by the number of entries THIS PR adds. Set \
     `estimation_basis="sibling_account"`, `measured=false`, \
     `confidence="medium"`. In `detail`, cite the precedent PR by \
     number and the sibling profile name.
  4. If precedent_lookup returns `usable: false`, do NOT invent a rate. \
     Set `direction="unknown"`, `est_daily_delta_usd=0`, \
     `estimation_basis="unknown"`, `confidence="low"`, and in `detail` \
     say: "This PR expands scope but no historical precedent or \
     reachable sibling AWS account is available, so the true delta \
     (which is > $0/day) cannot be quantified from available data."
  5. NEVER fabricate a numeric range from thin air. Numbers must trace \
     to either measured metrics or a precedent-derived rate.

LOGGING PRs (logger.info → logger.debug, adding/removing log statements, \
raising/lowering log level, changing what's logged): these ARE cost-affecting \
if prod runs at LOG_LEVEL=INFO. CloudWatch Logs ingestion is $0.50/GB and \
storage is $0.03/GB/month. Your workflow for logging PRs:
  1. Identify each affected log group (usually /aws/lambda/<function-name>).
  2. Call get_cloudwatch_metric with namespace='AWS/Logs' and \
metric_name='IncomingBytes', dimensions=[{Name:'LogGroupName',Value:...}], \
statistic='Sum', days=7. This gives current daily ingest.
  3. Count the log lines being demoted/added in the diff.
  4. Estimate average bytes per line (INFO lines are typically 100-500 \
bytes; lines that log full queries can be 1-10 KB).
  5. delta_bytes_per_day = lines_demoted × invocations_per_day × avg_bytes.
  6. delta_$/day = delta_bytes_per_day / (1024^3) × 0.50.
Do NOT return "logs are free" — they are not.

Return ONLY a JSON object with this shape (no prose outside JSON):
{
  "verdict": "one-line summary — 'this PR will INCREASE daily cost by ~$X'",
  "direction": "increase" | "decrease" | "neutral",
  "est_daily_delta_usd": <signed number, negative = savings>,
  "est_daily_delta_low_usd": <lower bound of the estimate range, omit or
                              set equal to est_daily_delta_usd when the
                              number came from measured metrics>,
  "est_daily_delta_high_usd": <upper bound of the estimate range>,
  "confidence": "low" | "medium" | "high",
  "measured": true | false,
  "estimation_basis": "measured" | "sibling_account" | "generic_rate" |
                      "unknown",
  "detail": "2-3 SHORT sentences in plain English explaining WHY, as if to a "
            "non-technical stakeholder. No jargon walls, no run-on "
            "sentences, no nested math. State the single biggest driver "
            "and the bottom line — that's it. If you couldn't measure "
            "and estimated from generic rates, say so.",
  "findings": [
    {"resource": "aws_lambda_function/bulkIngest", "action": "resize",
     "est_daily_delta_usd": -6.4, "rationale": "...", "confidence": "high"}
  ],
  "recommendations": [
    {"resource": "same resource or new one", "action": "resize",
     "est_daily_delta_usd": <further savings if the user applied this>,
     "rationale": "ONE short sentence — the concrete suggestion",
     "confidence": "medium",
     "current_code": "the actual existing snippet (Terraform/CDK/YAML/etc, "
                     "pulled from the diff or inferred) that causes the "
                     "cost — null if you can't identify a real snippet",
     "recommended_code": "the exact replacement snippet implementing this "
                         "recommendation — same language/format as "
                         "current_code, null if not applicable",
     "pros": ["1-3 short bullet strings — concrete benefits"],
     "cons": ["1-3 short bullet strings — real trade-offs/risks, e.g. "
             "'cold starts increase', 'requires re-deploy'; empty list "
             "only if there are truly none"]}
  ]
}
Findings describe what the PR ALREADY DOES to cost.
Recommendations describe what the PR SHOULD ALSO DO to reduce cost further \
— always include current_code/recommended_code when the resource's config \
is visible in the diff, so the PR author can copy-paste the fix."""


SYSTEM_ACCOUNT = """You are a senior AWS FinOps engineer auditing an AWS \
account for cost-reduction opportunities. You have read-only tools to \
inspect resources, metrics, events, and Cost Explorer. Ground every \
recommendation in real data — call the tools rather than guessing.

Return ONLY a JSON object:
{
  "verdict": "one-line high-level summary",
  "detail": "one paragraph on the state of the account",
  "recommendations": [
    {"resource": "e.g. NAT gateway nat-0abc / EBS vol-1def / RDS mydb",
     "action": "remove" | "resize" | "reduce" | "review",
     "est_daily_delta_usd": <savings if implemented, negative>,
     "rationale": "why — reference tool output",
     "confidence": "low" | "medium" | "high"}
  ]
}"""


def _extract_json(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if "```" in t:
            t = t.split("```", 1)[0]
    start = t.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:i+1])
                    except json.JSONDecodeError:
                        break
        start = t.find("{", start + 1)
    raise json.JSONDecodeError("no JSON object found", text, 0)


def _parse_verdict(text: str, model_id: str, tool_calls: int) -> AgentVerdict:
    try:
        parsed = _extract_json(text)
    except json.JSONDecodeError as e:
        return AgentVerdict(model_id=model_id, tool_calls=tool_calls,
                            error=f"no JSON in agent response: {e}",
                            raw_text=text[:500])
    findings = [Finding(
        resource=str(f.get("resource", "")),
        action=str(f.get("action", "review")),
        est_daily_delta_usd=float(f.get("est_daily_delta_usd", 0.0) or 0.0),
        rationale=str(f.get("rationale", "")),
        confidence=str(f.get("confidence", "medium")),
    ) for f in (parsed.get("findings") or [])]
    recs = [Finding(
        resource=str(f.get("resource", "")),
        action=str(f.get("action", "review")),
        est_daily_delta_usd=float(f.get("est_daily_delta_usd", 0.0) or 0.0),
        rationale=str(f.get("rationale", "")),
        confidence=str(f.get("confidence", "medium")),
        current_code=(f.get("current_code") or None),
        recommended_code=(f.get("recommended_code") or None),
        pros=[str(p) for p in (f.get("pros") or [])],
        cons=[str(c) for c in (f.get("cons") or [])],
    ) for f in (parsed.get("recommendations") or [])]
    _low = parsed.get("est_daily_delta_low_usd")
    _high = parsed.get("est_daily_delta_high_usd")
    return AgentVerdict(
        verdict=str(parsed.get("verdict", "")),
        detail=str(parsed.get("detail", "")),
        est_daily_delta_usd=float(parsed.get("est_daily_delta_usd", 0.0) or 0.0),
        est_daily_delta_low_usd=(float(_low) if _low is not None else None),
        est_daily_delta_high_usd=(float(_high) if _high is not None else None),
        direction=str(parsed.get("direction", "unknown")),
        confidence=str(parsed.get("confidence", "medium")),
        measured=bool(parsed.get("measured", True)),
        estimation_basis=str(parsed.get("estimation_basis", "measured")),
        findings=findings,
        recommendations=recs,
        tool_calls=tool_calls,
        model_id=model_id,
        raw_text=text,
    )


def _run_agent(
    system: str,
    user_msg: str,
    profile: str | None,
    model_id: str = DEFAULT_MODEL,
) -> AgentVerdict:
    client = _client(profile)
    messages: list = [{"role": "user", "content": user_msg}]
    tool_calls = 0

    for _ in range(MAX_TOOL_TURNS + 1):
        try:
            resp = client.invoke_model(
                modelId=model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.0,
                    "system": system,
                    "tools": tool_specs(),
                    "messages": messages,
                }),
            )
            payload = json.loads(resp["body"].read())
        except Exception as e:  # noqa: BLE001
            return AgentVerdict(model_id=model_id, tool_calls=tool_calls,
                                error=f"bedrock invoke failed: {e}")

        content = payload.get("content", [])
        tool_use_blocks = [b for b in content if b.get("type") == "tool_use"]

        if tool_use_blocks:
            messages.append({"role": "assistant", "content": content})
            results = []
            for blk in tool_use_blocks:
                tool_calls += 1
                result = call_tool(blk["name"], blk.get("input") or {}, profile)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": blk["id"],
                    "content": json.dumps(result, default=str)[:8000],
                })
            messages.append({"role": "user", "content": results})
            continue

        # No more tool calls — the model tried to finalize.
        text_block = next((b for b in content if b.get("type") == "text"), None)
        if not text_block:
            return AgentVerdict(model_id=model_id, tool_calls=tool_calls,
                                error=f"no text in final response",
                                raw_text=json.dumps(content)[:500])
        text = text_block.get("text", "")

        # Sneak-peek the verdict so we can enforce a stricter bar for
        # "neutral" — the model's favorite shortcut.
        preview = _parse_verdict(text, model_id, tool_calls)
        is_neutral_or_zero = (
            preview.error is None and (
                preview.direction == "neutral"
                or abs(preview.est_daily_delta_usd) < 0.005
            )
        )
        # If the model already flagged this as an unmeasured estimate (e.g.
        # scope-expansion where the target account has no visibility), don't
        # keep pushing it to make more CloudWatch calls that will only return
        # empty — accept the low-confidence range.
        is_generic_estimate = (
            preview.error is None
            and preview.measured is False
            and preview.estimation_basis in ("generic_rate", "sibling_account")
        )
        if is_generic_estimate:
            floor = MIN_TOOL_CALLS_FOR_VERDICT
        else:
            floor = (MIN_TOOL_CALLS_FOR_NEUTRAL_VERDICT if is_neutral_or_zero
                     else MIN_TOOL_CALLS_FOR_VERDICT)

        if tool_calls < floor:
            messages.append({"role": "assistant", "content": content})
            reminder = (
                f"You have only made {tool_calls} tool call(s) so far. That "
                f"is below the required floor of {floor} for this kind of "
                "verdict. Before finalizing:\n"
                "  1. Call cost_by_service to see what actually costs money "
                "on this account today.\n"
                "  2. For every AWS resource named in the diff, call "
                "get_cloudwatch_metric — Invocations AND Duration for "
                "Lambda; IncomingBytes AND IncomingLogEvents for CloudWatch "
                "log groups; CPU/Connections for RDS; etc.\n"
                "  3. For LOGGING-related changes (logger.info→debug, log "
                "retention, batching of PutMetricData): the relevant metric "
                "is CloudWatch Logs IncomingBytes on the affected log groups "
                "(namespace=AWS/Logs, dimension=LogGroupName). Log ingestion "
                "is $0.50/GB — even 100 demoted log lines × millions of "
                "Lambda invocations is material.\n"
                "  4. Do NOT say 'no metrics available' without trying at "
                "least 3 name variations (with/without env prefix, "
                "with/without region suffix, dev-vs-prod).\n"
                "  5. Do NOT return the SAME verdict with the SAME rationale. "
                "Cite the NEW tool output that changed your mind, or update "
                "the number based on it."
            )
            if is_neutral_or_zero:
                reminder += (
                    "\n\nSPECIAL: You currently plan to return 'neutral' / $0. "
                    "That is the highest-bar verdict — it means you have "
                    "affirmative evidence that all the diff's changes wash "
                    "out. Prove it, don't assume it. Show the invocations/day "
                    "× bytes/invocation math for the demoted logs, or explain "
                    "why the log group in question has zero volume."
                )
            messages.append({"role": "user", "content": reminder})
            continue

        return preview

    return AgentVerdict(model_id=model_id, tool_calls=tool_calls,
                        error=f"exceeded {MAX_TOOL_TURNS} tool turns")


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------

def analyze_pr(
    pr_url: str,
    profile: str | None = None,
    model_id: str = DEFAULT_MODEL,
) -> AgentVerdict:
    """Fetch the PR and predict cost impact.

    Pipeline:
      1. Static extraction — regex-scan the diff to find every AWS-relevant
         change (Lambda memory bumps, log-level demotions, etc.) BEFORE the
         LLM sees anything.
      2. Pre-fetch AWS context — call cost_by_service so the LLM already
         has a picture of where money is spent.
      3. Hand LLM the diff + the pre-computed resource list + the AWS
         context. It still has all tools but starts with a floor of
         grounded facts it can't hand-wave away.
    """
    from src.ai_agent.diff_resources import (
        extract_resources,
        resources_to_prompt_hint,
    )
    try:
        repo, num, diff, title = fetch_pr_diff(pr_url)
    except Exception as e:  # noqa: BLE001
        return AgentVerdict(model_id=model_id,
                            error=f"couldn't fetch PR: {e}")

    # Stash the (repo, diff) so the `precedent_lookup` tool the LLM can
    # invoke has access to them. Cleared after the run.
    from src.ai_agent.aws_tools import set_precedent_context
    set_precedent_context(repo, diff)

    # Step 1: static extraction on the FULL diff (before any truncation)
    static_resources = extract_resources(diff)
    resource_hint = resources_to_prompt_hint(static_resources)

    # Step 2: pre-fetch cost_by_service so the LLM starts with account context
    from src.ai_agent.aws_tools import call_tool as _call_base_tool
    try:
        cost_context = _call_base_tool(
            "cost_by_service", {"days": 14}, profile,
        )
        top_services = cost_context.get("top_services_by_total", [])[:5]
        cost_hint = (
            "ACCOUNT CONTEXT (cost_by_service, last 14 days):\n"
            + "\n".join(f"  ${s['total_usd']:.2f}  {s['service']}"
                        for s in top_services)
        ) if top_services else ""
    except Exception:  # noqa: BLE001
        cost_hint = ""

    # Truncate the diff — huge diffs waste tokens on unrelated file changes.
    if len(diff) > 40_000:
        diff = diff[:20_000] + f"\n\n[...{len(diff) - 40_000} chars truncated...]\n\n" + diff[-20_000:]

    user_msg = (
        f"PR: https://github.com/{repo}/pull/{num}\n"
        f"Title: {title}\n\n"
    )
    if resource_hint:
        user_msg += resource_hint + "\n\n"
    if cost_hint:
        user_msg += cost_hint + "\n\n"
    user_msg += (
        f"Diff:\n```diff\n{diff}\n```\n\n"
        "Predict the cost impact of THIS PR, then suggest changes to reduce "
        "cost further. Call AWS tools to ground your numbers in real "
        "usage. If the PR is a scope-expansion (adding entries to a "
        "whitelist/allowlist/config collection), call `precedent_lookup` "
        "to get a measured $/entry/day rate; do NOT guess.\n"
        "Return only the JSON verdict."
    )
    try:
        return _run_agent(SYSTEM_PR, user_msg, profile, model_id)
    finally:
        set_precedent_context(None, None)


SYSTEM_NARRATIVE = """You are an AWS FinOps analyst writing a short, plain-\
English summary for a non-technical stakeholder.

You will be given:
  - The account's current average daily AWS spend (from Cost Explorer,
    last 7 days).
  - The projected new daily spend after a pull request merges.
  - A verdict object describing what the PR changes and by how much per day.

Write ONE short paragraph (3-5 sentences, no jargon) that:
  1. States the CURRENT daily cost in dollars.
  2. States what the PROJECTED daily cost will be after the PR merges, and
     whether that is an increase or decrease.
  3. Explains the single biggest driver of the change in plain terms.
  4. Notes the monthly implication (delta x 30) so the reader has a
     tangible number.

Do NOT use markdown headings, bullet points, or code blocks. Just the
paragraph. No preamble like "Here is the summary". Start directly with the
first sentence."""


def narrate_pr_impact(
    current_daily_usd: float,
    verdict: AgentVerdict,
    profile: str | None = None,
    model_id: str = DEFAULT_MODEL,
) -> str:
    """Ask Bedrock for a plain-English paragraph describing what the account
    currently spends, what it will spend after this PR, and why."""
    projected = current_daily_usd + verdict.est_daily_delta_usd
    payload = {
        "current_daily_usd": round(current_daily_usd, 2),
        "projected_daily_usd": round(projected, 2),
        "delta_daily_usd": round(verdict.est_daily_delta_usd, 2),
        "direction": verdict.direction,
        "verdict": verdict.verdict,
        "detail": verdict.detail,
        "findings": [
            {
                "resource": f.resource,
                "action": f.action,
                "delta_daily_usd": f.est_daily_delta_usd,
                "rationale": f.rationale,
            }
            for f in verdict.findings
        ],
    }
    user_msg = (
        "Here is the cost context for this pull request:\n\n"
        + json.dumps(payload, indent=2)
        + "\n\nWrite the paragraph now."
    )
    client = _client(profile)
    try:
        resp = client.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 500,
                "temperature": 0.2,
                "system": SYSTEM_NARRATIVE,
                "messages": [{"role": "user", "content": user_msg}],
            }),
        )
        body = json.loads(resp["body"].read())
    except Exception as e:  # noqa: BLE001
        return f"(narrative unavailable: {e})"
    for block in body.get("content", []):
        if block.get("type") == "text":
            return block.get("text", "").strip()
    return ""


SYSTEM_TRAJECTORY = """You are a senior AWS FinOps engineer estimating the \
future cost trajectory of an AWS account.

You are given:
  1. The account's current average daily spend (last 7 days from Cost Explorer).
  2. The estimated daily delta from ONE specific PR (already analyzed).
  3. A sweep of every repository that deploys into this AWS account:
     - open pull requests (leading indicator of what will merge soon)
     - infrastructure-as-code files touched in the last 30 days
     - files declaring scheduled/recurring compute (EventBridge, cron, etc.)

You have read-only tools (cost_by_service, cloudwatch_metric, cloudtrail, \
rightsizing) to ground your estimate in real data.

Your job: estimate what this account will run at daily, ~30 days from now, \
assuming pending PRs land at typical merge rates and current usage patterns \
continue. Do NOT reprice the ONE PR that was already analyzed — that delta \
is already accounted for. Focus on the REMAINING pending signal.

Return ONLY a JSON object (no prose outside JSON):
{
  "pending_delta_daily_usd": <signed number — additional $/day from OTHER
                              pending changes beyond the analyzed PR>,
  "projected_daily_usd": <current_daily + analyzed_pr_delta +
                          pending_delta — the number you predict the
                          account will run at in ~30 days>,
  "confidence": "low" | "medium" | "high",
  "drivers": [
    "short bullet — one concrete driver of the trajectory, referencing a
     specific PR title / IaC file / scheduled rule from the sweep",
    "..."
  ],
  "summary": "2-3 SHORT sentences in plain English. State the projected
              daily cost, the direction vs today, and the single biggest
              driver beyond the analyzed PR. No jargon, no markdown."
}"""


@dataclass
class TrajectoryEstimate:
    projected_daily_usd: float = 0.0
    pending_delta_daily_usd: float = 0.0
    confidence: str = "low"
    drivers: list[str] = field(default_factory=list)
    summary: str = ""
    tool_calls: int = 0
    model_id: str = ""
    error: str | None = None
    raw_text: str = ""


def _parse_trajectory(
    text: str, model_id: str, tool_calls: int,
) -> TrajectoryEstimate:
    try:
        parsed = _extract_json(text)
    except json.JSONDecodeError as e:
        return TrajectoryEstimate(
            model_id=model_id, tool_calls=tool_calls,
            error=f"no JSON in response: {e}", raw_text=text[:500],
        )
    return TrajectoryEstimate(
        projected_daily_usd=float(parsed.get("projected_daily_usd", 0.0) or 0.0),
        pending_delta_daily_usd=float(
            parsed.get("pending_delta_daily_usd", 0.0) or 0.0,
        ),
        confidence=str(parsed.get("confidence", "low")),
        drivers=[str(d) for d in (parsed.get("drivers") or [])],
        summary=str(parsed.get("summary", "")),
        tool_calls=tool_calls,
        model_id=model_id,
        raw_text=text,
    )


def estimate_repo_trajectory(
    repos: list[str],
    verdict: AgentVerdict,
    current_daily_usd: float,
    profile: str | None = None,
    model_id: str = DEFAULT_MODEL,
    max_tool_turns: int = 6,
) -> TrajectoryEstimate:
    """Sweep the repos that deploy into this AWS account and ask the model
    to project the account's daily cost ~30 days out.

    The single PR that was already analyzed is passed in so the model
    doesn't double-count it — it only prices the REMAINING pending
    signal (open PRs, recent IaC churn, scheduled compute)."""
    from src.ai_agent.repo_sweep import sweep_repos, sweep_to_summary

    if not repos:
        return TrajectoryEstimate(
            projected_daily_usd=current_daily_usd + verdict.est_daily_delta_usd,
            pending_delta_daily_usd=0.0,
            confidence="low",
            drivers=[],
            summary=(
                "No matched repositories for this AWS profile, so no "
                "repo-wide trajectory could be computed. Projection "
                "reflects only the analyzed PR."
            ),
            model_id=model_id,
        )

    try:
        sweep = sweep_to_summary(sweep_repos(repos))
    except Exception as e:  # noqa: BLE001
        return TrajectoryEstimate(
            model_id=model_id, error=f"repo sweep failed: {e}",
        )

    payload = {
        "current_daily_usd": round(current_daily_usd, 2),
        "analyzed_pr": {
            "verdict": verdict.verdict,
            "direction": verdict.direction,
            "est_daily_delta_usd": round(verdict.est_daily_delta_usd, 2),
            "detail": verdict.detail,
        },
        "repo_sweep": sweep,
    }
    user_msg = (
        "Repo + account context:\n\n"
        + json.dumps(payload, indent=2, default=str)
        + "\n\nProject the account's daily cost ~30 days from now. "
        "Use tools to sanity-check big claims. Return the JSON now."
    )

    client = _client(profile)
    messages: list = [{"role": "user", "content": user_msg}]
    tool_calls = 0

    for _ in range(max_tool_turns + 1):
        try:
            resp = client.invoke_model(
                modelId=model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.0,
                    "system": SYSTEM_TRAJECTORY,
                    "tools": tool_specs(),
                    "messages": messages,
                }),
            )
            body = json.loads(resp["body"].read())
        except Exception as e:  # noqa: BLE001
            return TrajectoryEstimate(
                model_id=model_id, tool_calls=tool_calls,
                error=f"bedrock invoke failed: {e}",
            )

        content = body.get("content", [])
        tool_use_blocks = [b for b in content if b.get("type") == "tool_use"]
        if tool_use_blocks:
            messages.append({"role": "assistant", "content": content})
            results = []
            for blk in tool_use_blocks:
                tool_calls += 1
                result = call_tool(blk["name"], blk.get("input") or {}, profile)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": blk["id"],
                    "content": json.dumps(result, default=str)[:8000],
                })
            messages.append({"role": "user", "content": results})
            continue

        text_block = next((b for b in content if b.get("type") == "text"), None)
        if not text_block:
            return TrajectoryEstimate(
                model_id=model_id, tool_calls=tool_calls,
                error="no text in final response",
                raw_text=json.dumps(content)[:500],
            )
        return _parse_trajectory(
            text_block.get("text", ""), model_id, tool_calls,
        )

    return TrajectoryEstimate(
        model_id=model_id, tool_calls=tool_calls,
        error=f"exceeded {max_tool_turns} tool turns",
    )


def recommend_account(
    focus: str = "",
    profile: str | None = None,
    model_id: str = DEFAULT_MODEL,
) -> AgentVerdict:
    """Free-form audit of the current account for cost-reduction ideas."""
    focus_line = f"Focus: {focus}\n\n" if focus else ""
    user_msg = (
        f"{focus_line}"
        "Audit this AWS account for cost-reduction opportunities. Use the "
        "read-only tools to look at what's actually running, what's changed "
        "recently (CloudTrail), and where cost is going (Cost Explorer). "
        "Then propose specific, actionable recommendations with estimated "
        "daily savings. Return only the JSON verdict."
    )
    return _run_agent(SYSTEM_ACCOUNT, user_msg, profile, model_id)
