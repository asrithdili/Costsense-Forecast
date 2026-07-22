"""Extract AWS resource names + change types from a raw PR diff.

This runs BEFORE the LLM sees the diff so we can:
  1. Pre-fetch CloudWatch / Cost Explorer context for each resource
  2. Hand the LLM a structured "resources you MUST price" list
  3. Detect kinds of changes that have well-known pricing formulas
     (Lambda memory, log-level demotion, PC changes) — so the LLM can't
     shrug them off as "no cost impact".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class DiffResource:
    kind: str                          # lambda | rds | ecs | dynamodb | s3 | log_group | cloudwatch | ...
    name: str                          # best-guess resource identifier
    change_type: str                   # add | remove | modify | log_level | memory | pc | retention | metric_batch
    evidence: str = ""                 # the diff line(s) that hinted at it
    quantitative_hint: dict = field(default_factory=dict)  # e.g. {"old_memory": 10240, "new_memory": 4096}


# --- Terraform / CDK resource declarations ---------------------------------

_TF_RESOURCE_RE = re.compile(
    r'^([+-])\s*resource\s+"(aws_[a-zA-Z0-9_]+)"\s+"([a-zA-Z0-9_\-]+)"'
)
_CDK_NEW_RE = re.compile(
    r'^([+-])\s*.*\bnew\s+(?:aws_[a-zA-Z0-9_]+\.)?([A-Z][a-zA-Z0-9]+)\s*\('
)
_CDK_CLASS_TO_KIND = {
    "Function": "lambda", "Instance": "ec2", "Bucket": "s3",
    "Table": "dynamodb", "Queue": "sqs", "Topic": "sns",
    "DatabaseInstance": "rds", "DatabaseCluster": "rds",
    "NatGateway": "nat", "LogGroup": "log_group",
    "Distribution": "cloudfront", "Cluster": "ecs",
    "FargateService": "ecs", "RestApi": "apigateway",
    "HttpApi": "apigateway",
}


# --- Config-level changes (the kind Haiku misses) --------------------------

_MEMORY_RE = re.compile(
    r'(?:memorySize|memory_?size|memorySize:\s*)\s*[:=]\s*(\d+)'
)
_PC_RE = re.compile(
    r'(?:provisionedConcurrency|provisioned_concurrency)\s*[:=]\s*(\d+)'
)
_RETENTION_RE = re.compile(
    r'(?:retentionInDays|retention_in_days|retention_?days)\s*[:=]\s*(\d+)'
)
_INSTANCE_TYPE_RE = re.compile(
    r'(?:instanceType|instance_type|instance_class)\s*[:=]\s*["\']?([a-z]\d[a-z]?\.[a-z]+)'
)
_LOG_LEVEL_DOWN_RE = re.compile(r'^-\s*.*logger\.(info|warning)\(')
_LOG_LEVEL_UP_RE = re.compile(r'^\+\s*.*logger\.(debug)\(')
_METRIC_BATCH_HINT = re.compile(
    r'(_metric_buffer|flush_metrics|batch.*put_metric|put_metric_data.*batch)',
    re.IGNORECASE,
)


def _resource_type_to_kind(rtype: str) -> str:
    lookup = {
        "aws_lambda_function": "lambda",
        "aws_instance": "ec2",
        "aws_db_instance": "rds",
        "aws_rds_instance": "rds",
        "aws_rds_cluster": "rds",
        "aws_s3_bucket": "s3",
        "aws_dynamodb_table": "dynamodb",
        "aws_sqs_queue": "sqs",
        "aws_sns_topic": "sns",
        "aws_nat_gateway": "nat",
        "aws_cloudfront_distribution": "cloudfront",
        "aws_ecs_cluster": "ecs",
        "aws_ecs_service": "ecs",
        "aws_cloudwatch_log_group": "log_group",
        "aws_apigateway_rest_api": "apigateway",
        "aws_apigatewayv2_api": "apigateway",
    }
    return lookup.get(rtype, "unknown")


def extract_resources(diff: str) -> list[DiffResource]:
    """Best-effort static extraction. Returns a list of resources touched
    plus the kind of change so downstream code can decide what tools to run."""
    out: list[DiffResource] = []
    lines = diff.splitlines()

    # 1. Terraform + CDK resource add/remove
    for i, line in enumerate(lines):
        m = _TF_RESOURCE_RE.match(line)
        if m:
            sign, rtype, rname = m.group(1), m.group(2), m.group(3)
            out.append(DiffResource(
                kind=_resource_type_to_kind(rtype),
                name=rname,
                change_type="add" if sign == "+" else "remove",
                evidence=line.strip(),
            ))
            continue
        m = _CDK_NEW_RE.match(line)
        if m:
            sign, cls = m.group(1), m.group(2)
            kind = _CDK_CLASS_TO_KIND.get(cls)
            if kind:
                out.append(DiffResource(
                    kind=kind, name=cls,
                    change_type="add" if sign == "+" else "remove",
                    evidence=line.strip(),
                ))

    # 2. Memory changes on Lambda functions
    memory_changes: dict[str, dict] = {}
    for i, line in enumerate(lines):
        if not line.startswith(("+", "-")):
            continue
        m = _MEMORY_RE.search(line)
        if not m:
            continue
        val = int(m.group(1))
        # try to identify which Lambda this is inside — look backward for a
        # function name declaration
        fn_name = ""
        for lookback in reversed(lines[max(0, i - 40):i]):
            fname_m = re.search(r'(?:functionName|FunctionName|"([a-zA-Z0-9_-]+)"\s*=>|const\s+(\w+)\s*=)', lookback)
            if fname_m:
                fn_name = fname_m.group(1) or fname_m.group(2) or ""
                if fn_name:
                    break
        entry = memory_changes.setdefault(fn_name or f"line_{i}", {})
        if line.startswith("-"):
            entry["old"] = val
        else:
            entry["new"] = val
        entry.setdefault("evidence", []).append(line.strip()[:200])
    for fn, ch in memory_changes.items():
        if "old" in ch and "new" in ch and ch["old"] != ch["new"]:
            out.append(DiffResource(
                kind="lambda", name=fn, change_type="memory",
                evidence=" ; ".join(ch["evidence"])[:400],
                quantitative_hint={"old_memory": ch["old"], "new_memory": ch["new"]},
            ))

    # 3. Provisioned concurrency changes
    pc_changes: dict[str, dict] = {}
    for line in lines:
        if not line.startswith(("+", "-")):
            continue
        m = _PC_RE.search(line)
        if not m:
            continue
        val = int(m.group(1))
        entry = pc_changes.setdefault("pc", {})
        if line.startswith("-"):
            entry.setdefault("old", val)
        else:
            entry.setdefault("new", val)
    if "old" in pc_changes.get("pc", {}) and "new" in pc_changes.get("pc", {}):
        old, new = pc_changes["pc"]["old"], pc_changes["pc"]["new"]
        if old != new:
            out.append(DiffResource(
                kind="lambda", name="provisioned_concurrency",
                change_type="pc",
                quantitative_hint={"old_pc": old, "new_pc": new},
            ))

    # 4. Log retention changes
    ret_changes: dict[str, dict] = {}
    for line in lines:
        m = _RETENTION_RE.search(line)
        if not m or not line.startswith(("+", "-")):
            continue
        val = int(m.group(1))
        entry = ret_changes.setdefault("retention", {})
        if line.startswith("-"):
            entry.setdefault("old", val)
        else:
            entry.setdefault("new", val)
    if "old" in ret_changes.get("retention", {}) and "new" in ret_changes.get("retention", {}):
        out.append(DiffResource(
            kind="log_group", name="retention_change", change_type="retention",
            quantitative_hint={
                "old_days": ret_changes["retention"]["old"],
                "new_days": ret_changes["retention"]["new"],
            },
        ))

    # 5. Instance-type changes
    it_changes: dict[str, dict] = {}
    for line in lines:
        m = _INSTANCE_TYPE_RE.search(line)
        if not m or not line.startswith(("+", "-")):
            continue
        val = m.group(1)
        entry = it_changes.setdefault("it", {})
        if line.startswith("-"):
            entry.setdefault("old", val)
        else:
            entry.setdefault("new", val)
    if "old" in it_changes.get("it", {}) and "new" in it_changes.get("it", {}):
        old, new = it_changes["it"]["old"], it_changes["it"]["new"]
        if old != new:
            out.append(DiffResource(
                kind="ec2", name="instance_type_change",
                change_type="modify",
                quantitative_hint={"old_type": old, "new_type": new},
            ))

    # 6. Log-level demotions (info → debug) — MUCH cheaper on CloudWatch Logs
    demoted = sum(1 for l in lines if _LOG_LEVEL_DOWN_RE.match(l))
    added_debug = sum(1 for l in lines if _LOG_LEVEL_UP_RE.match(l))
    net_demotions = min(demoted, added_debug)
    if net_demotions >= 5:
        out.append(DiffResource(
            kind="log_group", name="info_to_debug_demotion",
            change_type="log_level",
            evidence=f"~{net_demotions} logger.info→debug lines detected",
            quantitative_hint={"demoted_lines": net_demotions},
        ))

    # 7. Metric-batching change
    if any(_METRIC_BATCH_HINT.search(l) for l in lines if l.startswith(("+", "-"))):
        out.append(DiffResource(
            kind="cloudwatch", name="metric_emission_batching",
            change_type="metric_batch",
            evidence="metric buffer / batching keywords found in diff",
        ))

    # NOTE: We used to detect "scope expansion" statically (whitelist/tenant
    # additions) and hand the LLM a precedent-based rate. That produced
    # false positives on refactored Python code that just added multi-line
    # function arguments. Scope expansion is now decided by the LLM itself
    # from the diff + AWS context, and it can invoke the `precedent_lookup`
    # tool on demand when the diff actually looks like tenant onboarding.

    return out


# --- Scope expansion (collection growth) -----------------------------------
#
# Detects any PR that adds many similar entries to a bracketed collection —
# whether the code calls it a whitelist, allowlist, enabledCustomers,
# partnerIds, or anything else. We don't rely on a keyword list.
#
# The signal is purely structural: a `+` line whose payload is one or more
# comma/whitespace-separated value-shaped tokens, sitting inside a diff
# hunk that is dominated by such additions rather than logic changes.
_TOKEN_SPLIT_RE = re.compile(r"[,;\s]+")
_LOOKS_LIKE_VALUE_RE = re.compile(r"^[\"'`]?[\w.\-:/@]+[\"'`]?$")


def _tokens_on_added_line(line: str) -> list[str]:
    """Extract the comma/whitespace-separated value-shaped tokens on a
    single `+` line. Returns [] if the line is not a pure list of values
    (e.g. it also declares a variable, opens a block, or contains an
    operator)."""
    if not line.startswith("+") or line.startswith("+++"):
        return []
    payload = line[1:]
    payload = re.split(r"(?://|#)", payload, maxsplit=1)[0]
    payload = payload.strip(" \t[]{}()")
    if not payload:
        return []
    # Any = / => / : / ( / { in the payload means this is logic, not
    # a pure list continuation. Reject.
    if re.search(r"[=:{(]|=>", payload):
        return []
    tokens = [t.strip(" \t\"'`,;") for t in _TOKEN_SPLIT_RE.split(payload)]
    return [t for t in tokens if t and _LOOKS_LIKE_VALUE_RE.match(t)]


def _count_added_scope_ids(lines: list[str], min_run: int = 3) -> int:
    """Count NEW entries added to a bracketed collection.

    Shape-based, no keyword list: we look for RUNS of consecutive `+` lines
    (allowing whitespace-only context lines to sit between them) where each
    line is a pure list of comma-separated value tokens. A run of ≥
    `min_run` such lines is treated as a scope-expansion; the total token
    count across the run is returned. Multiple runs across the diff are
    summed."""
    n = len(lines)
    i = 0
    total = 0
    while i < n:
        # Skip anything that isn't the start of a plus run.
        toks = _tokens_on_added_line(lines[i]) if lines[i].startswith("+") else []
        if not toks:
            i += 1
            continue
        run_tokens = list(toks)
        run_lines = 1
        j = i + 1
        # Extend the run over subsequent `+` value-only lines. Allow at
        # most one non-`+` context line between (line breaks in a list).
        while j < n:
            if lines[j].startswith("+"):
                more = _tokens_on_added_line(lines[j])
                if not more:
                    break
                run_tokens.extend(more)
                run_lines += 1
                j += 1
                continue
            # A single blank/whitespace context line is OK; anything else
            # breaks the run.
            if lines[j].strip() == "":
                j += 1
                continue
            break
        if run_lines >= min_run:
            total += len(run_tokens)
        i = j
    return total


def resources_to_prompt_hint(resources: list[DiffResource]) -> str:
    """Format the extracted resource list into a prompt block Claude gets
    with its user message. Explicit list = harder to skip."""
    if not resources:
        return ""
    lines = [
        "PRE-COMPUTED DIFF ANALYSIS (do not re-derive, but VERIFY each with tools):"
    ]
    for r in resources:
        base = f"  • [{r.kind}] {r.name} — {r.change_type}"
        if r.quantitative_hint:
            base += f"  hint={r.quantitative_hint}"
        if r.evidence:
            base += f"  evidence=\"{r.evidence[:120]}\""
        lines.append(base)
    lines.append(
        "You MUST call CloudWatch (or the appropriate AWS tool) for EACH "
        "resource above to compute its dollar impact. Do not skip any."
    )
    return "\n".join(lines)
