"""Bridge between the raw GitHub scanner and the forecast pipeline.

Supports two analyzers:
  - `regex` (default): fast, deterministic, catches add/remove.
  - `llm` (Bedrock Claude): reads the actual diff, catches config tweaks
    (instance-type changes on the same resource name, capacity changes, etc.).
  - `hybrid`: run BOTH, prefer LLM verdict when it disagrees on cost impact.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Literal

from src.forecast.timeseries import PrStep
from src.pr_scanner.github_scan import PrRecord, pr_diff, scan_repos
from src.pr_scanner.llm_analyzer import LlmVerdict, analyze_pr_diff
from src.pr_scanner.pricing import PriceEstimate, estimate_daily_usd


Analyzer = Literal["regex", "llm", "hybrid"]


@dataclass
class PricedChange:
    repo: str
    pr_number: int
    pr_title: str
    pr_url: str
    resource_type: str
    resource_name: str
    action: str
    instance_hint: str | None
    est_daily_delta_usd: float
    price_source: str        # pricing-api | table | unknown | llm
    rationale: str = ""      # populated when source == llm


@dataclass
class PrImpact:
    repo: str
    pr_number: int
    pr_title: str
    pr_url: str
    author: str
    merged_at: str
    est_daily_delta_usd: float
    analyzer: str            # "regex" | "llm" | "hybrid"
    llm_summary: str = ""
    llm_model: str = ""
    llm_error: str | None = None
    changes: list[PricedChange] = field(default_factory=list)


def _apply_sign(action: str, price_daily: float) -> float:
    if action == "add":
        return price_daily
    if action == "remove":
        return -price_daily
    return 0.0


def _price_regex_changes(pr: PrRecord, aws_profile: str | None) -> tuple[list[PricedChange], float]:
    priced: list[PricedChange] = []
    total = 0.0
    for c in pr.changes:
        est: PriceEstimate = estimate_daily_usd(
            c.resource_type, instance_hint=c.instance_hint, profile=aws_profile,
        )
        delta = _apply_sign(c.action, est.daily_usd)
        priced.append(PricedChange(
            repo=pr.repo, pr_number=pr.number, pr_title=pr.title, pr_url=pr.url,
            resource_type=c.resource_type, resource_name=c.resource_name,
            action=c.action, instance_hint=c.instance_hint,
            est_daily_delta_usd=round(delta, 4), price_source=est.source,
        ))
        total += delta
    return priced, round(total, 4)


def _price_llm_changes(pr: PrRecord, verdict: LlmVerdict) -> tuple[list[PricedChange], float]:
    priced: list[PricedChange] = []
    for c in verdict.changes:
        priced.append(PricedChange(
            repo=pr.repo, pr_number=pr.number, pr_title=pr.title, pr_url=pr.url,
            resource_type=c.resource_type, resource_name=c.resource_name,
            action=c.action, instance_hint=c.instance_hint,
            est_daily_delta_usd=round(c.est_daily_delta_usd, 4),
            price_source="llm",
            rationale=c.rationale,
        ))
    return priced, verdict.total_daily_delta_usd


def scan_and_price(
    repos: Iterable[str],
    base: str,
    lookback_days: int = 14,
    aws_profile: str | None = None,
    analyzer: Analyzer = "hybrid",
    llm_model: str | None = None,
) -> tuple[list[PrImpact], float]:
    # LLM/hybrid modes need every merged PR (config-only tweaks slip past the
    # regex filter); regex mode can drop non-IaC PRs eagerly to save time.
    keep_empty = analyzer in ("llm", "hybrid")
    prs = scan_repos(repos, base=base, lookback_days=lookback_days,
                     keep_empty=keep_empty)

    # Pre-fetch LLM verdicts in parallel — Bedrock invoke is I/O bound.
    llm_verdicts: dict[tuple[str, int], LlmVerdict] = {}
    if analyzer in ("llm", "hybrid"):
        def _llm_one(pr):
            try:
                diff = pr_diff(pr.repo, pr.number)
                return analyze_pr_diff(
                    diff, pr_title=pr.title, profile=aws_profile,
                    model_id=llm_model or "anthropic.claude-3-haiku-20240307-v1:0",
                )
            except Exception as e:  # noqa: BLE001
                return LlmVerdict(error=str(e))
        with ThreadPoolExecutor(max_workers=6) as pool:
            for pr, verdict in zip(prs, pool.map(_llm_one, prs)):
                llm_verdicts[(pr.repo, pr.number)] = verdict

    impacts: list[PrImpact] = []
    for pr in prs:
        regex_priced: list[PricedChange] = []
        regex_delta = 0.0
        if analyzer in ("regex", "hybrid"):
            regex_priced, regex_delta = _price_regex_changes(pr, aws_profile)

        llm_verdict: LlmVerdict | None = llm_verdicts.get((pr.repo, pr.number))
        llm_priced: list[PricedChange] = []
        llm_delta = 0.0
        if llm_verdict is not None:
            llm_priced, llm_delta = _price_llm_changes(pr, llm_verdict)

        # Choose which set of changes represents the PR.
        # In hybrid mode: prefer LLM when it succeeded AND either party sees
        # cost impact. Regex is the fallback for zero-LLM or LLM-error cases.
        chosen: list[PricedChange]
        chosen_delta: float
        if analyzer == "llm":
            chosen, chosen_delta = llm_priced, llm_delta
        elif analyzer == "regex":
            chosen, chosen_delta = regex_priced, regex_delta
        else:  # hybrid
            if llm_verdict and llm_verdict.error is None:
                chosen, chosen_delta = llm_priced, llm_delta
                # if LLM said "no impact" but regex found something, keep regex
                if not llm_priced and regex_priced:
                    chosen, chosen_delta = regex_priced, regex_delta
            else:
                chosen, chosen_delta = regex_priced, regex_delta

        if not chosen and not (llm_verdict and llm_verdict.summary):
            continue  # nothing to say about this PR

        impacts.append(PrImpact(
            repo=pr.repo,
            pr_number=pr.number,
            pr_title=pr.title,
            pr_url=pr.url,
            author=pr.author,
            merged_at=pr.merged_at,
            est_daily_delta_usd=round(chosen_delta, 4),
            analyzer=analyzer,
            llm_summary=(llm_verdict.summary if llm_verdict else ""),
            llm_model=(llm_verdict.model_id if llm_verdict else ""),
            llm_error=(llm_verdict.error if llm_verdict else None),
            changes=chosen,
        ))

    _drop_hallucinated_duplicates(impacts)
    total = round(sum(i.est_daily_delta_usd for i in impacts), 4)
    return impacts, total


def _drop_hallucinated_duplicates(impacts: list[PrImpact]) -> None:
    """LLMs sometimes repeat the same $ estimate across many PRs when they
    can't really analyze the diff. If ≥3 PRs share the same nonzero
    est_daily_delta_usd AND the same LLM summary, zero them out — that's a
    clear hallucination pattern, not real signal.

    Mutates `impacts` in place.
    """
    from collections import defaultdict
    groups: dict[tuple[float, str], list[PrImpact]] = defaultdict(list)
    for imp in impacts:
        if not imp.est_daily_delta_usd:
            continue
        key = (round(imp.est_daily_delta_usd, 4),
               (imp.llm_summary or "").strip().lower()[:80])
        groups[key].append(imp)
    for key, group in groups.items():
        if len(group) >= 3:
            for imp in group:
                imp.est_daily_delta_usd = 0.0
                imp.llm_error = (
                    "dropped: LLM returned the same $ estimate + summary "
                    f"across {len(group)} PRs (hallucination)"
                )
                for c in imp.changes:
                    c.est_daily_delta_usd = 0.0


def impacts_to_steps(impacts: list[PrImpact]) -> list[PrStep]:
    steps: list[PrStep] = []
    for imp in impacts:
        if not imp.est_daily_delta_usd:
            continue
        try:
            merge_day = date.fromisoformat(imp.merged_at[:10])
        except ValueError:
            continue
        steps.append(PrStep(
            from_day=merge_day,
            delta_usd=imp.est_daily_delta_usd,
            pr_id=f"{imp.repo}#{imp.pr_number}",
        ))
    return steps


def impacts_to_dict(impacts: list[PrImpact]) -> list[dict]:
    return [
        {
            "repo": i.repo,
            "pr_number": i.pr_number,
            "pr_title": i.pr_title,
            "pr_url": i.pr_url,
            "author": i.author,
            "merged_at": i.merged_at,
            "est_daily_delta_usd": i.est_daily_delta_usd,
            "analyzer": i.analyzer,
            "llm_summary": i.llm_summary,
            "llm_model": i.llm_model,
            "llm_error": i.llm_error,
            "changes": [
                {
                    "resource_type": c.resource_type,
                    "resource_name": c.resource_name,
                    "action": c.action,
                    "instance_hint": c.instance_hint,
                    "est_daily_delta_usd": c.est_daily_delta_usd,
                    "price_source": c.price_source,
                    "rationale": c.rationale,
                }
                for c in i.changes
            ],
        }
        for i in impacts
    ]
