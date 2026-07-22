"""PR cost-check orchestration for GitHub Actions."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from src.ai_agent.agent import AgentVerdict, analyze_pr
from src.ci.forecast_chart import save_forecast_chart
from src.ci.forecast_context import ForecastContext, build_forecast_context


@dataclass
class PolicyConfig:
    max_daily_increase_usd: float = 5.0
    min_tool_calls: int = 5


@dataclass
class PrCostCheckResult:
    verdict: AgentVerdict
    failures: list[str] = field(default_factory=list)
    forecast: ForecastContext | None = None
    chart_path: Path | None = None
    chart_warning: str | None = None

    @property
    def passed(self) -> bool:
        return not self.failures


def run_policy_checks(
    verdict: AgentVerdict,
    policy: PolicyConfig,
) -> list[str]:
    failures: list[str] = []
    if verdict.error:
        failures.append(verdict.error)
    if verdict.tool_calls < policy.min_tool_calls:
        failures.append(
            f"only {verdict.tool_calls} AWS tool call(s) "
            f"(minimum {policy.min_tool_calls})"
        )
    if (
        verdict.direction == "increase"
        and verdict.est_daily_delta_usd > policy.max_daily_increase_usd
    ):
        failures.append(
            f"estimated +${verdict.est_daily_delta_usd:.2f}/day exceeds "
            f"threshold ${policy.max_daily_increase_usd:.2f}/day"
        )
    return failures


def _direction_emoji(direction: str) -> str:
    return {"increase": "↗", "decrease": "↘", "neutral": "→"}.get(direction, "?")


def _verdict_to_dict(verdict: AgentVerdict) -> dict:
    return asdict(verdict)


def write_verdict_json(verdict: AgentVerdict, path: str | Path) -> Path:
    out = Path(path)
    out.write_text(json.dumps(_verdict_to_dict(verdict), indent=2))
    return out


def write_step_summary(
    result: PrCostCheckResult,
    path: str | Path,
    *,
    pr_url: str,
    chart_filename: str | None = None,
) -> Path:
    """Write markdown for GITHUB_STEP_SUMMARY (supports relative image paths)."""
    verdict = result.verdict
    emoji = _direction_emoji(verdict.direction)
    status = "PASSED" if result.passed else "FAILED"

    lines = [
        "## CostSense PR Cost Check",
        "",
        f"**Status:** {status}",
        f"**PR:** {pr_url}",
        "",
        f"### {emoji} {verdict.verdict or 'No verdict'}",
        "",
        f"| Metric | Value |",
        f"|---|---:|",
        f"| Direction | {verdict.direction} |",
        f"| Est. daily impact | ${verdict.est_daily_delta_usd:+,.2f} |",
        f"| Est. monthly impact | ${verdict.est_daily_delta_usd * 30:+,.0f} |",
        f"| AWS tool calls | {verdict.tool_calls} |",
        f"| Model | {verdict.model_id or '—'} |",
        "",
    ]

    if verdict.detail:
        lines.extend([f"**In plain terms:** {verdict.detail}", ""])

    if verdict.findings:
        lines.extend(["### What this PR does to cost", ""])
        lines.append("| Resource | Action | $/day Δ | Rationale |")
        lines.append("|---|---|---:|---|")
        for f in verdict.findings:
            rationale = (f.rationale or "").replace("|", "\\|")
            lines.append(
                f"| {f.resource} | {f.action} | "
                f"${f.est_daily_delta_usd:+,.2f} | {rationale} |"
            )
        lines.append("")

    if verdict.recommendations:
        lines.extend(["### Recommendations", ""])
        for i, r in enumerate(verdict.recommendations, start=1):
            lines.append(
                f"{i}. **{r.resource}** — _{r.action}_ "
                f"(${r.est_daily_delta_usd:+,.2f}/day): {r.rationale}"
            )
        lines.append("")

    if result.forecast is not None:
        ctx = result.forecast
        next_7 = (
            float(ctx.fc_df["adjusted_usd"].sum())
            if not ctx.fc_df.empty else None
        )
        lines.extend([
            "### Forecast context",
            "",
            f"- Account: `{ctx.account_id}`",
            f"- Model: `{ctx.model}`",
            f"- Cutoff: `{ctx.cutoff.isoformat()}`",
        ])
        if next_7 is not None:
            lines.append(f"- Next 7d adjusted forecast total: **${next_7:,.0f}**")
        lines.append("")

    if chart_filename and result.chart_path is not None:
        lines.extend([
            "### Forecast chart",
            "",
            f"![Forecast with PR impact]({chart_filename})",
            "",
        ])

    if result.chart_warning:
        lines.extend([
            "### Forecast chart (skipped)",
            "",
            f"_{result.chart_warning}_",
            "",
        ])

    if result.failures:
        lines.extend(["### Policy failures", ""])
        for failure in result.failures:
            lines.append(f"- {failure}")
        lines.append("")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out


def run_pr_cost_check(
    pr_url: str,
    *,
    profile: str | None = None,
    model_id: str | None = None,
    policy: PolicyConfig | None = None,
    history_days: int = 60,
    forecast_chart_path: str | Path | None = None,
    cutoff: date | None = None,
) -> PrCostCheckResult:
    """Analyze a PR, run policy checks, and optionally render a forecast chart."""
    policy = policy or PolicyConfig()
    from src.ai_agent.agent import DEFAULT_MODEL

    verdict = analyze_pr(
        pr_url,
        profile=profile,
        model_id=model_id or DEFAULT_MODEL,
    )
    failures = run_policy_checks(verdict, policy)

    forecast: ForecastContext | None = None
    chart_path: Path | None = None
    chart_warning: str | None = None
    if forecast_chart_path is not None:
        try:
            forecast = build_forecast_context(
                profile=profile,
                pr_daily_delta_usd=verdict.est_daily_delta_usd,
                history_days=history_days,
                cutoff=cutoff,
            )
            chart_path = save_forecast_chart(forecast, forecast_chart_path)
        except Exception as exc:  # noqa: BLE001
            chart_warning = f"forecast chart failed: {exc}"

    return PrCostCheckResult(
        verdict=verdict,
        failures=failures,
        forecast=forecast,
        chart_path=chart_path,
        chart_warning=chart_warning,
    )
