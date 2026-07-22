#!/usr/bin/env python3
"""Headless PR cost check for GitHub Actions.

Usage:
  python scripts/pr_cost_check.py \\
    --pr-url https://github.com/org/repo/pull/123 \\
    --profile dil-data-platform-dev \\
    --forecast-chart forecast.png

Writes pr-cost-verdict.json and appends to GITHUB_STEP_SUMMARY when set.
Exits 0 on pass, 1 on policy failure.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ai_agent.agent import DEFAULT_MODEL
from src.ci.pr_check import (
    PolicyConfig,
    run_pr_cost_check,
    write_step_summary,
    write_verdict_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="CostSense PR cost check")
    parser.add_argument("--pr-url", required=True)
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument(
        "--max-daily-increase-usd", type=float, default=5.0,
    )
    parser.add_argument("--min-tool-calls", type=int, default=5)
    parser.add_argument("--history-days", type=int, default=60)
    parser.add_argument(
        "--forecast-chart",
        help="PNG path for the forecast chart (embedded in job summary)",
    )
    parser.add_argument(
        "--output-json", default="pr-cost-verdict.json",
        help="Where to write the structured verdict",
    )
    parser.add_argument(
        "--step-summary",
        default=os.environ.get("GITHUB_STEP_SUMMARY"),
        help="Markdown output path (defaults to GITHUB_STEP_SUMMARY)",
    )
    args = parser.parse_args()

    result = run_pr_cost_check(
        args.pr_url,
        profile=args.profile,
        model_id=args.model_id,
        policy=PolicyConfig(
            max_daily_increase_usd=args.max_daily_increase_usd,
            min_tool_calls=args.min_tool_calls,
        ),
        history_days=args.history_days,
        forecast_chart_path=args.forecast_chart,
    )

    write_verdict_json(result.verdict, args.output_json)

    chart_name: str | None = None
    if args.forecast_chart and result.chart_path is not None:
        chart_path = result.chart_path.resolve()
        workspace = os.environ.get("GITHUB_WORKSPACE")
        if workspace:
            try:
                chart_name = str(chart_path.relative_to(Path(workspace)))
            except ValueError:
                chart_name = chart_path.name
        else:
            chart_name = chart_path.name

    if args.step_summary:
        write_step_summary(
            result,
            args.step_summary,
            pr_url=args.pr_url,
            chart_filename=chart_name,
        )

    if result.passed:
        print(result.verdict.verdict or "PR cost check passed")
        return 0

    print("PR cost check FAILED:", file=sys.stderr)
    for failure in result.failures:
        print(f"  - {failure}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
