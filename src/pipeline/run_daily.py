"""Daily orchestrator — invoked by GitHub Actions cron or on-demand from the UI."""
from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.aws.cost_explorer import fetch_daily_totals
from src.aws.profiles import resolve
from src.forecast.ensemble import forecast_auto, tuned_params_dict
from src.forecast.aws_forecast import forecast_from_ce
from src.forecast.lightgbm_model import forecast_lightgbm
from src.forecast.timeseries import (
    forecast_next_7_days,
    forecast_with_pr_regressor,
)
from src.pipeline.adjust import adjust, to_dict
from src.pipeline.paths import actuals_dir, predictions_dir
from src.pr_scanner.scan import (
    impacts_to_dict,
    impacts_to_steps,
    scan_and_price,
)


def _open_prs_to_dict_safe(priced: list) -> list[dict]:
    try:
        from src.pr_scanner.open_prs import to_dict
        return to_dict(priced)
    except Exception:  # noqa: BLE001
        return []


def _persist_actuals(account_id: str, totals: list[tuple[date, float]]) -> None:
    d = actuals_dir(account_id)
    for day, amount in totals:
        (d / f"{day.isoformat()}.json").write_text(
            json.dumps({"day": day.isoformat(), "amount_usd": float(amount)})
        )


def run(
    cutoff: date | None = None,
    profile: str | None = None,
    history_days: int = 90,
    repos: list[str] | None = None,
    base_branch: str | None = None,
    pr_lookback_days: int | None = None,
    analyzer: str = "hybrid",
    llm_model: str | None = None,
    service: str | None = None,
    model: str = "lightgbm",
    include_open_prs: bool = True,
) -> Path:
    cutoff = cutoff or date.today()
    start = cutoff - timedelta(days=history_days)
    pr_lookback_days = pr_lookback_days if pr_lookback_days is not None else history_days

    info = resolve(profile) if profile else None
    account_id = (info.account_id if info else None) or "unknown"

    totals = fetch_daily_totals(start, cutoff, profile=profile, service=service)
    _persist_actuals(account_id, totals)

    history_df = pd.DataFrame(
        [{"day": pd.Timestamp(d), "amount_usd": a} for d, a in totals]
    )

    impacts: list = []
    pr_steps: list = []
    if repos and base_branch:
        impacts, _ = scan_and_price(
            repos, base=base_branch, lookback_days=pr_lookback_days,
            aws_profile=profile, analyzer=analyzer, llm_model=llm_model,
        )
        pr_steps = impacts_to_steps(impacts)

    # Open-PR forecasting: fetch open PRs on the same repos, price them with
    # the deep LLM analyzer, weight by merge probability, and add expected
    # deltas to the future forecast starting on the expected merge day.
    open_pr_priced: list = []
    open_pr_daily: dict = {}
    if repos and include_open_prs:
        try:
            from src.pr_scanner.open_prs import (
                analyze_open_prs, list_open_prs_many, to_step_series,
            )
            open_prs = list_open_prs_many(repos)
            if open_prs:
                open_pr_priced = analyze_open_prs(
                    open_prs, profile=profile,
                    llm_model=(llm_model or "us.anthropic.claude-sonnet-4-6"),
                )
                open_pr_daily = to_step_series(
                    open_pr_priced, cutoff=cutoff, horizon_days=14,
                )
        except Exception:  # noqa: BLE001
            pass

    step_df = pd.DataFrame(columns=["ds", "pr_cum_usd"])
    tuned = None
    if model == "prophet":
        if pr_steps:
            forecast, step_df = forecast_with_pr_regressor(
                history_df, cutoff=cutoff, pr_steps=pr_steps,
            )
        else:
            forecast = forecast_next_7_days(history_df, cutoff=cutoff)
    elif model == "ewm":
        forecast, tuned = forecast_auto(history_df, cutoff=cutoff)
        if pr_steps:
            from src.forecast.timeseries import build_step_series
            step_df = build_step_series(
                pr_steps,
                start=history_df["day"].min().date(),
                end=cutoff + timedelta(days=7),
            )
    elif model == "aws":
        forecast = forecast_from_ce(
            cutoff=cutoff, profile=profile, service=service,
        )
        if pr_steps:
            from src.forecast.timeseries import build_step_series
            step_df = build_step_series(
                pr_steps,
                start=history_df["day"].min().date(),
                end=cutoff + timedelta(days=7),
            )
    elif model == "lightgbm":
        forecast = forecast_lightgbm(history_df, cutoff=cutoff)
        if pr_steps:
            from src.forecast.timeseries import build_step_series
            step_df = build_step_series(
                pr_steps,
                start=history_df["day"].min().date(),
                end=cutoff + timedelta(days=7),
            )
    else:
        raise ValueError(f"unknown model: {model}")

    adjusted = adjust(forecast, pr_step_series=step_df)

    # Layer in open-PR expected deltas on top of the future forecast.
    # Each `adjusted` point gets bumped by the sum of expected daily deltas
    # from open PRs whose expected merge date is on or before that day.
    if open_pr_daily:
        for point in adjusted:
            bump = float(open_pr_daily.get(point.target_date, 0.0))
            point.adjusted_usd = max(0.0, point.adjusted_usd + bump)
            point.lower_usd = max(0.0, point.lower_usd + bump)
            point.upper_usd = max(0.0, point.upper_usd + bump)

    daily_pr_series = [
        {"day": pd.Timestamp(r["ds"]).date().isoformat(),
         "pr_cum_usd": float(r["pr_cum_usd"])}
        for _, r in step_df.iterrows()
    ] if not step_df.empty else []

    suffix = f"__{service.replace(' ', '_')}" if service else ""
    out_path = (predictions_dir(account_id)
                / f"forecast_{cutoff.isoformat()}{suffix}.json")
    payload = {
        "account_id": account_id,
        "profile": profile,
        "run_cutoff": cutoff.isoformat(),
        "history_days": history_days,
        "service_filter": service,
        "model": model,
        "tuned_params": tuned_params_dict(tuned) if tuned else None,
        "pr_scan": {
            "repos": list(repos) if repos else [],
            "base_branch": base_branch,
            "lookback_days": pr_lookback_days,
            "analyzer": analyzer,
            "llm_model": llm_model,
            "impacts": impacts_to_dict(impacts),
        },
        "pr_daily_series": daily_pr_series,
        "pr_delta_daily_usd_at_cutoff": (
            float(step_df[step_df["ds"] == pd.Timestamp(cutoff)]["pr_cum_usd"].iloc[0])
            if not step_df.empty and (step_df["ds"] == pd.Timestamp(cutoff)).any()
            else 0.0
        ),
        "open_pr_scan": {
            "count": len(open_pr_priced),
            "total_expected_daily_delta_usd": round(
                sum(p.expected_daily_delta_usd for p in open_pr_priced), 4,
            ),
            "prs": _open_prs_to_dict_safe(open_pr_priced),
            "daily_expected_delta": {
                d.isoformat(): round(v, 4)
                for d, v in sorted(open_pr_daily.items())
            } if open_pr_daily else {},
        },
        "forecast": to_dict(adjusted),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


if __name__ == "__main__":
    profile = os.environ.get("AWS_PROFILE")
    repos_env = os.environ.get("COSTSENSE_REPOS", "")
    repos = [r.strip() for r in repos_env.split(",") if r.strip()] or None
    out = run(profile=profile, repos=repos)
    print(f"wrote {out}")
