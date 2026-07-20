"""Backtest scorer — joins 7-day-old predictions with actuals, writes MAE/MAPE."""
from __future__ import annotations

import json
from datetime import date, timedelta

from src.aws.cost_explorer import fetch_actual_total
from src.aws.profiles import resolve
from src.pipeline.paths import actuals_dir, backtest_dir, predictions_dir


def _load_actual(account_id: str, day: date, profile: str | None) -> float:
    cached = actuals_dir(account_id) / f"{day.isoformat()}.json"
    if cached.exists():
        return float(json.loads(cached.read_text())["amount_usd"])
    amount = fetch_actual_total(day, profile=profile)
    cached.write_text(json.dumps({"day": day.isoformat(), "amount_usd": amount}))
    return amount


def score_for_target(target_day: date, profile: str | None = None) -> dict | None:
    info = resolve(profile) if profile else None
    account_id = (info.account_id if info else None) or "unknown"

    origin = target_day - timedelta(days=7)
    forecast_file = predictions_dir(account_id) / f"forecast_{origin.isoformat()}.json"
    if not forecast_file.exists():
        return None

    payload = json.loads(forecast_file.read_text())
    row = next(
        (p for p in payload["forecast"] if p["target_date"] == target_day.isoformat()),
        None,
    )
    if row is None:
        return None

    actual = _load_actual(account_id, target_day, profile=profile)
    predicted = float(row["adjusted_usd"])
    abs_err = abs(predicted - actual)
    ape = abs_err / actual if actual else None

    result = {
        "account_id": account_id,
        "target_date": target_day.isoformat(),
        "run_cutoff": payload["run_cutoff"],
        "predicted_usd": predicted,
        "actual_usd": actual,
        "abs_error_usd": abs_err,
        "ape": ape,
    }
    (backtest_dir(account_id) / f"score_{target_day.isoformat()}.json").write_text(
        json.dumps(result, indent=2)
    )
    return result


def score_today(profile: str | None = None) -> dict | None:
    return score_for_target(date.today(), profile=profile)


if __name__ == "__main__":
    import os

    r = score_today(profile=os.environ.get("AWS_PROFILE"))
    print(json.dumps(r, indent=2) if r else "no forecast to score for today")
