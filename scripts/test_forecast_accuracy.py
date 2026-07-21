"""Backtest: does the forecast get the direction right?

Runs the SAME walk-forward code the Dashboard uses to draw its
past-predictions overlay, so the numbers here match what a user
sees on the chart.

For each of N past origins, we:
  1. Take history STRICTLY before that origin (no lookahead).
  2. Run the auto-tuned EWM forecast (regime-aware — the current
     default) for the next 7 days, optionally with merged-PR steps
     from the newest saved forecast JSON.
  3. Compare the predicted 7-day total vs the prior 7-day total to
     classify the model's CALL as increase / decrease / flat.
  4. Compare the actual 7-day total vs the prior 7-day total for the
     ACTUAL direction.
  5. Score direction hits and per-day $ error (MAE, WAPE).

Usage:
  python -m scripts.test_forecast_accuracy \\
    --profile dil-data-platform-dev \\
    --origins 8
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from src.aws.cost_explorer import fetch_daily_totals
from src.aws.profiles import resolve
from src.forecast.backtest_replay import walk_forward
from src.forecast.timeseries import PrStep


@dataclass
class OriginResult:
    origin: date
    prior_7d_actual: float
    next_7d_predicted: float
    next_7d_actual: float
    call: str                   # "increase" / "decrease" / "flat"
    actual: str
    hit: bool
    daily_mae: float
    daily_wape: float
    days_scored: int


def _classify(delta_pct: float, flat_threshold_pct: float) -> str:
    if abs(delta_pct) < flat_threshold_pct:
        return "flat"
    return "increase" if delta_pct > 0 else "decrease"


def _load_saved_pr_steps(account_id: str) -> list[PrStep]:
    """Read the newest forecast JSON on disk and pull out the merged-PR
    steps the Dashboard uses. Same code path the UI runs."""
    pattern = (_REPO_ROOT / "data" / account_id / "predictions"
               / "forecast_*.json")
    files = sorted(glob.glob(str(pattern)))
    if not files:
        return []
    try:
        payload = json.loads(Path(files[-1]).read_text())
    except Exception:  # noqa: BLE001
        return []
    steps: list[PrStep] = []
    for imp in (payload.get("pr_scan") or {}).get("impacts", []) or []:
        if not imp.get("est_daily_delta_usd"):
            continue
        try:
            merge_day = date.fromisoformat(imp["merged_at"][:10])
        except (ValueError, KeyError):
            continue
        steps.append(PrStep(
            from_day=merge_day,
            delta_usd=imp["est_daily_delta_usd"],
            pr_id=f"{imp['repo']}#{imp['pr_number']}",
        ))
    return steps


def _score_origins(
    hist: pd.DataFrame,
    end: date,
    origins: int,
    stride: int,
    actual_by_day: dict[date, float],
    flat_threshold_pct: float,
    pr_steps: list[PrStep] | None,
) -> list[OriginResult]:
    """Call the SAME walk_forward the Dashboard calls, then score."""
    replay = walk_forward(
        hist,
        end=end,
        n_origins=origins,
        stride_days=stride,
        horizon_days=7,
        pr_steps=pr_steps,
        model="ewm",
    )
    by_origin: dict[date, list] = {}
    for p in replay:
        by_origin.setdefault(p.origin_date, []).append(p)

    results: list[OriginResult] = []
    for origin, points in sorted(by_origin.items()):
        prior_start = origin - timedelta(days=7)
        prior_actuals = [actual_by_day.get(prior_start + timedelta(days=i))
                         for i in range(7)]
        if any(a is None for a in prior_actuals):
            continue
        prior_7d_actual = float(sum(prior_actuals))
        if prior_7d_actual <= 0:
            continue

        err_sum = 0.0
        act_sum = 0.0
        n = 0
        pred_total = 0.0
        actual_total = 0.0
        for p in points[:7]:
            a = actual_by_day.get(p.target_date)
            pred_total += p.predicted_usd
            if a is None:
                continue
            err_sum += abs(p.predicted_usd - a)
            act_sum += a
            actual_total += a
            n += 1
        if n == 0:
            continue

        call_pct = 100 * (pred_total - prior_7d_actual) / prior_7d_actual
        actual_pct = 100 * (actual_total - prior_7d_actual) / prior_7d_actual
        results.append(OriginResult(
            origin=origin,
            prior_7d_actual=round(prior_7d_actual, 2),
            next_7d_predicted=round(pred_total, 2),
            next_7d_actual=round(actual_total, 2),
            call=_classify(call_pct, flat_threshold_pct),
            actual=_classify(actual_pct, flat_threshold_pct),
            hit=(_classify(call_pct, flat_threshold_pct)
                 == _classify(actual_pct, flat_threshold_pct)),
            daily_mae=round(err_sum / n, 2),
            daily_wape=round(err_sum / act_sum, 4) if act_sum > 0 else float("nan"),
            days_scored=n,
        ))
    return results


def _summarize(results: list[OriginResult]) -> dict:
    if not results:
        return {"n": 0}
    n = len(results)
    hits = sum(1 for r in results if r.hit)

    def _dir_stats(direction: str) -> dict:
        called = [r for r in results if r.call == direction]
        actual = [r for r in results if r.actual == direction]
        called_and_hit = [r for r in called if r.hit]
        return {
            "times_called": len(called),
            "precision":     len(called_and_hit) / len(called) if called else None,
            "times_actual":  len(actual),
            "recall":        len(called_and_hit) / len(actual) if actual else None,
        }

    err_sum = sum(r.daily_mae * r.days_scored for r in results)
    day_sum = sum(r.days_scored for r in results)
    avg_wape = sum(r.daily_wape * r.days_scored for r in results) / max(1, day_sum)

    return {
        "n": n,
        "direction_hits": hits,
        "direction_accuracy_pct": round(100 * hits / n, 1),
        "per_direction": {
            "increase": _dir_stats("increase"),
            "decrease": _dir_stats("decrease"),
            "flat":     _dir_stats("flat"),
        },
        "mae_per_day_usd": round(err_sum / day_sum, 2) if day_sum else None,
        "avg_wape_pct":    round(100 * avg_wape, 1),
    }


def _print_report(results: list[OriginResult], summary: dict) -> None:
    print()
    print("=" * 92)
    print(f"{'origin':>12}  {'prior 7d':>10}  {'pred 7d':>10}  "
          f"{'act 7d':>10}  {'call':>10}  {'actual':>10}  {'hit':>4}  "
          f"{'MAE/d':>8}  {'WAPE':>6}")
    print("-" * 92)
    for r in results:
        print(f"{r.origin.isoformat():>12}  "
              f"${r.prior_7d_actual:>9,.0f}  "
              f"${r.next_7d_predicted:>9,.0f}  "
              f"${r.next_7d_actual:>9,.0f}  "
              f"{r.call:>10}  {r.actual:>10}  "
              f"{'YES' if r.hit else 'NO':>4}  "
              f"${r.daily_mae:>7,.2f}  {100*r.daily_wape:>5.1f}%")
    print("=" * 92)
    if summary["n"] == 0:
        print("No origins scored.")
        return
    print(f"Origins scored:      {summary['n']}")
    print(f"Direction accuracy:  "
          f"{summary['direction_hits']}/{summary['n']}  "
          f"({summary['direction_accuracy_pct']}%)")
    print(f"MAE per day:         ${summary['mae_per_day_usd']:.2f}")
    print(f"Weighted APE:        {summary['avg_wape_pct']}%")
    for direction, stats in summary["per_direction"].items():
        prec = stats["precision"]
        rec = stats["recall"]
        prec_s = f"{100 * prec:.0f}%" if prec is not None else "n/a"
        rec_s = f"{100 * rec:.0f}%" if rec is not None else "n/a"
        print(f"  {direction:<9} - called {stats['times_called']}, "
              f"actual {stats['times_actual']}, "
              f"precision {prec_s}, recall {rec_s}")


def run(
    profile: str,
    origins: int,
    stride: int,
    history_days: int,
    end: date,
    flat_threshold_pct: float,
) -> tuple[list[OriginResult], dict]:
    total_days = history_days + origins * stride + 14
    start = end - timedelta(days=total_days)
    print(f"Fetching {total_days}d of daily cost from Cost Explorer "
          f"via profile `{profile}`... ({start} -> {end})")
    totals = fetch_daily_totals(start, end, profile=profile)
    if not totals:
        print("ERROR: Cost Explorer returned no data.")
        return [], {}
    print(f"  got {len(totals)} days of cost data.")

    info = resolve(profile)
    account_id = (info.account_id if info else None) or "unknown"

    hist = pd.DataFrame([{"day": pd.Timestamp(d), "amount_usd": a}
                         for d, a in totals])
    actual_by_day = {d: float(a) for d, a in totals}

    pr_steps = _load_saved_pr_steps(account_id)
    print(f"Merged-PR steps loaded from saved forecast: {len(pr_steps)}")

    results = _score_origins(hist, end, origins, stride, actual_by_day,
                             flat_threshold_pct, pr_steps=pr_steps or None)
    return results, _summarize(results)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--profile", required=True)
    ap.add_argument("--origins", type=int, default=8)
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--history-days", type=int, default=90)
    ap.add_argument("--end", type=str, default=None,
                    help="Newest date to score (default: today).")
    ap.add_argument("--flat-threshold-pct", type=float, default=2.5,
                    help="Week-over-week change under this %% is 'flat'.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    end = date.fromisoformat(args.end) if args.end else date.today()
    results, summary = run(
        profile=args.profile,
        origins=args.origins,
        stride=args.stride,
        history_days=args.history_days,
        end=end,
        flat_threshold_pct=args.flat_threshold_pct,
    )

    if args.json:
        print(json.dumps({
            "params": {**vars(args), "end": end.isoformat()},
            "summary": summary,
            "results": [{**asdict(r), "origin": r.origin.isoformat()}
                        for r in results],
        }, indent=2))
    else:
        _print_report(results, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
