# costsense-forecast

Daily AWS cost forecast pipeline with PR-aware adjustments and rolling backtest accuracy.

## Flow

```
AWS Cost Explorer ─┐                       ┌─ GitHub PRs (Terraform/CDK diffs)
                   ↓                       ↓
              GitHub Actions daily cron (src/pipeline/run_daily.py)
                   │                       │
                   ↓                       ↓
        src/forecast/timeseries.py    src/pr_scanner/scan.py
        (Prophet, trained on          (parse diffs, estimate $ deltas)
         data before cutoff)
                   │                       │
                   └───────────┬───────────┘
                               ↓
                 src/pipeline/adjust.py
                 (baseline + PR deltas → 7-day adjusted forecast)
                               ↓
                 data/predictions/*.json  (predictions log)
                               ↓
              src/backtest/scorer.py (runs 7 days later)
              (join with actuals → MAE / MAPE)
                               ↓
                 data/backtest/*.json
                               ↓
              src/dashboard/app.py  (Streamlit — prediction vs actual, MAPE trend)
```

## Layout

- `src/aws/cost_explorer.py` — pulls daily cost by service via boto3
- `src/forecast/timeseries.py` — Prophet forecast trained only on data before the cutoff
- `src/pr_scanner/scan.py` — parses IaC diffs from PRs, estimates $ deltas
- `src/pipeline/adjust.py` — combines baseline forecast + PR deltas
- `src/pipeline/run_daily.py` — orchestrator invoked by the cron
- `src/backtest/scorer.py` — replays 7-day-old predictions, computes MAE/MAPE
- `src/dashboard/app.py` — Streamlit dashboard for predictions vs actuals and rolling MAPE
- `data/predictions/` — daily forecast files (one JSON per run, tagged with target date)
- `data/actuals/` — cached actual cost per date
- `data/backtest/` — scorer output
- `.github/workflows/daily.yml` — daily cron

## Run locally

```bash
pip install -r requirements.txt
export AWS_PROFILE=dil-team-hackfest    # account 609400232087
python -m src.pipeline.run_daily
streamlit run src/dashboard/app.py
```

## Status

Skeleton only — each module has a working stub that reads/writes the right shape so the pipeline runs end-to-end on synthetic data. Wire up real Cost Explorer / real PR diff parsing next.
