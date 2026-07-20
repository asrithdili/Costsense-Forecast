"""Per-account file layout. All predictions/actuals/backtests namespaced by account id."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


def predictions_dir(account_id: str) -> Path:
    p = DATA_DIR / account_id / "predictions"
    p.mkdir(parents=True, exist_ok=True)
    return p


def actuals_dir(account_id: str) -> Path:
    p = DATA_DIR / account_id / "actuals"
    p.mkdir(parents=True, exist_ok=True)
    return p


def backtest_dir(account_id: str) -> Path:
    p = DATA_DIR / account_id / "backtest"
    p.mkdir(parents=True, exist_ok=True)
    return p


def known_accounts() -> list[str]:
    if not DATA_DIR.exists():
        return []
    return sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
