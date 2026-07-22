"""LightGBM forecaster for daily AWS cost.

Trains on lag + rolling-mean + calendar features using history STRICTLY
before the cutoff, then recursively forecasts the next N days.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from src.forecast.timeseries import ForecastPoint

_LAGS = (1, 7, 14, 28)
_MIN_TRAIN_ROWS = 35
_lgb = None
_lgb_error: str | None = None


def _get_lgb():
    """Import lightgbm on first use so the dashboard loads without it."""
    global _lgb, _lgb_error
    if _lgb is not None:
        return _lgb
    if _lgb_error is not None:
        raise RuntimeError(_lgb_error)
    try:
        import lightgbm as lgb_mod
        import sklearn  # noqa: F401 — required by lightgbm.sklearn
    except (ImportError, OSError) as e:
        _lgb_error = (
            "lightgbm is unavailable. Install with "
            "`pip install lightgbm scikit-learn`. "
            "On macOS you may also need `brew install libomp`."
            f" ({e})"
        )
        raise RuntimeError(_lgb_error) from e
    _lgb = lgb_mod
    return _lgb


def _prepare_series(history: pd.DataFrame) -> pd.Series:
    hist = history.copy()
    hist["day"] = pd.to_datetime(hist["day"])
    hist = hist.sort_values("day").drop_duplicates("day", keep="last")
    s = hist.set_index("day")["amount_usd"].astype(float)
    if s.empty:
        return s
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    return s.reindex(full_idx).ffill().fillna(0.0)


def _feature_row(series: pd.Series, target_idx: int) -> dict[str, float] | None:
    max_lag = max(_LAGS)
    if target_idx < max_lag:
        return None
    row: dict[str, float] = {}
    for lag in _LAGS:
        row[f"lag_{lag}"] = float(series.iloc[target_idx - lag])
    window = series.iloc[target_idx - 7:target_idx]
    row["roll_mean_7"] = float(window.mean())
    window14 = series.iloc[target_idx - 14:target_idx]
    row["roll_mean_14"] = float(window14.mean())
    ts = series.index[target_idx]
    row["dow"] = float(ts.weekday())
    row["is_weekend"] = float(ts.weekday() >= 5)
    return row


def _build_training_frame(
    series: pd.Series, cutoff: date,
) -> tuple[pd.DataFrame, pd.Series, list[date]]:
    rows: list[dict[str, float]] = []
    targets: list[float] = []
    dates: list[date] = []
    cutoff_ts = pd.Timestamp(cutoff)
    for i in range(len(series)):
        if series.index[i] >= cutoff_ts:
            break
        feats = _feature_row(series, i)
        if feats is None:
            continue
        rows.append(feats)
        targets.append(float(series.iloc[i]))
        dates.append(series.index[i].date())
    if not rows:
        return pd.DataFrame(), pd.Series(dtype=float), []
    return pd.DataFrame(rows), pd.Series(targets, dtype=float), dates


def forecast_lightgbm(
    history: pd.DataFrame,
    cutoff: date,
    horizon_days: int = 7,
) -> list[ForecastPoint]:
    lgb = _get_lgb()

    series = _prepare_series(history)
    train_x, train_y, train_dates = _build_training_frame(series, cutoff=cutoff)
    if len(train_x) < _MIN_TRAIN_ROWS:
        raise ValueError(
            f"need at least {_MIN_TRAIN_ROWS} training rows before cutoff {cutoff}, "
            f"got {len(train_x)}",
        )

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=120,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=-1,
    )
    model.fit(train_x, train_y)

    residuals = train_y - model.predict(train_x)
    std = float(residuals.std()) if len(residuals) >= 3 else 0.0

    extended = series.copy()
    cutoff_ts = pd.Timestamp(cutoff)
    if cutoff_ts not in extended.index:
        extended = pd.concat(
            [extended, pd.Series([extended.iloc[-1]], index=[cutoff_ts])],
        ).sort_index()

    out: list[ForecastPoint] = []
    for i in range(1, horizon_days + 1):
        target_date = cutoff + timedelta(days=i)
        target_ts = pd.Timestamp(target_date)
        if target_ts not in extended.index:
            extended = pd.concat(
                [extended, pd.Series([extended.iloc[-1]], index=[target_ts])],
            ).sort_index()
        idx = extended.index.get_loc(target_ts)
        feats = _feature_row(extended, idx)
        if feats is None:
            raise ValueError(f"insufficient history to forecast {target_date}")
        pred = max(0.0, float(model.predict(pd.DataFrame([feats]))[0]))
        extended.iloc[idx] = pred
        out.append(ForecastPoint(
            target_date=target_date,
            predicted_usd=pred,
            lower_usd=max(0.0, pred - 1.28 * std),
            upper_usd=max(0.0, pred + 1.28 * std),
        ))
    return out


def in_sample_fit_lightgbm(
    history: pd.DataFrame,
    cutoff: date,
    lookback_days: int = 30,
) -> list[ForecastPoint]:
    """Train once at *cutoff*, return in-sample fitted values on recent days."""
    lgb = _get_lgb()

    series = _prepare_series(history)
    train_x, train_y, train_dates = _build_training_frame(series, cutoff=cutoff)
    if len(train_x) < _MIN_TRAIN_ROWS:
        raise ValueError(
            f"need at least {_MIN_TRAIN_ROWS} training rows before cutoff {cutoff}, "
            f"got {len(train_x)}",
        )

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=120,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=5,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=-1,
    )
    model.fit(train_x, train_y)
    preds = model.predict(train_x)
    residuals = train_y - preds
    std = float(residuals.std()) if len(residuals) >= 3 else 0.0

    lookback_start = cutoff - timedelta(days=lookback_days)
    out: list[ForecastPoint] = []
    for day, pred in zip(train_dates, preds):
        if day < lookback_start:
            continue
        pred = max(0.0, float(pred))
        out.append(ForecastPoint(
            target_date=day,
            predicted_usd=pred,
            lower_usd=max(0.0, pred - 1.28 * std),
            upper_usd=max(0.0, pred + 1.28 * std),
        ))
    return out
