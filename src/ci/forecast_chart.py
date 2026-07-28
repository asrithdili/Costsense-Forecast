"""Plotly forecast chart for GitHub Actions job summaries."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go

from src.ci.forecast_context import ForecastContext

# Match Dashboard chart tokens (costsense_theme.C) without importing Streamlit.
_ACTUAL = "#2B8A3E"   # C.GOOD
_PREDICTED = "#3B5BDB"  # C.INFO
_PREDICTED_BAND = "rgba(59,91,219,0.15)"


def _as_python_datetimes(values) -> list[datetime]:
    """Kaleido JSON export cannot serialize pandas.Timestamp."""
    return [pd.Timestamp(v).to_pydatetime() for v in pd.to_datetime(values)]


def _as_python_datetime(value) -> datetime:
    return pd.Timestamp(value).to_pydatetime()


def build_forecast_figure(
    ctx: ForecastContext,
    *,
    title: str | None = None,
) -> go.Figure:
    """Mirror the Dashboard's main forecast chart (without replay overlay)."""
    fig = go.Figure()
    hist_df = ctx.hist_df
    fc_df = ctx.fc_df
    pr_series_df = ctx.pr_series_df
    cutoff = ctx.cutoff

    if not hist_df.empty:
        fig.add_trace(go.Scatter(
            x=_as_python_datetimes(hist_df["day"]),
            y=hist_df["actual_usd"].tolist(),
            mode="lines+markers", name="actual (Cost Explorer)",
            line=dict(color=_ACTUAL, width=2.5, shape="spline",
                      smoothing=1.0),
        ))

    if not fc_df.empty:
        fc_x = _as_python_datetimes(fc_df["target_date"])
        fc_upper = fc_df["upper_usd"].tolist()
        fc_lower = fc_df["lower_usd"].tolist()
        fc_baseline = fc_df["baseline_usd"].tolist()
        fc_adjusted = fc_df["adjusted_usd"].tolist()
        fig.add_trace(go.Scatter(
            x=fc_x, y=fc_upper,
            mode="lines",
            line=dict(width=0, shape="spline", smoothing=1.0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=fc_x, y=fc_lower,
            mode="lines", fill="tonexty", name="forecast interval",
            line=dict(width=0, shape="spline", smoothing=1.0),
            fillcolor=_PREDICTED_BAND,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=fc_x, y=fc_baseline,
            mode="lines+markers", name="baseline forecast",
            line=dict(color=_PREDICTED, width=2, dash="dot",
                      shape="spline", smoothing=1.0),
        ))
        fig.add_trace(go.Scatter(
            x=fc_x, y=fc_adjusted,
            mode="lines+markers", name="adjusted (baseline + PR delta)",
            line=dict(color=_PREDICTED, width=2.5, shape="spline",
                      smoothing=1.0),
        ))
        # Plotly <6 crashes in add_vline(..., annotation_text=...) on date axes
        # (TypeError: int + datetime.date). Use a shape + label instead.
        cutoff_dt = _as_python_datetime(cutoff)
        fig.add_shape(
            type="line",
            x0=cutoff_dt,
            x1=cutoff_dt,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(dash="dash", color="#888", width=1),
        )
        fig.add_annotation(
            x=cutoff_dt,
            y=1,
            yref="paper",
            text="cutoff",
            showarrow=False,
            yanchor="bottom",
            font=dict(size=11, color="#888"),
        )

    if not pr_series_df.empty:
        fig.add_trace(go.Scatter(
            x=_as_python_datetimes(pr_series_df["day"]),
            y=pr_series_df["pr_cum_usd"].tolist(),
            mode="lines", name="PR-attributable ($/day)",
            line=dict(color="#E27D60", width=1.5, dash="dot"),
            hovertemplate="%{x}<br>PR delta $%{y:,.2f}<extra></extra>",
        ))

    chart_title = title or (
        f"Cost forecast — account {ctx.account_id} ({ctx.model})"
    )
    fig.update_layout(
        title=chart_title,
        height=440, margin=dict(l=10, r=10, t=50, b=10),
        yaxis_title="USD / day", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    return fig


def save_forecast_chart(
    ctx: ForecastContext,
    output_path: str | Path,
    *,
    title: str | None = None,
    width: int = 1200,
    height: int = 500,
    scale: int = 2,
) -> Path:
    """Render *ctx* to a PNG for embedding in GITHUB_STEP_SUMMARY."""
    path = Path(output_path)
    fig = build_forecast_figure(ctx, title=title)
    fig.write_image(str(path), width=width, height=height, scale=scale)
    return path
