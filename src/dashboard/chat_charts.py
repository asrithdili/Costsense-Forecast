"""Chart rendering + hallucination guard for the AI chatbot page.

The bot emits chart blocks as fenced JSON inside its markdown reply. This
module extracts those blocks, validates them against a strict schema, cross-
checks every numeric value against the tool-call outputs from the same turn,
and renders whatever survives with the shared Dashboard chart theme.

Contract the bot follows (taught by the SYSTEM prompt in chat_agent.py):

    ```chart
    {
      "type": "line" | "bar",
      "title": "...",
      "x_title": "...",
      "y_title": "...",
      "series": [
        {"name": "...", "x": [...], "y": [...]},
        ...
      ],
      "source_tool": "<tool name that produced these numbers>"
    }
    ```

Hallucination guard
-------------------
Every ``y`` value in every series must appear in at least one tool_result
from this turn. If any value doesn't match a tool output (to a small
rounding tolerance), the chart is suppressed and a warning is rendered
in its place. No exceptions — this is the deterministic backstop for the
"don't invent numbers" rule the SYSTEM prompt states.
"""
from __future__ import annotations

import json
import re
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from src.dashboard.costsense_theme import C, plotly_layout


# Fenced code block with language tag = "chart". Non-greedy so multiple
# blocks in one reply each get matched individually.
_CHART_FENCE_RE = re.compile(
    r"```chart\s*\n(.*?)\n```",
    re.DOTALL,
)


def strip_chart_blocks(reply: str) -> tuple[str, list[dict]]:
    """Extract every valid ``chart`` block from the reply and return
    ``(text_without_blocks, list_of_chart_dicts)``.

    JSON parse errors → block removed, no chart returned. That's a soft
    failure: better to render prose without a broken chart than to fail
    the whole message.
    """
    charts: list[dict] = []

    def _consume(match: re.Match) -> str:
        raw = match.group(1).strip()
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return ""  # Silently drop malformed blocks.
        if not isinstance(data, dict):
            return ""
        charts.append(data)
        return ""

    stripped = _CHART_FENCE_RE.sub(_consume, reply)
    # Collapse the (possibly leftover) blank lines from removed blocks.
    stripped = re.sub(r"\n{3,}", "\n\n", stripped).strip()
    return stripped, charts


def _all_tool_result_numbers(tool_calls: list) -> set[float]:
    """Every numeric value that appeared in any tool_result this turn.

    Rounds each value to 4 decimals — LLMs commonly re-serialise floats
    with different precision (12.499999 vs 12.5). Using a rounded set
    lets us detect drift-of-a-cent as the same number. The chart guard
    then compares candidate values against this set at 4 dp.
    """
    seen: set[float] = set()

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            try:
                seen.add(round(float(obj), 4))
            except (TypeError, ValueError):
                pass
        elif isinstance(obj, str):
            # Numbers embedded in JSON-serialised tool_output strings.
            for m in re.finditer(r"-?\d+(?:\.\d+)?", obj):
                try:
                    seen.add(round(float(m.group()), 4))
                except ValueError:
                    pass

    for call in tool_calls:
        # tool_calls entries carry an `output_summary` string (see
        # chat_agent.ToolCall). Parse each as best we can and walk.
        summary = getattr(call, "output_summary", "") or ""
        try:
            parsed = json.loads(summary.rstrip("…"))
            _walk(parsed)
        except (json.JSONDecodeError, ValueError):
            # Fall back to a regex walk over the raw string.
            _walk(summary)
    return seen


def _values_grounded(chart: dict, tool_numbers: set[float]) -> tuple[bool, list[float]]:
    """Return ``(ok, offending_values)``. ok=True when every y value in
    every series shows up in ``tool_numbers`` at 4dp. offending_values
    lists the first few values that failed — used for the warning banner
    so the user (and we during debugging) can see exactly what mismatched.
    """
    offending: list[float] = []
    for series in chart.get("series", []):
        for y in series.get("y", []):
            if not isinstance(y, (int, float)) or isinstance(y, bool):
                offending.append(y)  # non-numeric y = suspicious
                continue
            rounded = round(float(y), 4)
            if rounded in tool_numbers:
                continue
            # Allow 1% or $0.01 tolerance for LLM re-rounding of shown values.
            tolerated = any(
                abs(rounded - t) <= max(0.01, abs(t) * 0.01)
                for t in tool_numbers
            )
            if not tolerated:
                offending.append(y)
                if len(offending) >= 5:
                    return False, offending
    return len(offending) == 0, offending


def _schema_ok(chart: dict) -> tuple[bool, str]:
    """Validate the chart dict shape. Returns (ok, error_message)."""
    ctype = chart.get("type")
    if ctype not in ("line", "bar"):
        return False, f"unsupported chart type: {ctype!r}"
    series = chart.get("series")
    if not isinstance(series, list) or not series:
        return False, "chart has no series"
    for i, s in enumerate(series):
        if not isinstance(s, dict):
            return False, f"series[{i}] is not an object"
        if not isinstance(s.get("x"), list) or not s["x"]:
            return False, f"series[{i}] missing x"
        if not isinstance(s.get("y"), list) or not s["y"]:
            return False, f"series[{i}] missing y"
        if len(s["x"]) != len(s["y"]):
            return False, (
                f"series[{i}] x/y length mismatch: "
                f"{len(s['x'])} vs {len(s['y'])}"
            )
    return True, ""


def _build_figure(chart: dict) -> go.Figure:
    """Assemble the Plotly figure using the shared Dashboard chart theme
    (spline lines + markers + brand colour). Bar charts use a similar
    palette."""
    ctype = chart.get("type", "line")
    fig = go.Figure()
    for series in chart.get("series", []):
        name = str(series.get("name") or "")
        x = series.get("x") or []
        y = series.get("y") or []
        if ctype == "bar":
            fig.add_trace(go.Bar(
                x=x, y=y, name=name,
                marker=dict(color=C.BRAND),
                hovertemplate="%{x}<br>$%{y:,.2f}<extra>" + name + "</extra>",
            ))
        else:  # line
            fig.add_trace(go.Scatter(
                x=x, y=y, name=name,
                mode="lines+markers",
                line=dict(width=2.5, shape="spline", smoothing=1.0),
                marker=dict(size=6),
                hovertemplate="%{x}<br>%{y:,.2f}<extra>" + name + "</extra>",
            ))
    layout = plotly_layout(height=380)
    layout["yaxis_title"] = str(chart.get("y_title") or "")
    layout["xaxis_title"] = str(chart.get("x_title") or "")
    layout["title"] = str(chart.get("title") or "")
    layout["hovermode"] = "x unified" if ctype == "line" else "closest"
    fig.update_layout(**layout)
    return fig


def render_charts_inline(
    charts: list[dict],
    tool_calls: list,
) -> None:
    """Render every chart from ``charts``. Each chart is validated
    against schema + cross-checked against the tool-call outputs from
    the same turn. A block that fails either check is replaced with a
    warning box naming the reason — no silent drop.

    ``tool_calls`` is the same list attached to the chat_display entry
    (list of ``ToolCall`` dataclasses from chat_agent.py).
    """
    if not charts:
        return

    tool_numbers = _all_tool_result_numbers(tool_calls)

    for i, chart in enumerate(charts, start=1):
        ok, err = _schema_ok(chart)
        if not ok:
            st.warning(
                f"**Chart {i} not rendered.** Malformed chart block: "
                f"{err}. The bot returned a chart shape we can't render "
                f"safely — showing the accompanying text instead."
            )
            continue

        grounded, offending = _values_grounded(chart, tool_numbers)
        if not grounded:
            preview = ", ".join(str(v) for v in offending[:5])
            st.warning(
                f"**Chart {i} suppressed by hallucination guard.** The "
                f"bot tried to plot values that don't appear in any tool "
                f"call this turn: `{preview}`. Charts are only rendered "
                f"when every point traces back to a real API response — "
                f"this one didn't."
            )
            continue

        fig = _build_figure(chart)
        st.plotly_chart(fig, use_container_width=True,
                        key=f"chat_chart::{id(chart)}::{i}")

        source_tool = chart.get("source_tool")
        if source_tool:
            st.caption(
                f"Source tool: `{source_tool}` · every y-value verified "
                f"against tool output from this turn."
            )
