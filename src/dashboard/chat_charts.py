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
    seen, _lists = _tool_result_numbers_and_lists(tool_calls)
    return seen


def _tool_result_numbers_and_lists(
    tool_calls: list,
) -> tuple[set[float], list[list[float]]]:
    """Walk every tool_result and collect BOTH:

      * ``seen`` — every raw numeric value that appeared anywhere
      * ``numeric_lists`` — every homogeneous list of >=3 numbers
        (so we can compute mean/sum/max later and still call a
        derived value "grounded" if it matches an aggregate)

    The list-collection matters because the bot legitimately computes
    values like "30-day average" from the ``daily`` list of a
    ``cost_by_service`` call. The raw list is in the tool output; the
    mean is not — but it's an honest derivation, so we accept it.
    """
    seen: set[float] = set()
    numeric_lists: list[list[float]] = []

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            # Collect this list if it's a homogeneous run of numbers —
            # or a list of dicts where each dict has numeric fields
            # (the shape of `daily` under cost_by_service). For the
            # latter case, one list per numeric key discovered.
            flat: list[float] = []
            for v in obj:
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    try:
                        flat.append(float(v))
                    except (TypeError, ValueError):
                        pass
            if len(flat) >= 3:
                numeric_lists.append(flat)
            # Also gather numeric-valued keys from lists-of-dicts.
            if obj and isinstance(obj[0], dict):
                per_key: dict[str, list[float]] = {}
                for entry in obj:
                    if not isinstance(entry, dict):
                        continue
                    for k, v in entry.items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            per_key.setdefault(k, []).append(float(v))
                for values in per_key.values():
                    if len(values) >= 3:
                        numeric_lists.append(values)
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
        # Prefer `full_output` (untruncated dict/list) when the ToolCall
        # carries it — the chart-hallucination guard needs the full data
        # to find the daily-list a "30-day average = $X" claim was derived
        # from. Fall back to parsing `output_summary` for older ToolCall
        # instances that pre-date the full_output field.
        full = getattr(call, "full_output", None)
        if full is not None:
            _walk(full)
            continue
        summary = getattr(call, "output_summary", "") or ""
        try:
            parsed = json.loads(summary.rstrip("…"))
            _walk(parsed)
        except (json.JSONDecodeError, ValueError):
            _walk(summary)
    return seen, numeric_lists


def _value_grounded_by_aggregate(
    value: float, numeric_lists: list[list[float]],
) -> bool:
    """True when *value* matches the mean, sum, max, min, or falls
    within [min, max] of any collected numeric list, within 5%.

    LLMs computing 30-day averages produce numbers that don't appear
    verbatim in the tool output — but they ARE honest derivations from
    lists that DO appear. This function accepts those cases.
    """
    tolerance = max(0.5, abs(value) * 0.05)
    for values in numeric_lists:
        if not values:
            continue
        mean = sum(values) / len(values)
        total = sum(values)
        lo, hi = min(values), max(values)
        for anchor in (mean, total, lo, hi):
            if abs(value - anchor) <= tolerance:
                return True
        # Range check: if the value falls comfortably between min and
        # max, it's likely a legitimate sample the model picked from
        # the series.
        if lo <= value <= hi:
            return True
    return False


def _values_grounded(
    chart: dict,
    tool_numbers: set[float],
    numeric_lists: list[list[float]] | None = None,
) -> tuple[bool, list[float]]:
    """Return ``(ok, offending_values)``. ok=True when every y value in
    every series is grounded — either it appears in ``tool_numbers`` at
    4dp, OR it matches an aggregate (mean / sum / min / max / in-range)
    of any real ``numeric_lists`` from tool output. offending_values
    lists the first few values that failed — used for the warning banner
    so the user (and we during debugging) can see exactly what mismatched.

    Aggregate acceptance is the "honest derivation" carve-out. Example:
    the bot legitimately plots a 30-day average even though 30-day-average
    isn't a value the API ever returned — the DAILY LIST that produced
    that average IS in the API response. The guard treats the mean of
    that list as grounded because it's a real derivation of real data,
    not a fabrication.

    Prediction charts (with ``prediction_basis``) are skipped here — their
    "Change" and "Projected" bars are DERIVED from historical values,
    so they naturally don't appear in tool_results. Those charts are
    ground-checked instead via ``_prediction_grounded`` on their
    ``prediction_basis`` inputs (which DO need to come from tool_results).
    """
    if "prediction_basis" in chart:
        return True, []
    numeric_lists = numeric_lists or []
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
            if tolerated:
                continue
            # Aggregate carve-out: does this value match mean / sum / min /
            # max of any real list, or fall inside the [min, max] range of
            # any real list? This is what accepts honest derivations like
            # a 30-day average or a top-of-range projection.
            if _value_grounded_by_aggregate(rounded, numeric_lists):
                continue
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

    # If the chart claims to be a prediction (has `prediction_basis`), it
    # must be a 3-bar Current/Change/Projected shape. The three x-labels
    # are how the arithmetic guard finds the values it should reconcile.
    basis = chart.get("prediction_basis")
    if basis is not None:
        if not isinstance(basis, dict):
            return False, "prediction_basis must be an object"
        if ctype != "bar":
            return False, "prediction charts must be type=bar"
        if len(series) != 1:
            return False, (
                f"prediction charts must have exactly one series; "
                f"got {len(series)}"
            )
        # Accept labels that START WITH the three required base words,
        # case-insensitively. This lets the model add honest qualifiers
        # like "Change (ASSUMED)" or "Projected (low confidence)" without
        # us rejecting the chart — the base word is what identifies which
        # bar is which, the parenthetical is prose.
        xs = [str(x).strip().lower() for x in series[0]["x"]]
        required = ("current", "change", "projected")
        if len(xs) != 3 or not all(
            xs[i].startswith(required[i]) for i in range(3)
        ):
            return False, (
                f"prediction chart x-labels must start with "
                f"['Current', 'Change', 'Projected'] "
                f"(qualifiers in parentheses allowed); "
                f"got {series[0]['x']}"
            )
        for field in ("current_grounding", "rate_grounding"):
            if not isinstance(basis.get(field), list):
                return False, f"prediction_basis.{field} must be a list"
        if not isinstance(basis.get("note"), str) or not basis["note"].strip():
            return False, "prediction_basis.note must be a non-empty string"
    return True, ""


def _prediction_arithmetic_ok(
    chart: dict,
) -> tuple[bool, str]:
    """When `prediction_basis` is present, verify the third bar equals the
    first + second (within $0.50 or 1%). LLMs get this wrong more often
    than you'd expect — the point of the check is to catch cases where
    the "Projected" number was computed by a different path than the
    displayed "Change".
    """
    if "prediction_basis" not in chart:
        return True, ""
    series = chart["series"][0]
    y = series["y"]
    if len(y) != 3:
        return False, f"prediction chart must have 3 bars, got {len(y)}"
    try:
        current = float(y[0])
        change = float(y[1])
        projected = float(y[2])
    except (TypeError, ValueError):
        return False, "prediction chart bars must be numeric"
    expected = current + change
    tolerance = max(0.50, abs(expected) * 0.01)
    if abs(projected - expected) > tolerance:
        return False, (
            f"arithmetic mismatch: Current ({current:.2f}) + Change "
            f"({change:+.2f}) = {expected:.2f}, but Projected shown as "
            f"{projected:.2f} (diff {projected - expected:+.2f} exceeds "
            f"tolerance ${tolerance:.2f})"
        )
    return True, ""


def _prediction_grounded(
    chart: dict,
    tool_numbers: set[float],
    numeric_lists: list[list[float]] | None = None,
) -> tuple[bool, list[float]]:
    """When `prediction_basis` is present, verify that every value in
    both `current_grounding` and `rate_grounding` traces back to a real
    tool_result.

    A value is considered grounded when any of these are true:
      1. It matches a raw number in the tool output (exact / penny drift)
      2. It matches an aggregate (mean/sum/max/min) of any numeric list
         in the tool output within 5% — accepts values like "30-day
         average = $57/day" that are derived by honest arithmetic from
         values that ARE in the output
      3. It falls within [min, max] of a numeric list in the output —
         accepts values the model quoted as a representative sample
         from an observed range

    Only when all three fail is the value labelled "fabricated" and
    added to the offending list.
    """
    if "prediction_basis" not in chart:
        return True, []
    numeric_lists = numeric_lists or []
    basis = chart["prediction_basis"]
    offending: list[float] = []

    # ASSUMED-values escape hatch: when the basis note names its output
    # as ASSUMED / assumed / illustrative / rough-estimate, we accept
    # that the model is doing an honest what-if without historical
    # backing. Grounding of `current_grounding` is still enforced (that
    # value SHOULD be a real reading) but `rate_grounding` is allowed
    # to be empty or unverified because the whole point of an ASSUMED
    # chart is that the rate isn't measured.
    note = str(basis.get("note", "")).lower()
    is_assumed = any(
        marker in note for marker in
        ("assumed", "illustrative", "rough", "not measured", "no precedent")
    )

    fields_to_check = ["current_grounding"]
    if not is_assumed:
        fields_to_check.append("rate_grounding")

    for field in fields_to_check:
        for v in basis.get(field, []):
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                offending.append(v)
                continue
            rounded = round(float(v), 4)
            if rounded in tool_numbers:
                continue
            # Penny-drift tolerance against raw numbers.
            if any(
                abs(rounded - t) <= max(0.01, abs(t) * 0.01)
                for t in tool_numbers
            ):
                continue
            # Aggregate / range tolerance against numeric lists.
            if _value_grounded_by_aggregate(float(v), numeric_lists):
                continue
            offending.append(v)
            if len(offending) >= 5:
                return False, offending
    return len(offending) == 0, offending


def _build_figure(chart: dict) -> go.Figure:
    """Assemble the Plotly figure using the shared Dashboard chart theme
    (spline lines + markers + brand colour). Bar charts use a similar
    palette."""
    ctype = chart.get("type", "line")
    bar_color = C.INFO if "prediction_basis" in chart else C.BRAND
    fig = go.Figure()
    for series in chart.get("series", []):
        name = str(series.get("name") or "")
        x = series.get("x") or []
        y = series.get("y") or []
        if ctype == "bar":
            fig.add_trace(go.Bar(
                x=x, y=y, name=name,
                marker=dict(color=bar_color),
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

    tool_numbers, numeric_lists = _tool_result_numbers_and_lists(tool_calls)

    for i, chart in enumerate(charts, start=1):
        ok, err = _schema_ok(chart)
        if not ok:
            st.warning(
                f"**Chart {i} not rendered.** Malformed chart block: "
                f"{err}. The bot returned a chart shape we can't render "
                f"safely — showing the accompanying text instead."
            )
            continue

        # For prediction charts, verify the derived arithmetic AND that
        # the model's `prediction_basis` inputs came from tool_results.
        # For regular charts, verify every y value came from a tool_result.
        # Order matters: schema first (cheapest), then arithmetic (pure
        # math on the chart), then grounding (walks tool_results).
        arith_ok, arith_err = _prediction_arithmetic_ok(chart)
        if not arith_ok:
            st.warning(
                f"**Chart {i} suppressed — arithmetic error.** {arith_err}. "
                f"Prediction charts must satisfy Projected = Current + "
                f"Change. The bot's numbers don't add up, so the chart "
                f"is not rendered."
            )
            continue

        pred_ok, pred_offending = _prediction_grounded(
            chart, tool_numbers, numeric_lists,
        )
        if not pred_ok:
            preview = ", ".join(str(v) for v in pred_offending[:5])
            st.warning(
                f"**Chart {i} suppressed by prediction guard.** The bot "
                f"claimed to use these historical values, but they don't "
                f"appear in any tool call this turn: `{preview}`. "
                f"Prediction charts must trace their inputs back to real "
                f"AWS data — this one didn't."
            )
            continue

        grounded, offending = _values_grounded(
            chart, tool_numbers, numeric_lists,
        )
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

        # For prediction charts, show the model's basis note underneath —
        # named dates + dollar values so the user can audit the reasoning.
        basis = chart.get("prediction_basis")
        if isinstance(basis, dict) and basis.get("note"):
            st.caption(f"**Prediction basis:** {basis['note']}")

        source_tool = chart.get("source_tool")
        if source_tool:
            st.caption(
                f"Source tool: `{source_tool}` · every value verified "
                f"against tool output from this turn."
            )
