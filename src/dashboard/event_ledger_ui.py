"""Event ledger for the Dashboard — manual events validated by the CostSense agent."""
from __future__ import annotations

import html
import traceback
from datetime import date, timedelta
from typing import Optional

import plotly.graph_objects as go
import streamlit as st

from src.ai_agent.event_validator import (
    EventValidationResult,
    draft_from_user_input,
    validate_forecast_event,
)
from src.dashboard.costsense_theme import C, callout, money, plotly_layout, section
from src.forecast.event_store import get_stored_events, store_events
from src.forecast.events import CATEGORIES, CostEvent, Effect
from src.forecast.scenario import Projection

_CATEGORY_COLOR = {
    "demand": C.INFO,
    "release": C.SEV["Medium"],
    "optimization": C.GOOD,
    "commitment": C.BRAND,
    "pricing": C.SEV["High"],
}



def _waterfall_bar_label(value: float) -> str:
    """Always show exact dollars — never K/M abbreviation on the chart."""
    sign = "−" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def _wrap_event_label(name: str, *, max_chars: int = 22) -> str:
    """Wrap long event names for x-axis (Plotly supports <br>)."""
    short = name.split("·")[0].strip()
    if len(short) <= max_chars:
        return short
    words = short.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "<br>".join(lines[:3])


def _signed_delta_label(value: float) -> str:
    if value >= 0:
        return f"+{_waterfall_bar_label(value)}"
    return f"−{_waterfall_bar_label(abs(value))}"


def render_event_waterfall(
    proj: Projection,
    *,
    chart_window_end: date | None = None,
) -> None:
    """Waterfall: baseline → each event → projected total for the horizon."""
    st.markdown("##### What drives the forecast")
    enabled = [ev for ev in proj.events if ev.enabled]
    if not enabled:
        st.info("No enabled events — add one below or turn an event on.")
        return

    contribs = proj.contributions()
    material = [(ev, amt) for ev, amt in contribs if abs(amt) > 0.005]

    if not material:
        window = f"{proj.dates[0]:%d %b} – {proj.dates[-1]:%d %b %Y}"
        chart_note = ""
        if chart_window_end and chart_window_end < proj.dates[-1]:
            chart_note = (
                f" The cost chart only covers through **{chart_window_end:%d %b}**; "
                "this waterfall extends further to show when events land."
            )
        summaries = "; ".join(
            f"**{ev.name[:36]}** starts {ev.start_date:%d %b %Y}"
            for ev in enabled[:3]
        )
        extra = f" (+{len(enabled) - 3} more)" if len(enabled) > 3 else ""
        st.info(
            f"{len(enabled)} enabled event(s) are in the ledger but have no "
            f"material impact in **{window}**. {summaries}{extra}.{chart_note}"
        )
        return

    contrib_sum = sum(amt for _, amt in material)
    baseline = proj.total_baseline
    projected = proj.total_expected

    x_labels = ["Baseline"]
    full_names = ["Baseline (model, no events)"]
    y_values = [baseline]
    bar_colors = [C.FAINT]
    text_labels = [_waterfall_bar_label(baseline)]

    for ev, amt in material:
        x_labels.append(_wrap_event_label(ev.name))
        full_names.append(ev.name)
        y_values.append(amt)
        bar_colors.append(C.BAD if amt >= 0 else C.GOOD)
        text_labels.append(_signed_delta_label(amt))

    x_labels.append("Projected")
    full_names.append("Projected (with events)")
    y_values.append(projected)
    bar_colors.append(C.BRAND)
    text_labels.append(_waterfall_bar_label(projected))

    fig = go.Figure(go.Bar(
        x=x_labels,
        y=y_values,
        marker_color=bar_colors,
        text=text_labels,
        textposition="outside",
        customdata=full_names,
        hovertemplate=(
            "<b>%{customdata}</b><br>$%{y:,.0f}<extra></extra>"
        ),
        cliponaxis=False,
    ))

    y_min = min(0.0, min(y_values))
    y_max = max(y_values)
    span = y_max - y_min
    pad = max(span * 0.12, abs(contrib_sum) * 0.5, 250)

    lay = plotly_layout(height=max(360, 80 + 28 * len(x_labels)))
    lay["showlegend"] = False
    lay["yaxis_title"] = f"{proj.horizon_days}-day spend (USD)"
    lay["yaxis_tickformat"] = "$,.0f"
    lay["yaxis_range"] = [y_min - pad * 0.05, y_max + pad]
    lay["margin"] = dict(l=56, r=24, t=40, b=90)
    lay["xaxis"] = dict(tickangle=0, automargin=True)
    fig.update_layout(**lay)
    st.plotly_chart(fig, use_container_width=True)
    if abs(contrib_sum) > 0.5:
        sign = "+" if contrib_sum >= 0 else "−"
        horizon = (
            f"{proj.dates[0]:%d %b} – {proj.dates[-1]:%d %b %Y}"
        )
        st.caption(
            f"{horizon} · baseline {_waterfall_bar_label(proj.total_baseline)} "
            f"{sign} {_waterfall_bar_label(abs(contrib_sum))} from "
            f"{len(material)} event(s) = "
            f"{_waterfall_bar_label(proj.total_expected)} projected."
        )


def render_event_ledger_section(
    *,
    profile: str,
    projection: Projection,
    start_day: date,
    model_id: str,
    github_repos: list[str] | None = None,
    chart_window_end: date | None = None,
) -> None:
    """Show the ledger and add-event form when future events are enabled."""
    events = get_stored_events(profile)
    contrib_by_name = {
        ev.name: amt for ev, amt in projection.all_enabled_contributions()
    }

    section(
        "Event ledger",
        "Describe a planned change in plain language. The CostSense agent "
        "checks AWS and GitHub and only saves the event when $/day can be "
        "grounded in tool data — not guesses or industry averages.",
        kicker="Future events",
    )

    render_event_waterfall(projection, chart_window_end=chart_window_end)
    st.divider()

    if not events:
        st.info(
            "No events yet. Describe what will change below — the agent "
            "sizes impact only from AWS/GitHub evidence."
        )

    for i, ev in enumerate(events):
        _event_row(ev, i, contrib_by_name.get(ev.name), profile)

    with st.expander("Add an event", expanded=not events):
        _add_event_form(start_day, profile, model_id, github_repos or [])


def _event_row(
    ev: CostEvent,
    idx: int,
    contribution: Optional[float],
    profile: str,
) -> None:
    colour = _CATEGORY_COLOR.get(ev.category, C.MUTED)
    with st.container(border=True):
        head, amt, tog = st.columns([5, 1.3, 0.9])
        with head:
            safe_name = html.escape(ev.name)
            st.markdown(
                f"<span style='font-weight:650;word-break:break-word;"
                f"display:block;line-height:1.35;"
                f"{'' if ev.enabled else f'color:{C.FAINT};'}'>{safe_name}</span>",
                unsafe_allow_html=True,
            )
            chips = [
                f"<span style='background:{colour}1A;color:{colour};"
                f"padding:2px 9px;border-radius:999px;font-size:.75rem;"
                f"font-weight:600;'>{ev.shape_label}</span>",
                f"<span style='color:{C.MUTED};font-size:.75rem;"
                f"border:1px solid {C.HAIRLINE};padding:2px 9px;"
                f"border-radius:999px;'>confidence {ev.confidence:.0f}%</span>",
            ]
            if ev.source != "manual":
                chips.append(
                    f"<span style='color:{C.MUTED};font-size:.75rem;"
                    f"border:1px solid {C.HAIRLINE};padding:2px 9px;"
                    f"border-radius:999px;'>{ev.source.replace('_', ' ')}</span>"
                )
            st.markdown(" ".join(chips), unsafe_allow_html=True)
            if ev.note:
                st.markdown(
                    f"<span style='color:{C.MUTED};font-size:.83rem;"
                    f"word-break:break-word;display:block;line-height:1.35;'>"
                    f"{html.escape(ev.note)}</span>",
                    unsafe_allow_html=True,
                )
        with amt:
            if not ev.enabled:
                txt, col = "excluded", C.FAINT
            elif contribution is None:
                txt, col = "outside window", C.FAINT
            elif abs(contribution) <= 0.005:
                txt, col = f"starts {ev.start_date:%d %b}", C.MUTED
            else:
                txt = f"{'+' if contribution >= 0 else '−'}{money(abs(contribution))}"
                col = C.BAD if contribution >= 0 else C.GOOD
            st.markdown(
                f"<div style='text-align:right;padding-top:4px;' class='cs-num'>"
                f"<span style='color:{col};font-weight:680;'>{txt}</span></div>",
                unsafe_allow_html=True,
            )
        with tog:
            new_state = st.checkbox(
                "On", value=ev.enabled, key=f"fc_en_{profile}_{idx}",
                label_visibility="collapsed",
            )
            if new_state != ev.enabled:
                events = get_stored_events(profile)
                events[idx].enabled = new_state
                store_events(profile, events)
                st.rerun()


def _impact_summary(result: EventValidationResult) -> str:
    if result.effect == Effect.MULTIPLIER.value:
        return f"{result.multiplier_pct:+.0f}% on baseline"
    if result.amount_daily is not None:
        daily = money(abs(result.amount_daily))
        sign = "+" if (result.amount_daily or 0) >= 0 else "−"
        return f"{sign}{daily}/day"
    return "—"


def _render_validation_result(result: EventValidationResult) -> None:
    if result.error:
        callout(f"Validation failed: {result.error}", tone="error")
        if result.note and result.note != result.error:
            st.caption(result.note)
    elif not result.accepted:
        callout(
            result.note
            or "Event was not accepted — add more detail (services, scale, "
            "regions, or repo paths) so the agent can size impact from data.",
            tone="warning",
        )
    else:
        cat_label = CATEGORIES.get(result.category, result.category)
        callout(
            f"Accepted as **{result.standardized_name}** · "
            f"{result.effect} · {cat_label} · "
            f"{_impact_summary(result)} · "
            f"starts {result.start_date:%d %b %Y} · "
            f"confidence {result.confidence:.0f}%. "
            f"{result.note or ''}",
            tone="success",
        )
    if result.tool_calls:
        with st.expander(
            f"Agent checks ({len(result.tool_calls)} tool call(s))",
            expanded=False,
        ):
            for tc in result.tool_calls:
                st.markdown(f"**`{tc.name}`** — `{tc.output_summary}`")


def _add_event_form(
    start_day: date,
    profile: str,
    model_id: str,
    github_repos: list[str],
) -> None:
    if not github_repos:
        callout(
            "Select **Repos** in Controls above so the agent can read your "
            "codebase when validating events.",
            tone="info",
        )
    else:
        st.caption(
            f"Validation scope: {', '.join(github_repos[:4])}"
            + (f" + {len(github_repos) - 4} more" if len(github_repos) > 4 else "")
        )
    description = st.text_area(
        "What is changing?",
        key="fc_new_desc",
        placeholder=(
            "e.g. Three enterprise orgs onboarding in August, each adding "
            "roughly 200 DynamoDB tables and higher Lambda traffic"
        ),
        help="Plain language only — $/day must come from AWS/GitHub tool data.",
    )
    expected_start = st.date_input(
        "Expected around (optional hint)",
        value=start_day + timedelta(days=14),
        key="fc_new_start_hint",
        help="Optional — the agent may adjust the start date based on evidence.",
    )
    use_hint = st.checkbox(
        "Include expected date as a hint to the agent",
        value=True,
        key="fc_new_use_hint",
    )

    if st.button(
        "Validate and add",
        type="primary",
        key="fc_new_add",
        help="Agent checks AWS/GitHub and sets shape, cost, and timing.",
    ):
        if not description.strip():
            st.warning("Describe what is changing.")
            return

        draft = draft_from_user_input(
            description=description,
            expected_start=expected_start if use_hint else None,
            forecast_horizon_start=start_day,
            github_repos=github_repos,
        )

        with st.spinner(
            "CostSense is checking AWS and GitHub and sizing the event… "
            "this can take a minute.",
        ):
            try:
                result = validate_forecast_event(
                    profile=profile,
                    model_id=model_id,
                    draft=draft,
                )
            except Exception as e:  # noqa: BLE001
                callout(f"Validation crashed: {e}", tone="error")
                st.code(traceback.format_exc())
                return

        _render_validation_result(result)
        if result.error or not result.accepted:
            return

        event = result.to_cost_event()
        if event is None:
            callout("Agent accepted the event but returned incomplete fields.", tone="error")
            return

        events = get_stored_events(profile)
        events.append(event)
        store_events(profile, events)
        st.rerun()
