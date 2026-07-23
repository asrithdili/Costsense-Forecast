"""Streamlit UI for the event-based future forecast panel."""
from __future__ import annotations

import html
import traceback
from datetime import date, timedelta
from typing import List, Optional, Sequence

import plotly.graph_objects as go
import streamlit as st

from src.dashboard.costsense_theme import C, callout, metric, money, plotly_layout
from src.forecast import baselines as B
from src.forecast.adapters import (
    events_from_anomaly_actions,
    events_from_priced_open_prs,
    events_from_pr_impacts,
    find_anomaly_reports_for_profile,
    merge_events,
)
from src.forecast.event_store import (
    get_stored_events,
    set_import_flash,
    store_events,
)
from src.forecast.events import CATEGORIES, CostEvent, Effect
from src.forecast.scenario import Projection
from src.pr_scanner.profile_repo_match import match_repos
from src.pr_scanner.repos import (
    gh_login,
    gh_orgs,
    recent_base_branches,
    repo_default_branch,
    repos_with_user_prs,
)
from src.pr_scanner.scan import scan_and_price

_MAX_OPEN_PRS_DEEP_ANALYSIS = 3


HORIZONS = {"30 days": 30, "60 days": 60, "90 days": 90, "180 days": 180}
SCENARIOS = ["Expected", "Best case", "Worst case"]


def _forecast_total_label(total: float, baseline: float, contrib: float) -> str:
    """Show event deltas that K-abbreviation would hide (e.g. $20K vs $20,040)."""
    if abs(contrib) > 0.5 and money(total) == money(baseline):
        sign = "+" if contrib >= 0 else "−"
        return f"{money(baseline)} ({sign}{money(abs(contrib))})"
    return money(total)


def _waterfall_bar_label(value: float) -> str:
    """Comma-separated dollars so small event bars stay readable."""
    if abs(value) >= 1_000_000:
        return money(value)
    return f"${value:,.0f}"


_CATEGORY_COLOR = {
    "demand": C.INFO,
    "release": C.SEV["Medium"],
    "optimization": C.GOOD,
    "commitment": C.BRAND,
    "pricing": C.SEV["High"],
}


def _save_import_result(
    profile: str,
    *,
    added: int,
    incoming_count: int,
    scanned: int,
    priced: int,
    label: str,
    added_names: list[str],
) -> None:
    skipped = incoming_count - added
    if added:
        names = ", ".join(added_names[:5])
        extra = f" and {added - 5} more" if added > 5 else ""
        msg = (
            f"Added {added} {label} event(s): {names}{extra}. "
            f"({skipped} duplicate(s) skipped.)"
        )
        tone = "success"
    elif incoming_count:
        msg = (
            f"Found {incoming_count} {label} event(s) but all were already "
            "in the ledger."
        )
        tone = "info"
    else:
        msg = (
            f"No {label} events created — scanned {scanned}, priced {priced}, "
            "but none had a material $/day delta. Try hybrid/llm analyzer or "
            "a different repo/lookback."
        )
        tone = "warning"
    set_import_flash(profile, msg, tone)


def render_controls(
    *,
    budget_default: float,
    cost_per_unit_day: float,
    unit_label: str,
) -> tuple[int, str, str, float, Optional[List[float]]]:
    c1, c2, c3 = st.columns([1, 1.4, 1.2])
    with c1:
        horizon = HORIZONS[
            st.selectbox("Horizon", list(HORIZONS.keys()), index=2, key="fc_horizon")
        ]
    with c2:
        method_label = st.selectbox(
            "Baseline method",
            list(B.BASELINE_METHODS.keys()),
            index=2,
            key="fc_method",
            help="How existing spend is carried forward before events are applied.",
        )
        method = B.BASELINE_METHODS[method_label]
    with c3:
        scenario = st.selectbox(
            "Scenario",
            SCENARIOS,
            key="fc_scenario",
            help="Expected weights each event by confidence. Worst case "
            "assumes every cost increase lands and no saving does.",
        )

    units: Optional[List[float]] = None
    if method == "driver":
        d1, d2, d3 = st.columns(3)
        with d1:
            start_units = st.number_input(
                f"{unit_label.title()} today",
                0,
                100_000,
                750,
                step=50,
                key="fc_units_start",
            )
        with d2:
            end_units = st.number_input(
                f"{unit_label.title()} at horizon",
                0,
                100_000,
                1_400,
                step=50,
                key="fc_units_end",
            )
        with d3:
            ramp = st.number_input(
                "Ramp (days)", 1, 365, 45, step=5, key="fc_units_ramp",
            )
        units = B.ramp_units(start_units, end_units, horizon, ramp)
        st.caption(
            f"Driver-based: **{money(cost_per_unit_day, 2)}** per "
            f"{unit_label[:-1] if unit_label.endswith('s') else unit_label} "
            f"per day × the {unit_label} plan above."
        )

    with st.expander("Budget", expanded=False):
        budget = st.number_input(
            f"Budget for the {horizon}-day horizon ($)",
            0,
            100_000_000,
            int(budget_default),
            step=10_000,
            key="fc_budget",
        )

    return horizon, method, scenario, float(budget), units


def render_answer_band(proj: Projection, budget: float, scenario: str) -> None:
    headline = {
        "Expected": proj.total_expected,
        "Best case": proj.total_best,
        "Worst case": proj.total_worst,
    }[scenario]

    a, b, c = st.columns(3)
    with a:
        contrib = proj.event_contribution
        metric(
            f"Projected {proj.horizon_days}d spend",
            _forecast_total_label(headline, proj.total_baseline, contrib),
            delta=f"range {money(proj.total_best)} – {money(proj.total_worst)}",
        )
    with b:
        share = (contrib / proj.total_expected * 100) if proj.total_expected else 0
        metric(
            "From future events",
            f"{'+' if contrib >= 0 else '−'}{money(abs(contrib))}",
            delta=f"{abs(share):.1f}% of the forecast",
            good=contrib < 0,
        )
    with c:
        crossing = proj.budget_crossing(budget) if budget else None
        if crossing:
            days_out = (crossing - proj.dates[0]).days
            metric(
                "Budget crossed",
                f"{crossing:%d %b}",
                delta=f"day {days_out} of {proj.horizon_days}",
                good=False,
            )
        else:
            headroom = budget - proj.total_expected
            metric(
                "Budget",
                "Not crossed",
                delta=f"{money(headroom)} headroom",
                good=True,
            )


def render_chart(
    hist_dates: Sequence[date],
    hist_values: Sequence[float],
    proj: Projection,
    scenario: str,
    budget: float,
) -> None:
    st.markdown("##### Projection")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(hist_dates),
        y=list(hist_values),
        name="Actual",
        mode="lines",
        line=dict(color=C.INK, width=1.6),
        hovertemplate="%{x|%d %b}<br>$%{y:,.0f}<extra>Actual</extra>",
    ))

    fig.add_trace(go.Scatter(
        x=proj.dates + proj.dates[::-1],
        y=proj.worst + proj.best[::-1],
        fill="toself",
        fillcolor="rgba(12,124,116,0.13)",
        line=dict(width=0),
        name="Best–worst range",
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=proj.dates,
        y=proj.baseline,
        name="Baseline (no events)",
        mode="lines",
        line=dict(color=C.FAINT, width=1.2, dash="dot"),
        hovertemplate="%{x|%d %b}<br>$%{y:,.0f}<extra>Baseline</extra>",
    ))

    key = {
        "Expected": proj.expected,
        "Best case": proj.best,
        "Worst case": proj.worst,
    }[scenario]
    fig.add_trace(go.Scatter(
        x=proj.dates,
        y=key,
        name=scenario,
        mode="lines",
        line=dict(color=C.BRAND, width=2.4, dash="dash"),
        hovertemplate="%{x|%d %b}<br>$%{y:,.0f}<extra>" + scenario + "</extra>",
    ))

    for ev in proj.events:
        if (
            not ev.enabled
            or ev.start_date < proj.dates[0]
            or ev.start_date > proj.dates[-1]
        ):
            continue
        colour = _CATEGORY_COLOR.get(ev.category, C.MUTED)
        fig.add_vline(
            x=ev.start_date,
            line=dict(color=colour, width=1, dash="dot"),
        )
        fig.add_annotation(
            x=ev.start_date,
            y=1.0,
            yref="paper",
            text=ev.name.split("·")[0].strip()[:22],
            showarrow=False,
            yanchor="bottom",
            font=dict(size=10, color=colour),
            textangle=0,
        )

    if budget:
        daily_budget = budget / proj.horizon_days
        fig.add_hline(
            y=daily_budget,
            line=dict(color=C.BAD, width=1, dash="dashdot"),
            annotation_text="Budget pace",
            annotation_position="right",
            annotation_font=dict(size=10, color=C.BAD),
        )

    lay = plotly_layout(height=380)
    lay["yaxis_title"] = "Daily spend (USD)"
    lay["margin"] = dict(l=16, r=24, t=44, b=16)
    fig.update_layout(**lay)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Dotted grey is the baseline with every event switched off. The gap "
        "between it and the dashed line is what the events add."
    )


def render_waterfall(proj: Projection) -> None:
    st.markdown("##### What drives the forecast")
    contribs = proj.contributions()
    if not contribs:
        st.info("No enabled events — the forecast is pure baseline.")
        return

    contrib_sum = sum(amt for _, amt in contribs)
    labels = (
        ["Baseline"]
        + [ev.name.split("·")[0].strip()[:28] for ev, _ in contribs]
        + ["Projected"]
    )
    measures = ["absolute"] + ["relative"] * len(contribs) + ["total"]
    values = [proj.total_baseline] + [amt for _, amt in contribs] + [0]
    hover_values = (
        [proj.total_baseline]
        + [amt for _, amt in contribs]
        + [proj.total_expected]
    )

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measures,
        x=labels,
        y=values,
        customdata=hover_values,
        text=[
            _waterfall_bar_label(v) if m != "total"
            else _forecast_total_label(
                proj.total_expected, proj.total_baseline, contrib_sum,
            )
            for v, m in zip(values, measures)
        ],
        textposition="outside",
        connector=dict(line=dict(color=C.HAIRLINE)),
        increasing=dict(marker=dict(color=C.BAD)),
        decreasing=dict(marker=dict(color=C.GOOD)),
        totals=dict(marker=dict(color=C.BRAND)),
        hovertemplate="<b>%{x}</b><br>$%{customdata:,.0f}<extra></extra>",
    ))
    lay = plotly_layout(height=330)
    lay["showlegend"] = False
    lay["yaxis_title"] = f"{proj.horizon_days}-day spend (USD)"
    lay["yaxis_tickformat"] = "$,.0f"
    ymax = max(proj.total_baseline, proj.total_expected)
    if ymax:
        pad = max(abs(contrib_sum) * 2, ymax * 0.02, 500)
        lay["yaxis_range"] = [0, ymax + pad]
    lay["margin"] = dict(l=16, r=24, t=30, b=70)
    fig.update_layout(**lay)
    st.plotly_chart(fig, use_container_width=True)
    if abs(contrib_sum) > 0.5:
        sign = "+" if contrib_sum >= 0 else "−"
        st.caption(
            f"Baseline {_waterfall_bar_label(proj.total_baseline)} "
            f"{sign} {_waterfall_bar_label(abs(contrib_sum))} from "
            f"{len(contribs)} event(s) = "
            f"{_waterfall_bar_label(proj.total_expected)} projected."
        )


def render_pr_import_section(
    *,
    account_id: str,
    active_profile: str,
    widget_ver: int,
) -> None:
    """Import dated events from GitHub PR scans (explicit button — no auto-scan)."""
    st.markdown("##### Import from CostSense")
    st.caption(
        "Pull priced PRs from Scan → PR and open-PR analysis. Imports are "
        "deduplicated; toggle or remove events in the ledger below."
    )

    orgs: list[str] = []
    try:
        gh_login()
        orgs = list(gh_orgs())
    except Exception:  # noqa: BLE001
        orgs = []

    default_org_idx = 0
    gh_org_default = "DiligentCorp"
    if orgs:
        for i, o in enumerate(orgs):
            if o == "DiligentCorp":
                default_org_idx = i
                break
        gh_org_default = orgs[default_org_idx]

    with st.expander("GitHub PR import", expanded=False):
        r1c1, r1c2, r1c3, r1c4 = st.columns(
            [2, 4, 2, 2], gap="medium", vertical_alignment="bottom",
        )
        with r1c1:
            if orgs:
                gh_org = st.selectbox(
                    "GitHub org",
                    orgs,
                    index=(
                        orgs.index(gh_org_default)
                        if gh_org_default in orgs
                        else default_org_idx
                    ),
                    key="fc_gh_org",
                )
            else:
                gh_org = st.text_input(
                    "GitHub org",
                    value=gh_org_default,
                    key="fc_gh_org",
                )

        short_names: list[str] = []
        default_repos: list[str] = []
        try:
            suggested_full = list(repos_with_user_prs(gh_org)) if gh_org else []
            short_names = [r.split("/", 1)[-1] for r in suggested_full]
            default_repos = match_repos(active_profile, short_names) or short_names
        except Exception:  # noqa: BLE001
            pass

        with r1c2:
            picked_short = st.multiselect(
                "Repos",
                options=short_names,
                default=default_repos,
                key=f"fc_repos_v{widget_ver}",
            )
        selected_repos = [f"{gh_org}/{n}" for n in picked_short] if gh_org else []

        branch_choices: list[str] = []
        if selected_repos:
            for repo in selected_repos:
                try:
                    for b in recent_base_branches(repo):
                        if b not in branch_choices:
                            branch_choices.append(b)
                    default = repo_default_branch(repo)
                    if default not in branch_choices:
                        branch_choices.append(default)
                except Exception:  # noqa: BLE001
                    continue

        with r1c3:
            if branch_choices:
                base_branch = st.selectbox(
                    "Base branch", branch_choices, key="fc_base_branch",
                )
            else:
                base_branch = st.text_input(
                    "Base branch",
                    placeholder="e.g. main",
                    key="fc_base_branch_text",
                ) or None
        with r1c4:
            pr_lookback = st.slider(
                "PR lookback (d)", 3, 30, 14, step=1, key="fc_pr_lookback",
            )

        r2c1, r2c2, r2c3 = st.columns(
            [2, 2, 2], gap="medium", vertical_alignment="bottom",
        )
        with r2c1:
            analyzer = st.selectbox(
                "PR analyzer",
                options=["hybrid", "llm", "regex"],
                index=0,
                key="fc_analyzer",
            )
        with r2c2:
            llm_model = st.selectbox(
                "Bedrock model",
                options=[
                    "us.anthropic.claude-sonnet-4-6",
                    "anthropic.claude-3-haiku-20240307-v1:0",
                ],
                index=0,
                key="fc_pr_llm",
                disabled=(analyzer == "regex"),
            )
        with r2c3:
            st.caption(
                f"Open PR deep-analysis capped at {_MAX_OPEN_PRS_DEEP_ANALYSIS} "
                "most merge-ready PRs."
            )

        b1, b2, _ = st.columns([2, 2, 6], gap="medium")
        with b1:
            do_open = st.button(
                "Import open PRs",
                key="fc_import_open",
                disabled=not selected_repos,
                help="Deep-analyze open PRs and add expected future deltas.",
            )
        with b2:
            do_merged = st.button(
                "Import merged PRs",
                key="fc_import_merged",
                disabled=not (selected_repos and base_branch),
                help="Scan recently merged PRs for cost impact steps.",
            )

        if do_open and selected_repos:
            with st.spinner(
                f"Deep-analyzing top {_MAX_OPEN_PRS_DEEP_ANALYSIS} open PRs…",
            ):
                try:
                    from src.pr_scanner.open_prs import (
                        analyze_open_prs,
                        list_open_prs_many,
                    )

                    open_prs = list_open_prs_many(selected_repos)
                    priced = analyze_open_prs(
                        open_prs,
                        profile=active_profile,
                        llm_model=llm_model,
                        max_prs=_MAX_OPEN_PRS_DEEP_ANALYSIS,
                    )
                    incoming = events_from_priced_open_prs(priced)
                    existing = get_stored_events(active_profile)
                    before_ids = {e.external_id for e in existing if e.external_id}
                    merged, added = merge_events(existing, incoming)
                    store_events(active_profile, merged)
                    added_names = [
                        ev.name for ev in incoming
                        if not ev.external_id or ev.external_id not in before_ids
                    ]
                    _save_import_result(
                        active_profile,
                        added=added,
                        incoming_count=len(incoming),
                        scanned=len(open_prs),
                        priced=len(priced),
                        label="open PR",
                        added_names=added_names,
                    )
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    callout(f"Open PR import failed: {e}", tone="error")
                    st.code(traceback.format_exc())

        if do_merged and selected_repos and base_branch:
            with st.spinner("Scanning merged PRs…"):
                try:
                    impacts, _total = scan_and_price(
                        selected_repos,
                        base=base_branch,
                        lookback_days=pr_lookback,
                        aws_profile=active_profile,
                        analyzer=analyzer,
                        llm_model=llm_model if analyzer != "regex" else None,
                    )
                    incoming = events_from_pr_impacts(impacts)
                    existing = get_stored_events(active_profile)
                    before_ids = {e.external_id for e in existing if e.external_id}
                    merged, added = merge_events(existing, incoming)
                    store_events(active_profile, merged)
                    added_names = [
                        ev.name for ev in incoming
                        if not ev.external_id or ev.external_id not in before_ids
                    ]
                    _save_import_result(
                        active_profile,
                        added=added,
                        incoming_count=len(incoming),
                        scanned=len(impacts),
                        priced=sum(
                            1 for i in impacts if abs(i.est_daily_delta_usd) >= 0.01
                        ),
                        label="merged PR",
                        added_names=added_names,
                    )
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    callout(f"Merged PR import failed: {e}", tone="error")
                    st.code(traceback.format_exc())


def render_anomaly_import_section(
    *,
    active_profile: str,
) -> None:
    """Import savings events from a cached Anomalies scan for this profile."""
    reports = find_anomaly_reports_for_profile(active_profile)

    st.markdown("##### Import from Anomalies")
    st.caption(
        "Pull recommendations from your latest Anomalies scan for this account. "
        "Imports are deduplicated in the ledger below."
    )

    if not reports:
        st.info(
            "No Anomalies scan found for this account. Open **Anomalies**, "
            "pick the same AWS profile, run **Analyze**, then return here. "
            "You can also use **Add to future forecast** on each recommendation "
            "on the Anomalies page."
        )
        return

    labels = []
    for key, report in reports:
        repos_part = key.split("::", 3)[-1] if key.count("::") >= 3 else ""
        repo_count = len([r for r in repos_part.split(",") if r.strip()]) if repos_part else 0
        labels.append(
            f"{len(report.actions)} action(s)"
            + (f" · {repo_count} repo(s)" if repo_count else "")
        )

    with st.expander("Anomaly recommendations", expanded=False):
        pick_idx = st.selectbox(
            "Cached scan",
            range(len(labels)),
            format_func=lambda i: labels[i],
            key=f"fc_anom_pick::{active_profile}",
        )
        report_key, report = reports[pick_idx]

        apply_date = st.date_input(
            "Expected apply date",
            value=date.today() + timedelta(days=14),
            key=f"fc_anom_apply::{active_profile}::{pick_idx}",
        )

        action_labels = [
            (
                f"#{i + 1} · save ${a.est_daily_savings_usd:,.2f}/day · "
                f"{(a.issue or a.recommendation or 'action')[:40]}"
            )
            for i, a in enumerate(report.actions)
        ]
        picked = st.multiselect(
            "Actions to import",
            options=list(range(len(report.actions))),
            default=list(range(len(report.actions))),
            format_func=lambda i: action_labels[i],
            key=f"fc_anom_actions::{active_profile}::{pick_idx}",
        )

        if st.button(
            "Import selected recommendations",
            key=f"fc_anom_import::{active_profile}::{pick_idx}",
            disabled=not picked,
        ):
            incoming = events_from_anomaly_actions(
                report.actions,
                profile=active_profile,
                report_key=report_key,
                action_indices=picked,
                expected_apply=apply_date,
            )
            existing = get_stored_events(active_profile)
            before_ids = {e.external_id for e in existing if e.external_id}
            merged, added = merge_events(existing, incoming)
            store_events(active_profile, merged)
            added_names = [
                ev.name for ev in incoming
                if not ev.external_id or ev.external_id not in before_ids
            ]
            _save_import_result(
                active_profile,
                added=added,
                incoming_count=len(incoming),
                scanned=len(report.actions),
                priced=sum(
                    1 for a in report.actions if a.est_daily_savings_usd >= 0.01
                ),
                label="anomaly",
                added_names=added_names,
            )
            st.rerun()


def render_ledger(
    proj: Projection,
    profile: str,
    start_day: date,
) -> None:
    events = get_stored_events(profile)
    st.markdown("##### Event ledger")
    st.caption(
        "Every assumption behind the forecast, named and dated. Uncheck one "
        "to see the projection without it."
    )

    if not events:
        st.info(
            "No events yet. Import PRs or anomaly recommendations above, add "
            "one manually below, or queue from Anomalies / PR Predictor."
        )

    contrib_by_name = {ev.name: amt for ev, amt in proj.contributions()}

    for i, ev in enumerate(events):
        _event_row(ev, i, contrib_by_name.get(ev.name), profile)

    with st.expander("Add an event"):
        _add_event_form(start_day, profile)


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
                txt, col = "outside horizon", C.FAINT
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


def _add_event_form(start_day: date, profile: str) -> None:
    name = st.text_input(
        "Name",
        key="fc_new_name",
        placeholder="Region expansion — eu-central-1",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        shape = st.selectbox(
            "Shape", [e.value for e in Effect], key="fc_new_shape",
        )
    with c2:
        category = st.selectbox(
            "Category",
            list(CATEGORIES.keys()),
            format_func=lambda k: CATEGORIES[k],
            key="fc_new_cat",
        )
    with c3:
        confidence = st.slider(
            "Confidence %", 0, 100, 75, step=5, key="fc_new_conf",
        )

    d1, d2 = st.columns(2)
    with d1:
        start = st.date_input(
            "Starts",
            value=start_day + timedelta(days=14),
            key="fc_new_start",
        )
    with d2:
        end = st.date_input(
            "Ends (pulse only)",
            value=start_day + timedelta(days=44),
            key="fc_new_end",
        )

    e1, e2 = st.columns(2)
    with e1:
        if shape == Effect.MULTIPLIER.value:
            mult = st.number_input(
                "Multiplier %", -90.0, 500.0, 25.0, step=5.0, key="fc_new_mult",
            )
            monthly = 0.0
        else:
            monthly = st.number_input(
                "Monthly impact ($, negative = saving)",
                -1_000_000.0,
                1_000_000.0,
                12_000.0,
                step=500.0,
                key="fc_new_amt",
            )
            mult = 0.0
    with e2:
        ramp_days = st.number_input(
            "Ramp days (ramp only)", 0, 365, 30, step=5, key="fc_new_ramp",
        )

    if st.button("Add to forecast", type="primary", key="fc_new_add"):
        if not name.strip():
            st.warning("Give the event a name.")
            return
        events = get_stored_events(profile)
        events.append(CostEvent(
            name=name.strip(),
            start_date=start,
            effect=Effect(shape),
            category=category,
            amount_daily=monthly / 30.0,
            end_date=end if shape == Effect.PULSE.value else None,
            ramp_days=int(ramp_days) if shape == Effect.RAMP.value else 0,
            multiplier_pct=mult,
            confidence=float(confidence),
            source="manual",
        ))
        store_events(profile, events)
        st.rerun()
