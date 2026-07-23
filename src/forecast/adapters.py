"""Convert CostSense PR and anomaly outputs into forecast events."""
from __future__ import annotations

from datetime import date
from typing import Iterable, Sequence

from src.ai_agent.agent import AgentVerdict
from src.ai_agent.anomaly_agent import Action, AnomalyReport
from src.forecast.events import CostEvent, Effect
from src.pr_scanner.open_prs import PricedOpenPr
from src.pr_scanner.scan import PrImpact

CONFIDENCE_FROM_STRING = {
    "low": 50.0,
    "medium": 75.0,
    "high": 90.0,
}

CONFIDENCE_FROM_ANALYZER = {
    "hybrid": 85.0,
    "llm": 80.0,
    "regex": 60.0,
}


def confidence_from_string(conf: str | None) -> float:
    return CONFIDENCE_FROM_STRING.get((conf or "medium").lower(), 75.0)


def confidence_from_analyzer(analyzer: str | None) -> float:
    return CONFIDENCE_FROM_ANALYZER.get((analyzer or "hybrid").lower(), 75.0)


def event_from_priced_open_pr(p: PricedOpenPr) -> CostEvent | None:
    """Open PR priced by merge probability → future step event."""
    if abs(p.est_daily_delta_usd) < 0.01:
        return None
    try:
        merge_day = date.fromisoformat(p.expected_merge_day)
    except ValueError:
        return None
    pr = p.open_pr
    return CostEvent(
        name=f"Open PR · {pr.repo}#{pr.number} · {pr.title[:48]}",
        start_date=merge_day,
        effect=Effect.STEP,
        category="release" if p.est_daily_delta_usd > 0 else "optimization",
        amount_daily=p.est_daily_delta_usd,
        confidence=max(1.0, min(100.0, p.merge_probability * 100.0)),
        source="pr_predictor",
        external_id=f"open_pr:{pr.repo}#{pr.number}",
        note=p.llm_summary or (
            f"Est. ${p.est_daily_delta_usd:+,.2f}/day if merged · "
            f"{p.merge_probability:.0%} merge probability"
        ),
    )


def event_from_pr_impact(imp: PrImpact) -> CostEvent | None:
    """Merged CostSense PR scan result → step from merge date."""
    if abs(imp.est_daily_delta_usd) < 0.01:
        return None
    try:
        merge_day = date.fromisoformat(imp.merged_at[:10])
    except ValueError:
        return None
    category = "optimization" if imp.est_daily_delta_usd < 0 else "release"
    return CostEvent(
        name=f"Merged PR · {imp.repo}#{imp.pr_number} · {imp.pr_title[:48]}",
        start_date=merge_day,
        effect=Effect.STEP,
        category=category,
        amount_daily=imp.est_daily_delta_usd,
        confidence=confidence_from_analyzer(imp.analyzer),
        source="costsense_pr",
        external_id=f"merged_pr:{imp.repo}#{imp.pr_number}",
        note=imp.llm_summary or f"Merged {imp.merged_at[:10]} via {imp.analyzer}",
    )


def event_from_pr_predictor(
    pr_ref: str,
    verdict: AgentVerdict,
    *,
    expected_deploy: date,
    ramp_days: int = 0,
) -> CostEvent | None:
    """PR Predictor verdict → dated future step or ramp."""
    lo = verdict.est_daily_delta_low_usd
    hi = verdict.est_daily_delta_high_usd
    if lo is not None and hi is not None and abs(hi - lo) > 0.01:
        delta_daily = (lo + hi) / 2.0
    else:
        delta_daily = verdict.est_daily_delta_usd
    if abs(delta_daily) < 0.01:
        return None
    category = "optimization" if delta_daily < 0 else "release"
    return CostEvent(
        name=f"Predicted impact · {pr_ref}",
        start_date=expected_deploy,
        effect=Effect.RAMP if ramp_days else Effect.STEP,
        category=category,
        amount_daily=delta_daily,
        ramp_days=ramp_days,
        confidence=confidence_from_string(verdict.confidence),
        source="pr_predictor",
        external_id=f"pr_predictor:{pr_ref}",
        note=verdict.verdict or verdict.detail[:200],
    )


def events_from_priced_open_prs(
    priced: Iterable[PricedOpenPr],
) -> list[CostEvent]:
    out: list[CostEvent] = []
    for p in priced:
        ev = event_from_priced_open_pr(p)
        if ev is not None:
            out.append(ev)
    return out


def events_from_pr_impacts(impacts: Iterable[PrImpact]) -> list[CostEvent]:
    out: list[CostEvent] = []
    for imp in impacts:
        ev = event_from_pr_impact(imp)
        if ev is not None:
            out.append(ev)
    return out


def event_from_anomaly_action(
    action: Action,
    *,
    account_id: str,
    report_key: str,
    action_idx: int,
    expected_apply: date,
    ramp_days: int = 0,
) -> CostEvent | None:
    """Anomaly recommendation → dated savings step in the forecast ledger."""
    savings = action.est_daily_savings_usd
    if savings < 0.01:
        return None
    label = (action.issue or action.recommendation or "Cost action").strip()
    return CostEvent(
        name=f"Anomaly · {label[:48]}",
        start_date=expected_apply,
        effect=Effect.RAMP if ramp_days else Effect.STEP,
        category="optimization",
        amount_daily=-savings,
        ramp_days=ramp_days,
        confidence=confidence_from_string(action.confidence),
        source="anomalies",
        external_id=f"anomaly:{account_id}:{report_key}:{action_idx}",
        note=(action.recommendation or action.reason or "")[:200],
    )


def events_from_anomaly_actions(
    actions: Sequence[Action],
    *,
    account_id: str,
    report_key: str,
    action_indices: Iterable[int],
    expected_apply: date,
    ramp_days: int = 0,
) -> list[CostEvent]:
    out: list[CostEvent] = []
    actions_list = list(actions)
    for idx in action_indices:
        if idx < 0 or idx >= len(actions_list):
            continue
        ev = event_from_anomaly_action(
            actions_list[idx],
            account_id=account_id,
            report_key=report_key,
            action_idx=idx,
            expected_apply=expected_apply,
            ramp_days=ramp_days,
        )
        if ev is not None:
            out.append(ev)
    return out


def find_anomaly_reports_for_profile(profile: str) -> list[tuple[str, AnomalyReport]]:
    """Cached anomaly scans in session state for ``profile``."""
    import streamlit as st

    needle = f"::{profile}::"
    out: list[tuple[str, AnomalyReport]] = []
    for key, value in st.session_state.items():
        if not isinstance(key, str) or not key.startswith("anom::"):
            continue
        if needle not in key or key.endswith("::aws") or key.endswith("::repo"):
            continue
        if not isinstance(value, AnomalyReport):
            continue
        if value.error or not value.actions:
            continue
        out.append((key, value))
    out.sort(key=lambda pair: pair[0], reverse=True)
    return out


def merge_events(
    existing: Sequence[CostEvent],
    incoming: Iterable[CostEvent],
) -> tuple[list[CostEvent], int]:
    """Append events, skipping duplicates by ``external_id`` when set."""
    merged = list(existing)
    seen = {e.external_id for e in merged if e.external_id}
    added = 0
    for ev in incoming:
        if ev.external_id and ev.external_id in seen:
            continue
        if ev.external_id:
            seen.add(ev.external_id)
        merged.append(ev)
        added += 1
    return merged, added


def pending_import_key(account_id: str) -> str:
    return f"fc_pending_import::{account_id}"


def queue_pending_event(account_id: str, event: CostEvent) -> None:
    import streamlit as st

    key = pending_import_key(account_id)
    st.session_state.setdefault(key, []).append(event.to_dict())


def drain_pending_events(account_id: str) -> tuple[list[CostEvent], int]:
    """Merge queued cross-page events into the account event list."""
    import streamlit as st

    from src.forecast.event_store import get_stored_events, store_events

    key = pending_import_key(account_id)
    pending_raw = st.session_state.pop(key, [])
    if not pending_raw:
        return get_stored_events(account_id), 0
    pending = [
        CostEvent.from_dict(p) if isinstance(p, dict) else p
        for p in pending_raw
    ]
    events = get_stored_events(account_id)
    merged, added = merge_events(events, pending)
    store_events(account_id, merged)
    return merged, added
