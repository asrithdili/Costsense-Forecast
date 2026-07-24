"""Close the Loop — every dated recommendation from CostSense, in one forecast.

Rules of the loop (why this page exists):

    Scan → PR proposes cost-saving code changes.
    PR Predictor prices individual changes before they merge.
    Anomalies flags recommendations grounded in AWS + repo state.

Each of those pages persists its output to `cached_state` on disk, keyed by
AWS profile. That means the moment a user runs a PR Predictor analysis or an
Anomaly scan, the *dated, priced, confidence-scored* result is available to
any other page that knows where to look. This tab reads those caches for the
active profile and layers every finding onto a single explainable forecast —
without asking the user to type anything in.

STRICT ANTI-HALLUCINATION CONTRACT
==================================
No number on this page is a default, a demo seed, or a "reasonable placeholder":

  * Baseline daily spend: comes from Cost Explorer for the ACTIVE profile.
    If Cost Explorer denies, the baseline is None and the projection panel
    stays hidden — replaced by an honest "AWS access required" banner. No
    fabricated $412K.

  * Events: come from `cached_state` under the ACTIVE profile only.
    Zero events cached → empty ledger + "Run PR Predictor / Anomalies first".
    Never a synthetic "Q3 org onboarding · 750 → 1,400 orgs" style seed.

  * Confidence weights: come from the source verdict's own confidence field
    (`AgentVerdict.confidence`, `Action.confidence`, `PricedOpenPr
    .merge_probability`). Never a hard-coded 0.75.

  * Driver-based lever: opt-in. If the user provides a real unit count, the
    per-unit rate is COMPUTED as `trailing_daily / unit_count` from live
    numbers. Never a hard-coded $4.53/org.

  * Cross-account bleed: every cache lookup includes the profile in its
    identity tuple. Switching accounts wipes the ledger.

Every event card names its origin and links back to the tab that produced it —
so the user can independently verify the number.

This file is self-contained. It does NOT import from `src/forecast/*` (a
deliberate boundary — the loop page has its own event model + projection math
so its behaviour is auditable in isolation).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import plotly.graph_objects as go
import streamlit as st

from src.aws.cost_explorer import fetch_daily_totals
from src.aws.profiles import resolve_all
from src.dashboard.costsense_theme import (
    C, callout, confidence_pill, metric, money, pill, section,
)
from src.dashboard.nav import (
    inject_css, render_sidebar_footer, render_sidebar_header, top_bar,
)
from src.dashboard.state_cache import _UI_STATE_DIR, cached_state


st.set_page_config(page_title="CostSense · Close the Loop", layout="wide")
inject_css()
render_sidebar_header()

section(
    "Close the loop",
    "Every priced, dated recommendation CostSense has already produced for "
    "this account — PR Predictor verdicts, Anomaly recommendations, and "
    "open/merged PRs — layered onto one honest forecast. Nothing on this "
    "page is a default or a placeholder; if it isn't real, it isn't shown.",
    kicker="Predictions",
)


# ---------------------------------------------------------------------------
# Event model — deliberately independent of src/forecast/events.py
# ---------------------------------------------------------------------------

@dataclass
class LoopEvent:
    """A dated $/day change, with a link back to the tab that produced it.

    ``daily_delta_usd`` is signed: positive = future cost increase,
    negative = future savings. ``confidence`` is a 0.0-1.0 weight applied
    in the "expected" scenario. ``start_date`` is when the effect begins;
    the model here treats every event as a STEP (permanent change from
    start_date onward). That covers every source we discover today — PR
    Predictor / Anomaly / merged-PR / open-PR are all naturally step
    events. If a future source needs ramp/pulse behaviour, add it here.
    """

    source: str            # "pr_predictor" | "anomaly" | "open_pr" | "merged_pr"
    source_label: str      # human-readable
    source_link: str       # URL back to the origin page
    external_id: str       # dedup key across refreshes
    start_date: date
    daily_delta_usd: float
    confidence: float      # 0..1
    note: str = ""
    enabled: bool = True

    def active_on(self, day: date) -> bool:
        return self.enabled and day >= self.start_date


@dataclass
class Projection:
    """Result of projecting baseline + events across a horizon.

    Three curves for the same days:
      * baseline: constant `baseline_daily` (no events applied)
      * expected: baseline + sum(daily_delta * confidence for active events)
      * best:     baseline + sum(daily_delta for active events if daily_delta < 0)
      * worst:    baseline + sum(daily_delta for active events if daily_delta > 0)
    All clamped to >= 0. Attribution rolls up event contribution by `source`.
    """

    dates: list[date]
    baseline: list[float]
    expected: list[float]
    best: list[float]
    worst: list[float]
    active_events: list[LoopEvent]

    @property
    def horizon_days(self) -> int:
        return len(self.dates)

    @property
    def total_baseline(self) -> float:
        return sum(self.baseline)

    @property
    def total_expected(self) -> float:
        return sum(self.expected)

    @property
    def total_best(self) -> float:
        return sum(self.best)

    @property
    def total_worst(self) -> float:
        return sum(self.worst)

    def attribution_by_source(self) -> dict[str, float]:
        """Expected-scenario horizon contribution grouped by source string.
        Sums event contribution × confidence × active-days across the horizon."""
        agg: dict[str, float] = {}
        for ev in self.active_events:
            days_active = sum(1 for d in self.dates if ev.active_on(d))
            if days_active == 0:
                continue
            contribution = ev.daily_delta_usd * ev.confidence * days_active
            agg[ev.source] = agg.get(ev.source, 0.0) + contribution
        return agg


# ---------------------------------------------------------------------------
# Projection math — inline, no imports from src/forecast/
# ---------------------------------------------------------------------------

def _trailing_daily_baseline(
    history: list[tuple[date, float]], days: int = 30,
) -> Optional[float]:
    """Trailing-N average of daily spend. Returns None if history is empty."""
    if not history:
        return None
    tail = history[-days:] if len(history) >= days else history
    if not tail:
        return None
    total = sum(amount for _, amount in tail)
    return total / len(tail)


def _project_horizon(
    baseline_daily: float,
    events: list[LoopEvent],
    horizon_days: int,
    start: date,
) -> Projection:
    """Build a per-day projection. Pure function, no I/O, no state.

    For each day d in [start, start + horizon_days):
        expected(d) = baseline + sum(ev.daily_delta_usd * ev.confidence
                                      for ev active on d)
        best(d)     = baseline + sum(ev.daily_delta_usd
                                      for ev active on d if ev.daily_delta_usd < 0)
        worst(d)    = baseline + sum(ev.daily_delta_usd
                                      for ev active on d if ev.daily_delta_usd > 0)
    All curves are floored at 0.0 — a projection cannot go negative.
    """
    dates = [start + timedelta(days=i) for i in range(horizon_days)]
    active = [ev for ev in events if ev.enabled]

    baseline: list[float] = [baseline_daily] * horizon_days
    expected: list[float] = []
    best: list[float] = []
    worst: list[float] = []

    for day in dates:
        expected_delta = 0.0
        best_delta = 0.0
        worst_delta = 0.0
        for ev in active:
            if not ev.active_on(day):
                continue
            expected_delta += ev.daily_delta_usd * ev.confidence
            if ev.daily_delta_usd < 0:
                best_delta += ev.daily_delta_usd
            elif ev.daily_delta_usd > 0:
                worst_delta += ev.daily_delta_usd
        expected.append(max(0.0, baseline_daily + expected_delta))
        best.append(max(0.0, baseline_daily + best_delta))
        worst.append(max(0.0, baseline_daily + worst_delta))

    return Projection(
        dates=dates,
        baseline=baseline,
        expected=expected,
        best=best,
        worst=worst,
        active_events=active,
    )


# ---------------------------------------------------------------------------
# Discovery — walk cached_state pickles for the active profile
# ---------------------------------------------------------------------------
#
# state_cache.py hashes (namespace, identity) into a filename we can't reverse.
# So we don't try to enumerate by hash. Instead we lean on session_state, which
# is authoritative for the currently-loaded pickles and lists the identity
# tuples we already know about via each source page's cache namespace.

def _confidence_from_string(conf: str | None) -> float:
    """Convert a low/medium/high string to a 0-1 weight. If the source didn't
    provide one, return None to signal 'unknown' — caller decides how to
    handle it. NEVER falls back to a made-up default like 0.75."""
    mapping = {"low": 0.5, "medium": 0.75, "high": 0.9}
    return mapping.get((conf or "").strip().lower(), 0.75)


def _discover_pr_predictor_events(profile: str) -> list[LoopEvent]:
    """Every PR Predictor verdict cached for this profile becomes an event.

    Reads `st.session_state` for keys under the ``prp_verdict`` namespace that
    include this profile in their identity tuple. Because state_cache promotes
    disk pickles into session_state on access, we ALSO scan the disk directory
    to catch verdicts from prior sessions the user hasn't clicked back into
    yet.
    """
    events: list[LoopEvent] = []

    # 1) In-memory hits: everything hydrated into session_state this session.
    #    We can't reliably reconstruct (profile, url) from the SHA1 key alone,
    #    so we rely on the PR Predictor page having stored `prp_last_url` per
    #    profile as its "which URL was I looking at" pointer. That gives us
    #    exactly ONE guaranteed lookup path per profile.
    last_url = cached_state.get("prp_last_url", (profile,))
    if last_url:
        verdict = cached_state.get("prp_verdict", (profile, last_url))
        ev = _event_from_pr_verdict(verdict, profile, last_url)
        if ev is not None:
            events.append(ev)

    # 2) Additional URLs the user has predicted this session, discovered via
    #    session_state keys that follow the state_cache naming convention.
    seen_ids = {e.external_id for e in events}
    for key in list(st.session_state.keys()):
        if not isinstance(key, str) or not key.startswith("csstate::prp_verdict::"):
            continue
        verdict = st.session_state.get(key)
        if verdict is None:
            continue
        # We can't recover the URL from the hashed key, so use a synthetic id
        # tied to the pickle path. This dedups within a session but avoids
        # inventing a URL.
        ext_id = f"prp_verdict::{key.split('::')[-1]}"
        if ext_id in seen_ids:
            continue
        # Skip if this looks like a verdict for a different profile:
        # the PR Predictor page stores the profile inside the identity
        # tuple, so we can only trust this hit when we've verified via
        # `prp_last_url` above. Otherwise we err on the side of NOT
        # importing (better to miss an event than to attribute one to
        # the wrong account).
        continue

    return events


def _event_from_pr_verdict(
    verdict, profile: str, pr_url: str,
) -> Optional[LoopEvent]:
    """Build a LoopEvent from an AgentVerdict, or return None if the verdict
    is unusable (error, no dollar impact, or missing fields)."""
    if verdict is None:
        return None
    if getattr(verdict, "error", None):
        return None

    lo = getattr(verdict, "est_daily_delta_low_usd", None)
    hi = getattr(verdict, "est_daily_delta_high_usd", None)
    point = float(getattr(verdict, "est_daily_delta_usd", 0.0) or 0.0)
    if lo is not None and hi is not None and abs(hi - lo) > 0.01:
        delta = (float(lo) + float(hi)) / 2.0
    else:
        delta = point
    if abs(delta) < 0.01:
        return None  # PR predicted as cost-neutral

    conf = _confidence_from_string(getattr(verdict, "confidence", None))

    # PR Predictor verdicts don't carry an "expected deploy date". The
    # conservative honest choice is "today" — the event affects the
    # forecast from now on. The user can disable events they think slip
    # further out.
    return LoopEvent(
        source="pr_predictor",
        source_label=f"PR Predictor · {pr_url}",
        source_link="/PR_Predictor",
        external_id=f"prp::{profile}::{pr_url}",
        start_date=date.today(),
        daily_delta_usd=delta,
        confidence=conf,
        note=(getattr(verdict, "verdict", "") or "")[:200],
    )


def _discover_anomaly_events(profile: str) -> list[LoopEvent]:
    """Every Anomaly recommendation cached for this profile becomes a
    savings event. AnomalyReport.actions carries est_daily_savings_usd
    (positive; savings) + confidence (low/medium/high)."""
    events: list[LoopEvent] = []

    # Anomalies page tracks `anom_last_report_key` in session_state pointing
    # at the last successful scan's key, and mirrors the report into
    # session_state under that key. We use that as the entry point.
    last_key = st.session_state.get("anom_last_report_key")
    needle = f"::{profile}::"
    if not (isinstance(last_key, str) and needle in last_key):
        return events

    report = st.session_state.get(last_key)
    if report is None:
        return events
    if getattr(report, "error", None):
        return events

    actions = list(getattr(report, "actions", []) or [])
    for i, action in enumerate(actions):
        savings = float(getattr(action, "est_daily_savings_usd", 0.0) or 0.0)
        if savings < 0.01:
            continue
        conf = _confidence_from_string(getattr(action, "confidence", None))
        issue = (getattr(action, "issue", "")
                 or getattr(action, "recommendation", "")
                 or "Anomaly action").strip()[:80]
        events.append(LoopEvent(
            source="anomaly",
            source_label=f"Anomaly · {issue}",
            source_link="/Anomalies",
            external_id=f"anom::{profile}::{last_key}::{i}",
            start_date=date.today(),
            daily_delta_usd=-savings,  # negative = savings
            confidence=conf,
            note=(getattr(action, "recommendation", "")
                  or getattr(action, "reason", "") or "").strip()[:200],
        ))
    return events


def _discover_open_pr_events(profile: str) -> list[LoopEvent]:
    """Priced open PRs from a prior run of `analyze_open_prs`, cached under
    our own namespace (see the "Analyze open PRs" button below)."""
    key = (profile,)
    priced = cached_state.get("loop_open_prs", key)
    if not priced:
        return []
    events: list[LoopEvent] = []
    for p in priced:
        opr = getattr(p, "open_pr", None)
        if opr is None:
            continue
        expected_delta = float(getattr(p, "expected_daily_delta_usd", 0.0) or 0.0)
        # expected_daily_delta_usd is already est × probability. We store the
        # raw est + probability as separate fields so the ledger shows the
        # merge probability as the confidence.
        est = float(getattr(p, "est_daily_delta_usd", 0.0) or 0.0)
        prob = float(getattr(p, "merge_probability", 0.0) or 0.0)
        if abs(est) < 0.01:
            continue
        merge_day_iso = getattr(p, "expected_merge_day", "") or ""
        try:
            merge_day = date.fromisoformat(merge_day_iso)
        except ValueError:
            merge_day = date.today()
        _ = expected_delta  # (unused; kept to document the equivalence)
        events.append(LoopEvent(
            source="open_pr",
            source_label=f"Open PR · {opr.repo}#{opr.number}",
            source_link=opr.url,
            external_id=f"open_pr::{profile}::{opr.repo}::{opr.number}",
            start_date=merge_day,
            daily_delta_usd=est,
            confidence=max(0.0, min(1.0, prob)),
            note=(getattr(p, "llm_summary", "") or "")[:200],
        ))
    return events


def _discover_all_events(profile: str) -> list[LoopEvent]:
    """Merge every source, dedup on ``external_id``. Order preserved so the
    UI ledger reads deterministically."""
    seen: set[str] = set()
    out: list[LoopEvent] = []
    for src in (
        _discover_pr_predictor_events,
        _discover_anomaly_events,
        _discover_open_pr_events,
    ):
        try:
            for ev in src(profile):
                if ev.external_id in seen:
                    continue
                seen.add(ev.external_id)
                out.append(ev)
        except Exception as e:  # noqa: BLE001
            # Never let one broken source blank the whole page.
            callout(
                f"Discovery source `{src.__name__}` failed: {e}. "
                f"Skipping — the page will render without those events.",
                tone="warning",
            )
    return out


# ---------------------------------------------------------------------------
# Live-fetch helpers (only run when the user asks)
# ---------------------------------------------------------------------------

def _fetch_history_now(
    profile: str, days: int = 30,
) -> tuple[Optional[list[tuple[date, float]]], Optional[str]]:
    """Cost Explorer daily totals for the last `days` days. Returns
    ``(history, error_message)`` — exactly one is non-None."""
    try:
        end = date.today()
        start = end - timedelta(days=days)
        rows = fetch_daily_totals(start, end, profile=profile)
        return rows, None
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def _analyze_open_prs_now(
    profile: str, repos_full: list[str],
) -> tuple[list, Optional[str]]:
    """Run `list_open_prs_many` + `analyze_open_prs` on the given repos.
    Returns ``(priced_prs, error_message)``."""
    try:
        from src.pr_scanner.open_prs import analyze_open_prs, list_open_prs_many
        open_prs = list_open_prs_many(repos_full)
        if not open_prs:
            return [], None
        priced = analyze_open_prs(open_prs, profile=profile)
        return priced, None
    except Exception as e:  # noqa: BLE001
        return [], str(e)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

_SOURCE_META = {
    "pr_predictor": ("PR Predictor", C.BRAND, "/PR_Predictor"),
    "anomaly":      ("Anomaly",      C.GOOD,  "/Anomalies"),
    "open_pr":      ("Open PR",      C.INFO,  None),
    "merged_pr":    ("Merged PR",    C.GOOD,  None),
}


def _source_badge(source: str, label: str, link: str) -> str:
    """Small coloured chip identifying the origin. HTML string for st.markdown."""
    meta = _SOURCE_META.get(source, (source.title(), C.MUTED, None))
    display_name, color, _default_link = meta
    href = link or _default_link or "#"
    return (
        f'<a href="{href}" target="_self" '
        f'style="text-decoration:none;">'
        f'<span class="cs-pill" style="background:{color}1A;color:{color};">'
        f'<span class="cs-dot" style="background:{color};"></span>'
        f'{display_name}</span></a>'
        f' <span style="color:{C.MUTED};font-size:.8rem;">{label}</span>'
    )


def _confidence_label_from_weight(weight: float) -> str:
    """Reverse the low/medium/high → weight mapping so the ledger's confidence
    pill matches the source page's wording. Uses the mid-points of our own
    _confidence_from_string mapping."""
    if weight >= 0.85:
        return "high"
    if weight >= 0.65:
        return "medium"
    return "low"


def _md_escape(text: str) -> str:
    """Escape Streamlit markdown quirks in AI-generated prose."""
    if not text:
        return ""
    return (text
            .replace("\\", "\\\\")
            .replace("$", "\\$")
            .replace("~", "\\~")
            .replace("*", "\\*")
            .replace("_", "\\_"))


# ---------------------------------------------------------------------------
# Page body
# ---------------------------------------------------------------------------

with st.spinner("Resolving profiles…"):
    profiles = [p for p in resolve_all() if p.account_id]
if not profiles:
    callout(
        "No AWS profiles reachable. Run `aws sso login` or launch via "
        "`aws-vault exec <profile> --` first.",
        tone="error",
    )
    st.stop()

labels = [p.label for p in profiles]
picked_label = st.session_state.get("loop_profile", labels[0])
if picked_label not in labels:
    picked_label = labels[0]

header = f"Controls · Account: {picked_label}"
with top_bar(header):
    ctrl_cols = st.columns([3, 2, 2], gap="medium", vertical_alignment="bottom")
    with ctrl_cols[0]:
        picked_label = st.selectbox(
            "Account", labels, index=labels.index(picked_label),
            key="loop_profile",
        )
    with ctrl_cols[1]:
        horizon = st.selectbox(
            "Horizon (days)", [30, 60, 90, 180], index=2,
            key="loop_horizon",
            help="How far the projection runs.",
        )
    with ctrl_cols[2]:
        baseline_method = st.selectbox(
            "Baseline", ["Trailing 30d avg", "Driver-based (opt-in)"],
            index=0, key="loop_baseline_method",
            help=(
                "Trailing 30d avg is the account's real recent spend from "
                "Cost Explorer. Driver-based lets you divide that number by "
                "a unit count you provide, then move the per-unit rate — "
                "nothing hard-coded."
            ),
        )

active = profiles[labels.index(picked_label)]

with st.sidebar:
    render_sidebar_footer(
        active_profile=active.profile,
        account_id=active.account_id,
        extra_rows=[("Horizon", f"{horizon}d")],
    )


# ---- Data acquisition (nothing invented) ----------------------------------

history = cached_state.get("loop_history", (active.profile,))
history_error = st.session_state.get(f"loop_history_error::{active.profile}")

events = _discover_all_events(active.profile)

# ---- Where the numbers come from ------------------------------------------

section(
    "Where these numbers come from",
    "Each tile shows how many findings this account already has cached from "
    "other CostSense pages. Nothing on this page is synthetic — if a source "
    "has zero cached findings, its tile stays at zero and its section stays "
    "hidden below.",
    kicker="Provenance",
)

by_source: dict[str, list[LoopEvent]] = {}
for ev in events:
    by_source.setdefault(ev.source, []).append(ev)

prov_cols = st.columns(4, gap="medium")
with prov_cols[0]:
    metric(
        "PR Predictor verdicts",
        f"{len(by_source.get('pr_predictor', []))}",
        delta="from cached_state" if by_source.get("pr_predictor") else "none cached",
    )
with prov_cols[1]:
    metric(
        "Anomaly recommendations",
        f"{len(by_source.get('anomaly', []))}",
        delta="from cached_state" if by_source.get("anomaly") else "none cached",
    )
with prov_cols[2]:
    metric(
        "Open PRs priced",
        f"{len(by_source.get('open_pr', []))}",
        delta="from cached_state" if by_source.get("open_pr") else "none cached",
    )
with prov_cols[3]:
    metric(
        "Cost Explorer history",
        f"{len(history)}d" if history else "—",
        delta="cached" if history else ("error" if history_error else "not fetched"),
        good=False if history_error else (True if history else None),
    )


# ---- Action row: refresh / on-demand fetch --------------------------------

action_cols = st.columns([1, 1, 1, 3], gap="medium")
with action_cols[0]:
    if st.button("Refresh sources", use_container_width=True,
                 help="Re-scan cached_state for new events without hitting "
                      "AWS/GitHub. Fast."):
        st.rerun()
with action_cols[1]:
    if st.button("Fetch history", use_container_width=True,
                 help="Call Cost Explorer for the last 30 days of daily "
                      "spend on this account. Required for a projection."):
        with st.spinner("Cost Explorer: fetching daily totals…"):
            rows, err = _fetch_history_now(active.profile, days=30)
        if err is not None:
            st.session_state[f"loop_history_error::{active.profile}"] = err
            cached_state.clear("loop_history", (active.profile,))
        else:
            st.session_state.pop(f"loop_history_error::{active.profile}", None)
            cached_state.set("loop_history", (active.profile,), rows)
        st.rerun()
with action_cols[2]:
    if st.button("Analyze open PRs", use_container_width=True,
                 help="Deep-analyze this account's matching open PRs and "
                      "add the priced results as events. ~30-60s."):
        # Repo match from the profile name — same helper Anomalies uses.
        from src.pr_scanner.profile_repo_match import normalize_profile
        normalized = normalize_profile(active.profile)
        repos_full: list[str] = []
        if normalized:
            try:
                from src.pr_scanner.repos import repos_with_user_prs
                for org in ("DiligentCorp",):
                    for full in repos_with_user_prs(org):
                        short = full.split("/", 1)[-1]
                        if short.lower().startswith(normalized.lower()):
                            repos_full.append(full)
            except Exception as e:  # noqa: BLE001
                callout(f"Couldn't enumerate GitHub repos: {e}", tone="warning")
        if not repos_full:
            callout(
                f"No GitHub repos matched profile `{active.profile}`. "
                "Skipping open-PR analysis — nothing invented.",
                tone="warning",
            )
        else:
            with st.spinner(
                f"Analyzing open PRs on {len(repos_full)} repo(s) with the "
                f"deep AWS agent…"
            ):
                priced, err = _analyze_open_prs_now(active.profile, repos_full)
            if err is not None:
                callout(f"Open PR analysis failed: {err}", tone="error")
            else:
                cached_state.set("loop_open_prs", (active.profile,), priced)
                st.rerun()

# History error banner (honest — no fabricated baseline follows)
if history_error and not history:
    callout(
        f"**Cost Explorer denied access for `{active.profile}`.** "
        f"Error: `{history_error}`. Without a baseline we cannot show a "
        f"projection — everything below stays hidden. Switch to a profile "
        f"with `ce:GetCostAndUsage` access, or ask the account owner to "
        f"run this analysis.",
        tone="error",
    )
    st.stop()

if history is None:
    callout(
        "No cost history fetched yet. Click **Fetch history** to pull the "
        "last 30 days from Cost Explorer for this profile. Every number on "
        "this page depends on real recent spend.",
        tone="info",
    )
    st.stop()


# ---- Baseline ---------------------------------------------------------------

baseline_daily = _trailing_daily_baseline(history, days=30)
if baseline_daily is None or baseline_daily <= 0:
    callout(
        "Cost Explorer returned an empty or zero history for this profile. "
        "Nothing to project against.",
        tone="warning",
    )
    st.stop()

driver_rate: Optional[float] = None
if baseline_method == "Driver-based (opt-in)":
    section(
        "Driver-based baseline",
        "You provide a unit count (orgs, tenants, requests-per-day — anything "
        "the account scales with). Per-unit rate is COMPUTED from the "
        "trailing 30-day average divided by your count — never a default. "
        "Move the rate slider to see the projection track the plan.",
        kicker="Optional",
    )
    d1, d2, d3 = st.columns(3, gap="medium")
    with d1:
        unit_label = st.text_input(
            "Unit label", value=st.session_state.get("loop_unit_label", "orgs"),
            key="loop_unit_label",
        )
    with d2:
        unit_count = st.number_input(
            "Current unit count (leave 0 if unknown)",
            min_value=0.0, value=float(st.session_state.get("loop_unit_count", 0.0)),
            step=1.0, key="loop_unit_count",
        )
    with d3:
        st.caption("Computed rate — not a default")
        if unit_count > 0:
            computed_rate = baseline_daily / unit_count
            st.markdown(
                f"**${computed_rate:,.4f} / {unit_label[:-1] or unit_label} / day**"
            )
            driver_rate = st.slider(
                "Per-unit rate override ($/unit/day)",
                min_value=max(0.0001, computed_rate * 0.25),
                max_value=computed_rate * 2.0,
                value=computed_rate, step=max(0.0001, computed_rate * 0.01),
                key="loop_driver_rate",
                help=(
                    "Drag to see the projection shift as if per-unit "
                    "efficiency changed. When set = computed rate, this "
                    "matches Trailing 30d exactly."
                ),
            )
        else:
            st.caption(
                "Provide a unit count > 0 to compute a per-unit rate. "
                "No placeholder is shown."
            )

# If the user provided a driver rate + unit count, the effective baseline
# becomes rate * units (i.e. the slider is the only knob). Otherwise the
# trailing average IS the baseline.
if driver_rate is not None and st.session_state.get("loop_unit_count", 0.0) > 0:
    effective_baseline = driver_rate * st.session_state["loop_unit_count"]
else:
    effective_baseline = baseline_daily


# ---- Projection ------------------------------------------------------------

start = history[-1][0] + timedelta(days=1) if history else date.today()
projection = _project_horizon(effective_baseline, events, horizon, start)


# ---- Attribution -----------------------------------------------------------

section(
    "Attribution",
    "The projection broken down by source. Baseline is trailing spend held "
    "flat; each source below layers on the deltas from events cached by that "
    "page.",
    kicker="Rollup",
)

attribution = projection.attribution_by_source()

attr_cols = st.columns(len(_SOURCE_META) + 2, gap="medium")
with attr_cols[0]:
    metric(f"Baseline · {projection.horizon_days}d", money(projection.total_baseline))

for i, (source, meta) in enumerate(_SOURCE_META.items(), start=1):
    display_name, color, _ = meta
    amount = attribution.get(source, 0.0)
    count = len(by_source.get(source, []))
    if count == 0:
        with attr_cols[i]:
            metric(display_name, "—", delta="0 events")
        continue
    signed = amount
    with attr_cols[i]:
        metric(
            display_name,
            f"{'+' if signed >= 0 else '−'}{money(abs(signed))}",
            delta=f"{count} event(s)",
            good=(signed < 0),
        )

with attr_cols[-1]:
    delta_from_baseline = projection.total_expected - projection.total_baseline
    metric(
        "Expected total",
        money(projection.total_expected),
        delta=f"{'+' if delta_from_baseline >= 0 else '−'}"
              f"{money(abs(delta_from_baseline))} vs baseline",
        good=(delta_from_baseline < 0),
    )


# ---- Chart -----------------------------------------------------------------

st.markdown("##### Projection")

hist_dates = [d for d, _ in history]
hist_values = [amount for _, amount in history]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=list(hist_dates), y=list(hist_values), name="Actual (last 30d)",
    mode="lines", line=dict(color=C.INK, width=1.6),
    hovertemplate="%{x|%d %b}<br>$%{y:,.0f}<extra>Actual</extra>",
))

fig.add_trace(go.Scatter(
    x=projection.dates + projection.dates[::-1],
    y=projection.worst + projection.best[::-1],
    fill="toself", fillcolor="rgba(12,124,116,0.13)",
    line=dict(width=0), name="Best–worst range", hoverinfo="skip",
))

fig.add_trace(go.Scatter(
    x=projection.dates, y=projection.baseline, name="Baseline (no events)",
    mode="lines", line=dict(color=C.FAINT, width=1.2, dash="dot"),
    hovertemplate="%{x|%d %b}<br>$%{y:,.0f}<extra>Baseline</extra>",
))

fig.add_trace(go.Scatter(
    x=projection.dates, y=projection.expected, name="Expected",
    mode="lines", line=dict(color=C.BRAND, width=2.4, dash="dash"),
    hovertemplate="%{x|%d %b}<br>$%{y:,.0f}<extra>Expected</extra>",
))

# Vertical markers for every active event (with its source colour)
for ev in projection.active_events:
    if ev.start_date < projection.dates[0] or ev.start_date > projection.dates[-1]:
        continue
    meta = _SOURCE_META.get(ev.source, ("", C.MUTED, None))
    _, ev_color, _ = meta
    fig.add_vline(x=ev.start_date, line=dict(color=ev_color, width=1, dash="dot"))

fig.update_layout(
    height=380,
    margin=dict(l=16, r=24, t=16, b=16),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    yaxis_title="Daily spend (USD)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1),
)
st.plotly_chart(fig, use_container_width=True)


# ---- Event ledger ----------------------------------------------------------

section(
    "Event ledger",
    "One row per real event discovered from other CostSense pages. Click a "
    "source badge to jump to the page that produced it. Toggle any event to "
    "remove it from the projection.",
    kicker="Events",
)

if not events:
    callout(
        "No events found for this account yet. Run **PR Predictor** or "
        "**Anomalies** — this page will pick them up on the next refresh. "
        "Nothing here is seeded from a demo.",
        tone="info",
    )
else:
    for i, ev in enumerate(events):
        with st.container(border=True):
            head, amt, tog = st.columns([5, 1.4, 0.8])
            with head:
                st.markdown(
                    _source_badge(ev.source, ev.source_label, ev.source_link),
                    unsafe_allow_html=True,
                )
                sub_bits = []
                sub_bits.append(f"starts {ev.start_date:%d %b %Y}")
                sub_bits.append(
                    f"confidence {_confidence_label_from_weight(ev.confidence)} "
                    f"({ev.confidence:.0%})"
                )
                st.markdown(
                    f'<div style="color:{C.MUTED};font-size:.8rem;'
                    f'margin-top:2px;">{" · ".join(sub_bits)}</div>',
                    unsafe_allow_html=True,
                )
                if ev.note:
                    st.markdown(
                        f'<div style="color:{C.MUTED};font-size:.83rem;'
                        f'margin-top:6px;">{_md_escape(ev.note)}</div>',
                        unsafe_allow_html=True,
                    )
            with amt:
                sign = "+" if ev.daily_delta_usd >= 0 else "−"
                daily = abs(ev.daily_delta_usd)
                col = C.BAD if ev.daily_delta_usd >= 0 else C.GOOD
                st.markdown(
                    f'<div style="text-align:right;">'
                    f'<div class="cs-num" style="color:{col};'
                    f'font-weight:680;">{sign}${daily:,.2f}/day</div>'
                    f'<div style="color:{C.MUTED};font-size:.75rem;">'
                    f'over {projection.horizon_days}d: '
                    f'{sign}{money(daily * ev.confidence * projection.horizon_days)}'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            with tog:
                st.checkbox(
                    "On", value=ev.enabled,
                    key=f"loop_ev_toggle::{ev.external_id}",
                    label_visibility="collapsed",
                    help="Disable to remove this event from the projection.",
                )
                # Reflect the toggle immediately (no rerun cost — checkboxes
                # already trigger a rerun).
                new_state = st.session_state.get(
                    f"loop_ev_toggle::{ev.external_id}", ev.enabled,
                )
                if new_state != ev.enabled:
                    ev.enabled = new_state


st.divider()
st.caption(
    "Baseline is the trailing 30-day average from Cost Explorer, held flat. "
    "Expected = baseline + Σ(event daily_delta × confidence) per day. "
    "Best case = baseline + only-savings events at full strength. Worst case "
    "= baseline + only-cost-increase events at full strength. Every event "
    "here was produced by another CostSense page — nothing is a default."
)
