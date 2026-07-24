"""Close the Loop — one view of every priced recommendation CostSense has
already produced for the active account.

Design rule (deliberate)
========================
This page does NOT model anything. It does NOT forecast. It does NOT
extrapolate. It matches the honest pattern the other pages use:

  * PR Predictor:
        _projected_headline = current_daily + _delta_headline
        _monthly            = _delta_headline * 30

  * Anomalies:
        savings = report.total_daily_savings_usd
        (renders `actions` list verbatim)

Both compute the projected/potential number by ADDING a live $/day delta the
LLM already returned to a live $/day current from Cost Explorer. No trailing
averages, no per-day summation across a horizon, no best-vs-worst bands, no
driver-based abstractions. Just aggregation.

This tab does the same, one level up:

  net delta $/day       = Σ(PR-Predictor deltas)          [cached verdicts]
                          + Σ(Anomaly savings, negative)  [cached actions]
                          + Σ(Open-PR expected deltas)    [cached priced PRs]
  projected $/day       = current_daily + net_delta

Every number displayed is either:
  - a live boto3 Cost Explorer field
  - a cached AgentVerdict / Action / PricedOpenPr field the source page
    already showed the user on its own tab

If a source has nothing cached for the active profile, its tile shows an em
dash and its section stays hidden. Nothing on this page is a default, a
demo, or a projected curve.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import plotly.graph_objects as go
import streamlit as st

from src.aws.cost_explorer import fetch_daily_totals
from src.aws.profiles import resolve_all
from src.dashboard.costsense_theme import (
    C, callout, confidence_pill, metric, money, plotly_layout, section,
)


def _rgba(hex_color: str, alpha: float) -> str:
    """Hex → rgba() for Plotly fill colours. Mirrors Dashboard's helper so
    both pages produce visually identical translucent fills."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
from src.dashboard.nav import (
    inject_css, render_sidebar_footer, render_sidebar_header, top_bar,
)
from src.dashboard.state_cache import cached_state


st.set_page_config(page_title="CostSense · Close the Loop", layout="wide")
inject_css()
render_sidebar_header()

section(
    "Close the loop",
    "Every priced recommendation CostSense has already produced for this "
    "account — from PR Predictor, Anomalies, and Open PRs — aggregated in "
    "one view. No forecasting: every number is either a live Cost Explorer "
    "reading or a value returned by one of the other tabs.",
    kicker="Aggregate",
)


# ---------------------------------------------------------------------------
# Data model — just a passthrough of what other pages produced
# ---------------------------------------------------------------------------

@dataclass
class LoopItem:
    """One priced finding from another CostSense tab. Every field comes
    verbatim from the source cache — no interpretation."""

    source: str          # "pr_predictor" | "anomaly" | "open_pr"
    label: str           # human-readable, drawn from the source object
    daily_delta_usd: float   # signed: + cost, − saving. From the source.
    confidence: str          # raw string from the source: "low"|"medium"|"high"
                             # or (open PR) a "%%" merge-probability string
    link: str                # deep-link back to the tab that made this item
    note: str = ""


# ---------------------------------------------------------------------------
# Cache-only discovery (no live calls except Cost Explorer for current $/day)
# ---------------------------------------------------------------------------

def _pr_predictor_items(profile: str) -> list[LoopItem]:
    """Every AgentVerdict cached on this profile becomes one LoopItem.

    Mirrors PR Predictor page: uses `est_daily_delta_usd` (or the midpoint of
    the low/high range when both are present, matching PR Predictor's
    `_delta_headline` line 240-242). Confidence is the AgentVerdict's own
    `confidence` field verbatim."""
    items: list[LoopItem] = []
    last_url = cached_state.get("prp_last_url", (profile,))
    if not last_url:
        return items
    verdict = cached_state.get("prp_verdict", (profile, last_url))
    if verdict is None or getattr(verdict, "error", None):
        return items

    lo = getattr(verdict, "est_daily_delta_low_usd", None)
    hi = getattr(verdict, "est_daily_delta_high_usd", None)
    if lo is not None and hi is not None and abs(hi - lo) > 0.01:
        delta = (float(lo) + float(hi)) / 2.0
    else:
        delta = float(getattr(verdict, "est_daily_delta_usd", 0.0) or 0.0)
    if abs(delta) < 0.01:
        return items

    items.append(LoopItem(
        source="pr_predictor",
        label=f"PR Predictor · {last_url}",
        daily_delta_usd=delta,
        confidence=(getattr(verdict, "confidence", "") or "medium").lower(),
        link="PR_Predictor",
        note=(getattr(verdict, "verdict", "") or "")[:200],
    ))
    return items


def _anomaly_items(profile: str) -> list[LoopItem]:
    """Every Action from the last Anomaly scan cached on this profile.

    Mirrors Anomalies page (line 692): `save_amt = a.est_daily_savings_usd`.
    We just negate it (savings render as negative deltas here so they line
    up with PR-Predictor cost increases in the aggregation)."""
    items: list[LoopItem] = []
    last_key = st.session_state.get("anom_last_report_key")
    needle = f"::{profile}::"
    if not (isinstance(last_key, str) and needle in last_key):
        return items
    report = st.session_state.get(last_key)
    if report is None or getattr(report, "error", None):
        return items

    for action in getattr(report, "actions", []) or []:
        savings = float(getattr(action, "est_daily_savings_usd", 0.0) or 0.0)
        if savings < 0.01:
            continue
        issue = (getattr(action, "issue", "")
                 or getattr(action, "recommendation", "")
                 or "Anomaly action").strip()[:80]
        items.append(LoopItem(
            source="anomaly",
            label=f"Anomaly · {issue}",
            daily_delta_usd=-savings,
            confidence=(getattr(action, "confidence", "") or "medium").lower(),
            link="Anomalies",
            note=(getattr(action, "recommendation", "")
                  or getattr(action, "reason", "") or "").strip()[:200],
        ))
    return items


def _open_pr_items(profile: str) -> list[LoopItem]:
    """PricedOpenPrs cached under our namespace by the 'Analyze open PRs'
    button. Mirrors the field the Dashboard uses:
    ``expected_daily_delta_usd`` — already probability-weighted at the
    source (est × merge_probability)."""
    priced = cached_state.get("loop_open_prs", (profile,))
    if not priced:
        return []
    items: list[LoopItem] = []
    for p in priced:
        opr = getattr(p, "open_pr", None)
        if opr is None:
            continue
        expected = float(getattr(p, "expected_daily_delta_usd", 0.0) or 0.0)
        if abs(expected) < 0.01:
            continue
        prob = float(getattr(p, "merge_probability", 0.0) or 0.0)
        items.append(LoopItem(
            source="open_pr",
            label=f"Open PR · {opr.repo}#{opr.number}",
            daily_delta_usd=expected,
            confidence=f"{prob:.0%} merge",
            link=opr.url,
            note=(getattr(p, "llm_summary", "") or "")[:200],
        ))
    return items


def _all_items(profile: str) -> list[LoopItem]:
    out: list[LoopItem] = []
    for src in (_pr_predictor_items, _anomaly_items, _open_pr_items):
        try:
            out.extend(src(profile))
        except Exception as e:  # noqa: BLE001
            callout(
                f"Discovery source `{src.__name__}` failed: {e}. "
                f"Skipping — other sources still render.",
                tone="warning",
            )
    return out


# ---------------------------------------------------------------------------
# Live: current $/day — same shape PR Predictor uses (line 196-200)
# ---------------------------------------------------------------------------

def _fetch_current_daily(profile: str) -> tuple[Optional[float], Optional[str]]:
    """Trailing 7-day mean from Cost Explorer. Same computation PR Predictor
    performs on its own page — we reuse the identical pattern so the two
    tabs agree to the cent."""
    try:
        today = date.today()
        totals = fetch_daily_totals(today - timedelta(days=7), today, profile=profile)
        if not totals:
            return None, "Cost Explorer returned no rows for the last 7 days."
        avg = sum(a for _, a in totals) / len(totals)
        return avg, None
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def _fetch_history_30d(profile: str) -> tuple[Optional[list[tuple[date, float]]], Optional[str]]:
    try:
        today = date.today()
        rows = fetch_daily_totals(today - timedelta(days=30), today, profile=profile)
        return rows or [], None
    except Exception as e:  # noqa: BLE001
        return None, str(e)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

_SOURCE_META = {
    "pr_predictor": ("PR Predictor", C.BRAND),
    "anomaly":      ("Anomaly",      C.GOOD),
    "open_pr":      ("Open PR",      C.INFO),
}


def _md_escape(text: str) -> str:
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

with top_bar(f"Controls · Account: {picked_label}"):
    picked_label = st.selectbox(
        "Account", labels, index=labels.index(picked_label),
        key="loop_profile",
    )

active = profiles[labels.index(picked_label)]

with st.sidebar:
    render_sidebar_footer(
        active_profile=active.profile,
        account_id=active.account_id,
    )


# ---- Cost Explorer (live) --------------------------------------------------

current_daily = cached_state.get("loop_current_daily", (active.profile,))
current_daily_err = st.session_state.get(f"loop_current_daily_err::{active.profile}")
history_30d = cached_state.get("loop_history_30d", (active.profile,))

refresh_cols = st.columns([1, 1, 4], gap="medium")
with refresh_cols[0]:
    if st.button("Fetch current $/day", use_container_width=True,
                 help="Call Cost Explorer for the trailing 7-day average. "
                      "Same as PR Predictor uses."):
        with st.spinner("Cost Explorer…"):
            val, err = _fetch_current_daily(active.profile)
        if err:
            st.session_state[f"loop_current_daily_err::{active.profile}"] = err
            cached_state.clear("loop_current_daily", (active.profile,))
        else:
            st.session_state.pop(f"loop_current_daily_err::{active.profile}", None)
            cached_state.set("loop_current_daily", (active.profile,), val)
        st.rerun()
with refresh_cols[1]:
    if st.button("Fetch 30-day history", use_container_width=True,
                 help="Daily totals for the last 30 days — used for the "
                      "context chart. No projection is drawn."):
        with st.spinner("Cost Explorer…"):
            rows, err = _fetch_history_30d(active.profile)
        if err:
            callout(f"Cost Explorer denied: {err}", tone="error")
        else:
            cached_state.set("loop_history_30d", (active.profile,), rows)
            st.rerun()
with refresh_cols[2]:
    if st.button("Refresh cached findings",
                 help="Re-scan cached_state for PR Predictor / Anomaly / "
                      "Open PR items. No AWS or GitHub calls."):
        st.rerun()

if current_daily_err and current_daily is None:
    callout(
        f"**Cost Explorer denied access for `{active.profile}`.** "
        f"`{current_daily_err}`. Without a current $/day reading, projected "
        f"$/day can't be computed — the tiles stay empty rather than showing "
        f"a fabricated number.",
        tone="error",
    )


# ---- Aggregate the cached findings (no math beyond addition) --------------

items = _all_items(active.profile)

by_source: dict[str, list[LoopItem]] = {}
for it in items:
    by_source.setdefault(it.source, []).append(it)


# ---- Headline tiles — same shape PR Predictor uses -------------------------

section(
    "Today's picture",
    "Current $/day is the trailing 7-day average from Cost Explorer. Net "
    "delta is the sum of every cached recommendation's $/day. Projected is "
    "the addition — the same one-line calculation PR Predictor shows for a "
    "single PR, only rolled up.",
    kicker="Headline",
)

net_delta = sum(it.daily_delta_usd for it in items)
projected = (current_daily + net_delta) if current_daily is not None else None

tiles = st.columns(4, gap="medium")
with tiles[0]:
    metric(
        "Current account $/day",
        f"${current_daily:,.2f}" if current_daily is not None else "—",
        delta="trailing 7-day avg" if current_daily is not None
              else "click Fetch current $/day",
    )
with tiles[1]:
    metric(
        "Net delta from recommendations",
        f"${net_delta:+,.2f}/day" if items else "—",
        delta=f"{len(items)} finding(s) aggregated" if items else "no findings cached",
        good=(net_delta < 0) if items else None,
    )
with tiles[2]:
    metric(
        "Projected $/day (current + net)",
        f"${projected:,.2f}" if projected is not None else "—",
        delta=(f"{net_delta:+,.2f} vs current"
               if projected is not None and items else None),
        good=(net_delta < 0) if items else None,
    )
with tiles[3]:
    metric(
        "Monthly equivalent",
        f"${net_delta * 30:+,.0f}" if items else "—",
        delta="net delta × 30" if items else None,
        good=(net_delta < 0) if items else None,
    )


# ---- Attribution: same total, broken down by source -----------------------

section(
    "Where the net delta comes from",
    "The same net-delta $/day above, split by source page. Each row is a "
    "sum of the deltas that page contributed — no weighting, no scaling.",
    kicker="Attribution",
)

attr_cols = st.columns(3, gap="medium")
for i, source in enumerate(("pr_predictor", "anomaly", "open_pr")):
    display_name, _ = _SOURCE_META[source]
    src_items = by_source.get(source, [])
    src_sum = sum(it.daily_delta_usd for it in src_items)
    with attr_cols[i]:
        if not src_items:
            metric(display_name, "—", delta="no cached findings")
        else:
            metric(
                display_name,
                f"${src_sum:+,.2f}/day",
                delta=f"{len(src_items)} finding(s)",
                good=(src_sum < 0),
            )


# ---- Context chart: real history only, no forecast line -------------------

if history_30d:
    st.markdown("##### 30-day history")
    dates = [d for d, _ in history_30d]
    vals = [a for _, a in history_30d]

    # Match Dashboard's chart recipe exactly: spline curve + markers + brand
    # teal + shared `plotly_layout()` for consistent theme (grid colour,
    # hover card, font, legend position). Dashboard uses this at
    # pages/2_Dashboard.py:802-806 and its identical helper — copying that
    # so the two pages read as one visual system.
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=vals, name="Actual daily spend",
        mode="lines+markers",
        line=dict(color=C.BRAND, width=2.5, shape="spline", smoothing=1.0),
        marker=dict(size=6),
        hovertemplate="%{x|%d %b}<br>$%{y:,.0f}<extra>Actual</extra>",
    ))
    # Horizontal reference lines: both are single live values already shown
    # in tiles above. Drawing them makes the "current vs projected" gap
    # visible — no forecast curve, no projected future line.
    if current_daily is not None:
        fig.add_hline(y=current_daily, line=dict(color=C.MUTED, width=1, dash="dot"),
                      annotation_text="Current $/day (7d avg)",
                      annotation_position="top left",
                      annotation_font=dict(size=10, color=C.MUTED))
    if projected is not None and items:
        fig.add_hline(y=projected, line=dict(color=C.BRAND_DARK, width=1.5, dash="dashdot"),
                      annotation_text="Projected $/day (current + net delta)",
                      annotation_position="bottom left",
                      annotation_font=dict(size=10, color=C.BRAND_DARK))
    fig.update_layout(
        **plotly_layout(height=380),
        yaxis_title="USD / day",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


# ---- Findings ledger — one row per cached recommendation ------------------

section(
    "Cached findings",
    "One row per item CostSense has already produced for this account. Each "
    "row links back to the tab that created it, where the full context "
    "(diff, tool calls, rationale) lives.",
    kicker="Ledger",
)

if not items:
    callout(
        "No cached findings for this account. Run **PR Predictor** on a PR, "
        "run an **Anomaly** scan, or click **Analyze open PRs** below — "
        "this page will pick them up on the next refresh. Nothing here is "
        "seeded.",
        tone="info",
    )

for it in items:
    with st.container(border=True):
        head, amt = st.columns([5, 1.4])
        with head:
            display_name, color = _SOURCE_META.get(it.source, (it.source, C.MUTED))
            st.markdown(
                f'<a href="{it.link}" target="_self" style="text-decoration:none;">'
                f'<span class="cs-pill" style="background:{color}1A;color:{color};">'
                f'<span class="cs-dot" style="background:{color};"></span>'
                f'{display_name}</span></a>'
                f' <span style="color:{C.MUTED};font-size:.8rem;">'
                f'{_md_escape(it.label)}</span>',
                unsafe_allow_html=True,
            )
            # Confidence badge: for PR-Predictor/Anomaly this is the source
            # low/medium/high string. For Open PR it's the raw merge
            # probability formatted as %%. In both cases it's what the
            # origin cached — never re-derived here.
            if it.confidence in ("low", "medium", "high"):
                st.markdown(confidence_pill(it.confidence),
                            unsafe_allow_html=True)
            else:
                # Open-PR probability string like "72%% merge"
                st.markdown(
                    f'<span class="cs-pill" style="background:{C.INFO}1A;'
                    f'color:{C.INFO};"><span class="cs-dot" '
                    f'style="background:{C.INFO};"></span>'
                    f'{it.confidence}</span>',
                    unsafe_allow_html=True,
                )
            if it.note:
                st.markdown(
                    f'<div style="color:{C.MUTED};font-size:.83rem;'
                    f'margin-top:6px;">{_md_escape(it.note)}</div>',
                    unsafe_allow_html=True,
                )
        with amt:
            sign = "+" if it.daily_delta_usd >= 0 else "−"
            col = C.BAD if it.daily_delta_usd >= 0 else C.GOOD
            st.markdown(
                f'<div style="text-align:right;">'
                f'<div class="cs-num" style="color:{col};font-weight:680;">'
                f'{sign}${abs(it.daily_delta_usd):,.2f}/day</div>'
                f'<div style="color:{C.MUTED};font-size:.75rem;">'
                f'({sign}${abs(it.daily_delta_usd) * 30:,.0f}/month)</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ---- On-demand: analyse open PRs for this account -------------------------

st.divider()
section(
    "Add open PR analysis",
    "Optional: run the deep AWS agent against this account's matching open "
    "PRs to price them before they merge. Same analyser the Dashboard uses. "
    "Results are cached — subsequent visits find them without re-analysing.",
    kicker="Live analysis",
)

# Open-PR scan cap. Default 12 (mid of the user-requested 10-15 range).
# The underlying `analyze_open_prs` defaults to 8; we override explicitly so
# behaviour is deterministic regardless of upstream defaults. A slider lets
# the user tune within 10-15 to trade coverage for wall-clock time (each
# PR is ~10-15s of deep-agent work).
_DEFAULT_OPEN_PR_SCAN_CAP = 12

scan_cols = st.columns([1, 2, 3], gap="medium", vertical_alignment="bottom")
with scan_cols[0]:
    scan_button = st.button(
        "Analyze open PRs now",
        help="~2-3 min for the default 12 PRs. Uses the deep agent.",
        use_container_width=True,
    )
with scan_cols[1]:
    open_pr_cap = st.slider(
        "Max PRs to analyse",
        min_value=10, max_value=15, value=_DEFAULT_OPEN_PR_SCAN_CAP,
        step=1, key="loop_open_pr_cap",
        help=(
            "Top-N open PRs by likely-to-merge-soon ranking. Higher = "
            "more coverage, longer wall-clock. Each PR is ~10-15s of "
            "deep-agent time."
        ),
    )
with scan_cols[2]:
    st.caption(
        f"Expected wall-clock: ~{open_pr_cap * 10}-{open_pr_cap * 15}s "
        f"(runs 4 in parallel; Bedrock rate-limits the concurrency)."
    )

if scan_button:
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
            f"No GitHub repos matched profile `{active.profile}` — nothing "
            "invented. Try running the analysis from the Dashboard tab, "
            "which uses the same repo-matcher.",
            tone="warning",
        )
    else:
        with st.spinner(f"Deep-analysing up to {open_pr_cap} open PRs "
                        f"on {len(repos_full)} repo(s)…"):
            try:
                from src.pr_scanner.open_prs import (
                    analyze_open_prs, list_open_prs_many,
                )
                open_prs = list_open_prs_many(repos_full)
                priced = (
                    analyze_open_prs(
                        open_prs,
                        profile=active.profile,
                        max_prs=open_pr_cap,
                    )
                    if open_prs else []
                )
                cached_state.set("loop_open_prs", (active.profile,), priced)
                st.rerun()
            except Exception as e:  # noqa: BLE001
                callout(f"Open PR analysis failed: {e}", tone="error")

st.caption(
    "Aggregation logic (mirrors PR Predictor): "
    "projected $/day = current $/day + Σ (cached recommendation $/day). "
    "No forecasting, no horizon multiplication, no confidence-weighted "
    "curves. If a source has nothing cached, its tile shows an em dash."
)
