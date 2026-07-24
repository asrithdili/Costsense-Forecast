"""Anomalies — full repo + full AWS sweep with ranked cost-cutting actions.

Flow:
  1. Sidebar: pick AWS profile + one or more GitHub repos.
  2. On "Scan": pre-fetches AWS-wide state (idle resources, rightsizing recs,
     top services, budgets) AND scans each repo (recent IaC changes, open
     PRs, scheduled rules).
  3. Feeds both sweeps to Bedrock Claude, which returns a ranked list of
     concrete actions: resource, action verb, $/day savings, confidence,
     category, rationale.
"""
from __future__ import annotations

import sys
import time
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from src.ai_agent.anomaly_agent import analyze_anomalies, Approach
from src.ai_agent.pr_fix_agent import plan_pr_fix
from src.pr_scanner.gh_write import (
    apply_pr_plan,
    build_diff_preview,
    github_write_auth_status,
)
from src.ai_agent.aws_sweep import sweep_account
from src.ai_agent.aws_sweep import sweep_to_summary as aws_summary
from src.ai_agent.repo_sweep import sweep_repos
from src.ai_agent.repo_sweep import sweep_to_summary as repo_summary
from src.aws.profiles import resolve_all
from src.dashboard.costsense_theme import callout, confidence_pill, metric, money, section
from src.dashboard.notifications_ui import NotificationDraft, render_notification_button
from src.dashboard.state_cache import cached_state
from src.pr_scanner.repos import gh_login, gh_orgs, repos_with_user_prs
from src.dashboard.nav import (
    inject_css, render_sidebar_footer, render_sidebar_header, top_bar,
)

st.set_page_config(page_title="CostSense · Anomalies", layout="wide")
inject_css()
render_sidebar_header()  # Diligent card renders before any AWS calls

section(
    "Anomalies & Recommendations",
    "Full-repo + full-AWS sweep. Ranked list of concrete cost-cutting "
    "actions with $/day savings and confidence, grounded in real "
    "AWS + GitHub data.",
    kicker="Analysis",
)


# ---------- top control bar ----------

with st.spinner("Resolving profiles…"):
    profiles = [p for p in resolve_all() if p.account_id]
if not profiles:
    callout("No AWS profiles reachable.", tone="error")
    st.stop()
labels = [p.label for p in profiles]

try:
    orgs = list(gh_orgs())
    _gh_user = gh_login()
except Exception:  # noqa: BLE001
    orgs = []
    _gh_user = "?"

MODEL_OPTIONS = [
    ("us.anthropic.claude-sonnet-4-6",         "Claude Sonnet 4.6"),
    ("anthropic.claude-3-haiku-20240307-v1:0", "Claude 3 Haiku"),
]
model_ids = [mid for mid, _ in MODEL_OPTIONS]
model_labels = [name for _, name in MODEL_OPTIONS]

CONTINUOUS_FREQ_OPTIONS: dict[str, int] = {
    "3 min": 3 * 60,
    "15 min": 15 * 60,
    "30 min": 30 * 60,
    "1 hr": 60 * 60,
    "6 hr": 6 * 60 * 60,
    "12 hr": 12 * 60 * 60,
    "24 hr": 24 * 60 * 60,
}

# Restore page settings that must survive navigation away and back.
if "anom_continuous_enabled" not in st.session_state:
    st.session_state["anom_continuous_enabled"] = bool(
        st.session_state.get("anom_continuous_persist", False)
    )
if "anom_continuous_freq" not in st.session_state:
    st.session_state["anom_continuous_freq"] = st.session_state.get(
        "anom_continuous_freq_persist", "15 min"
    )

picked_label = st.session_state.get("anom_profile", labels[0])
if picked_label not in labels:
    picked_label = labels[0]
picked_model_idx = st.session_state.get("anom_model_idx", 0)
if not (0 <= picked_model_idx < len(model_ids)):
    picked_model_idx = 0

# Detect account change and bump a widget-version counter so the
# repos multiselect re-instantiates with a fresh default (see
# `_widget_ver` — used to salt the widget key below).
_new_profile = profiles[labels.index(picked_label)].profile
_last_profile = st.session_state.get("anom_last_profile")
if _last_profile != _new_profile:
    st.session_state["anom_last_profile"] = _new_profile
    st.session_state["anom_widget_ver"] = (
        st.session_state.get("anom_widget_ver", 0) + 1
    )
_widget_ver = st.session_state.get("anom_widget_ver", 0)

header = (f"Controls  ·  Account: {picked_label}  ·  "
          f"Model: {model_labels[picked_model_idx]}")

# Compute default repo selection BEFORE the expander body renders so the
# multiselect widget has the right default when it first appears.
active_preview = profiles[labels.index(picked_label)]
if orgs:
    default_org_idx = 0
    for i, o in enumerate(orgs):
        if o == "DiligentCorp":
            default_org_idx = i
            break
    org_preview = st.session_state.get("anom_gh_org", orgs[default_org_idx])
else:
    org_preview = st.session_state.get("anom_gh_org_text", "DiligentCorp")

try:
    suggested_full_preview = list(repos_with_user_prs(org_preview)) if org_preview else []
except Exception:  # noqa: BLE001
    suggested_full_preview = []
short_names_preview = [r.split("/", 1)[-1] for r in suggested_full_preview]

from src.pr_scanner.profile_repo_match import match_repos as _match
default_repos = _match(active_preview.profile, short_names_preview) or short_names_preview

with top_bar(header):
    c1, c2 = st.columns([3, 3], gap="medium", vertical_alignment="bottom")
    with c1:
        picked_label = st.selectbox(
            "Account", labels,
            index=labels.index(picked_label),
            key="anom_profile",
        )
    with c2:
        picked_model_idx = st.selectbox(
            "Bedrock model", range(len(model_labels)),
            index=picked_model_idx,
            format_func=lambda i: model_labels[i],
            key="anom_model_idx",
        )

    c3, c4 = st.columns([2, 5], gap="medium", vertical_alignment="bottom")
    with c3:
        if orgs:
            gh_org = st.selectbox(
                "GitHub org", orgs,
                index=orgs.index(org_preview) if org_preview in orgs else 0,
                key="anom_gh_org",
            )
        else:
            gh_org = st.text_input(
                "GitHub org", value=org_preview,
                key="anom_gh_org_text",
            )
    with c4:
        try:
            suggested_full = list(repos_with_user_prs(gh_org)) if gh_org else []
        except Exception:  # noqa: BLE001
            suggested_full = []
        short_names = [r.split("/", 1)[-1] for r in suggested_full]
        # Profile-matched repos only. When no repo maps to this profile
        # (common on team/sandbox accounts like dil-team-hackfest that
        # aren't the primary deploy target for any repo), default to
        # NO repos rather than every repo in the org. That prevents an
        # account-level anomaly scan from silently pulling in unrelated
        # repos' PRs — which caused bogus DIA-* recommendations to
        # appear on hackfest deep-links.
        default_selection = _match(active_preview.profile, short_names) or []
        _persisted_short = st.session_state.get("anom_repos_persist", [])
        short_names_merged = list(dict.fromkeys(
            short_names + [r for r in _persisted_short if r not in short_names]
        ))
        _repos_widget_key = f"anom_repos_v{_widget_ver}"
        if _repos_widget_key not in st.session_state and _persisted_short:
            restored = [r for r in _persisted_short if r in short_names_merged]
            if restored:
                st.session_state[_repos_widget_key] = restored
        # Key salted by widget-version counter — bumped on account
        # change so `default=` takes effect for the new profile.
        picked_short = st.multiselect(
            "Repos to scan", options=short_names_merged,
            default=default_selection,
            key=_repos_widget_key,
            help="Empty is valid — the scan will focus on AWS-side "
                 "anomalies for the selected account only.",
        )
        st.session_state["anom_repos_persist"] = list(picked_short)

    c_cont1, c_cont2 = st.columns([3, 3], gap="medium", vertical_alignment="bottom")
    with c_cont1:
        continuous_on = st.toggle(
            "Continuous analysis",
            key="anom_continuous_enabled",
            help="Re-run analysis on the chosen schedule while this page is open.",
        )
    with c_cont2:
        st.selectbox(
            "Frequency",
            options=list(CONTINUOUS_FREQ_OPTIONS),
            key="anom_continuous_freq",
            disabled=not continuous_on,
        )

    run_btn = st.button("Analyze", type="primary",
                        use_container_width=True)

st.session_state["anom_continuous_persist"] = bool(
    st.session_state.get("anom_continuous_enabled", False)
)
st.session_state["anom_continuous_freq_persist"] = st.session_state.get(
    "anom_continuous_freq", "15 min"
)
continuous_on = bool(st.session_state.get("anom_continuous_enabled", False))

active = profiles[labels.index(picked_label)]
model_id = model_ids[picked_model_idx]
selected_repos = [f"{gh_org}/{n}" for n in picked_short] if gh_org else []
# Fallback to persisted list ONLY when this isn't an account-only
# deep-link (from Org's View anomalies). Otherwise carrying over the
# previous manual scan's repos pollutes the report with unrelated PRs.
if not selected_repos and not st.session_state.get(
    "anom_autorun_account_only_if_no_match"
):
    selected_repos = list(
        st.session_state.get("anom_selected_repos_persist", [])
    )
st.session_state["anom_selected_repos_persist"] = list(selected_repos)

with st.sidebar:
    render_sidebar_footer(
        active_profile=active.profile,
        account_id=active.account_id,
        extra_rows=[
            ("Model", model_labels[picked_model_idx]),
            ("Repos", f"{len(selected_repos)} selected"),
        ],
    )


# ---------- session cache ----------

# A fresh page run never has an active scan yet. Clear stale flags left when
# a prior run was interrupted (e.g. user navigated away mid-scan).
st.session_state["anom_scan_in_progress"] = False

# Bump this whenever the Action / AnomalyReport schema changes so we don't
# render stale cached objects that are missing new fields (e.g. `approaches`).
_SCHEMA_VERSION = "v4-schema-guard"
report_key = (f"anom::{_SCHEMA_VERSION}::{active.profile}::"
              f"{','.join(sorted(selected_repos))}")

# Disk-backed cache — restores the last report the user ran across tab
# switches, browser reloads, and server restarts. Identity is (schema,
# profile, sorted repos) so any change to those forces a re-analyze.
_anom_identity = (_SCHEMA_VERSION, active.profile,
                   tuple(sorted(selected_repos)))
config_key = (
    f"{active.profile}::{model_id}::{','.join(sorted(selected_repos))}"
)
report = st.session_state.get(report_key)
if report is None:
    # Try the disk-backed cache first, THEN fall back to the last-run
    # config match. Disk wins because it survives things session_state
    # can't (browser restart, server restart, tab crash).
    report = cached_state.get("anom_report", _anom_identity)
    if report is None:
        last_key = st.session_state.get("anom_last_report_key")
        if (
            last_key
            and st.session_state.get("anom_last_config_key") == config_key
        ):
            report = st.session_state.get(last_key)
if report is not None:
    # Mirror onto session_state so downstream code that still reads
    # `st.session_state[report_key]` (PR-plan helpers etc) keeps working
    # even when we hydrated `report` from the disk cache above.
    st.session_state[report_key] = report

freq_label = st.session_state.get("anom_continuous_freq", "15 min")
if freq_label not in CONTINUOUS_FREQ_OPTIONS:
    freq_label = "15 min"
interval_sec = CONTINUOUS_FREQ_OPTIONS[freq_label]


def _format_countdown(seconds: int) -> str:
    """Live countdown: under 1h → ``2m59s``; 1h+ → ``1:14:59``."""
    seconds = max(0, int(seconds))
    if seconds < 3600:
        mins, secs = divmod(seconds, 60)
        if mins:
            return f"{mins}m{secs:02d}s"
        return f"{secs}s"
    hours, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    return f"{hours}:{mins:02d}:{secs:02d}"


def _continuous_config_changed() -> bool:
    stored = st.session_state.get("anom_continuous_config_key")
    return stored is not None and stored != config_key


def _continuous_interval_elapsed() -> bool:
    last_run = st.session_state.get("anom_continuous_last_run")
    if last_run is None:
        return False
    return (time.time() - last_run) >= interval_sec


def _clear_anomaly_caches() -> None:
    stale = [k for k in list(st.session_state.keys())
             if isinstance(k, str) and (
                 k.startswith("anom::")
                 or "::pr_plan::" in k
                 or "::pr_result::" in k
             )]
    for k in stale:
        del st.session_state[k]


def _run_anomaly_scan(*, clear_caches: bool = False) -> object | None:
    if clear_caches:
        _clear_anomaly_caches()

    st.session_state["anom_scan_in_progress"] = True
    st.session_state["anom_scan_started_at"] = time.time()
    try:
        with st.spinner("Sweeping AWS (~30s) — Cost Explorer, Compute Optimizer, "
                        "resource inventory…"):
            try:
                aws_raw = sweep_account(active.profile)
                aws_sum = aws_summary(aws_raw)
            except Exception as e:  # noqa: BLE001
                callout(f"AWS sweep failed: {e}", tone="error")
                st.code(traceback.format_exc())
                return None
        with st.spinner(f"Sweeping {len(selected_repos)} repo(s) via GitHub…"):
            try:
                repo_raw = sweep_repos(selected_repos) if selected_repos else []
                repo_sum = repo_summary(repo_raw)
            except Exception as e:  # noqa: BLE001
                callout(f"Repo sweep failed: {e}", tone="error")
                st.code(traceback.format_exc())
                return None
        with st.spinner("Analyzing with Claude…"):
            try:
                scanned = analyze_anomalies(
                    aws_summary=aws_sum, repo_summary=repo_sum,
                    profile=active.profile, model_id=model_id,
                )
                st.session_state[report_key] = scanned
                st.session_state[report_key + "::aws"] = aws_sum
                st.session_state[report_key + "::repo"] = repo_sum
                st.session_state["anom_continuous_last_run"] = time.time()
                st.session_state["anom_continuous_config_key"] = config_key
                st.session_state["anom_last_report_key"] = report_key
                st.session_state["anom_last_config_key"] = config_key
                st.session_state["anom_selected_repos_persist"] = list(selected_repos)
                st.session_state["anom_repos_persist"] = [
                    r.split("/", 1)[-1] for r in selected_repos
                ]
                # Also persist to disk so browser reload / server restart
                # restores the report instead of forcing a re-scan.
                cached_state.set("anom_report", _anom_identity, scanned)
                return scanned
            except Exception as e:  # noqa: BLE001
                callout(f"anomaly agent failed: {e}", tone="error")
                st.code(traceback.format_exc())
                return None
    finally:
        st.session_state["anom_scan_in_progress"] = False


# Cross-tab hardening: if the cached report has ANY action missing the
# `approaches` field, treat it as stale and discard it. This catches the
# case where Streamlit rerendered a session from before the module was
# reimported with the new dataclass shape.
def _is_stale(rep) -> bool:
    if rep is None or not getattr(rep, "actions", None):
        return False
    for a in rep.actions:
        aps = getattr(a, "approaches", None)
        if aps is None or not isinstance(aps, list):
            return True
    return False

if _is_stale(report):
    # Purge and force the user to click Analyze — safer than silently rendering
    # nothing.
    for k in [k for k in list(st.session_state.keys())
              if isinstance(k, str) and k.startswith("anom::")]:
        del st.session_state[k]
    report = None

# Deep-link auto-run: another page (Org-Level Impact) can drop the user
# here with `anom_autorun=True` in session state after setting the
# account context via `anom_profile`. We treat that as a synthetic click
# on Analyze, then clear the flag so subsequent reruns don't loop.
if st.session_state.pop("anom_autorun", False):
    run_btn = True
    # Also consume the companion "account-only if no repo match" flag
    # so it doesn't affect the next non-deep-link scan the user runs.
    st.session_state.pop("anom_autorun_account_only_if_no_match", None)

# Don't re-run immediately when continuous is turned on and results already
# exist, or when resuming a session that already has a cached report.
_was_continuous = st.session_state.get("_anom_continuous_prev", False)
if continuous_on and not _was_continuous and report is not None:
    st.session_state["anom_continuous_last_run"] = time.time()
    st.session_state["anom_continuous_config_key"] = config_key
elif (
    continuous_on
    and report is not None
    and st.session_state.get("anom_continuous_last_run") is None
):
    st.session_state["anom_continuous_last_run"] = time.time()
    st.session_state["anom_continuous_config_key"] = config_key
st.session_state["_anom_continuous_prev"] = continuous_on

should_auto_run = (
    continuous_on
    and not run_btn
    and not st.session_state.get("anom_scan_in_progress")
    and (
        _continuous_config_changed()
        or (report is None and st.session_state.get("anom_continuous_last_run") is None)
        or _continuous_interval_elapsed()
    )
)

# Every Analyze click nukes prior anomaly caches — prevents "half-populated
# from an earlier schema" artifacts. Also clears any siblings (aws/repo).
if run_btn:
    if st.session_state.get("anom_scan_in_progress"):
        callout("A scan is already running. Please wait for it to finish.", tone="info")
    else:
        _clear_anomaly_caches()
        report = _run_anomaly_scan()
elif should_auto_run:
    new_report = _run_anomaly_scan()
    if new_report is not None:
        report = new_report
    elif report is None:
        report = st.session_state.get(report_key)

if continuous_on:

    @st.fragment(run_every=1)
    def _continuous_status_tick() -> None:
        if not st.session_state.get("anom_continuous_enabled"):
            return
        if st.session_state.get("anom_scan_in_progress"):
            st.caption("Continuous analysis · scan in progress…")
            return

        last_run = st.session_state.get("anom_continuous_last_run")
        if last_run is None:
            st.caption("Continuous analysis enabled · preparing first scan…")
            return

        tick_freq = st.session_state.get("anom_continuous_freq", "15 min")
        tick_interval = CONTINUOUS_FREQ_OPTIONS.get(tick_freq, 15 * 60)
        last_txt = datetime.fromtimestamp(last_run).strftime("%H:%M:%S")
        remaining = int((last_run + tick_interval) - time.time())

        if remaining > 0:
            st.caption(
                f"Continuous analysis · last run {last_txt} · "
                f"next run in {_format_countdown(remaining)}"
            )
            return

        st.caption(
            f"Continuous analysis · last run {last_txt} · "
            "starting next scan…"
        )
        st.rerun()

    _continuous_status_tick()


# ---------- render ----------

if report is None:
    if st.session_state.get("anom_scan_in_progress"):
        callout("Analysis in progress. Results will appear when the scan finishes.", tone="info")
    elif continuous_on:
        callout(
            "Continuous analysis is enabled. The first scan will start "
            "automatically — or click **Analyze** to run now.",
            tone="info",
        )
    else:
        callout(
            "Pick an AWS profile and repos in the sidebar, then click "
            "**Analyze**.",
            tone="info",
        )
    st.stop()

if report.error:
    callout(f"Agent error: {report.error}", tone="error")
    st.stop()


def _plain(text: str) -> str:
    """Escape dollar signs so Streamlit's markdown doesn't render $..$ as
    LaTeX math. Also strip leading/trailing whitespace."""
    if not text:
        return ""
    return text.replace("$", "\\$").strip()


PR_ELIGIBLE_LANGUAGES = frozenset({
    "terraform", "hcl", "typescript", "javascript", "python",
    "yaml", "yml", "json",
})


def _approach_pr_eligible(ap: Approach) -> bool:
    code = (getattr(ap, "code", "") or "").strip()
    lang = (getattr(ap, "language", "") or "").strip().lower()
    if lang == "yml":
        lang = "yaml"
    return bool(code) and lang in PR_ELIGIBLE_LANGUAGES


def _pr_plan_key(report_key: str, action_idx: int) -> str:
    return f"{report_key}::pr_plan::{action_idx}"


def _pr_result_key(report_key: str, action_idx: int) -> str:
    return f"{report_key}::pr_result::{action_idx}"


section(
    "Scan results",
    "Ranked actions from the latest AWS and repository sweep.",
    kicker="Results",
)

# KPI row first — clean numbers at a glance.
savings = report.total_daily_savings_usd
savings_display = (
    money(savings) if savings >= 1_000 else f"${savings:,.2f}"
)
kpis = st.columns(3, gap="medium")
with kpis[0]:
    metric("Recommended actions", len(report.actions))
with kpis[1]:
    metric(
        "Potential savings / day",
        savings_display,
        delta="recurring",
        good=True,
    )
with kpis[2]:
    metric("AWS tool calls (drill-down)", report.tool_calls)

# Summary — theme-consistent bordered card instead of heavy st.info banner.
if report.summary:
    with st.container(border=True):
        section("Summary", _plain(report.summary), kicker="Overview")

_ANOM_MIN_SAVINGS = 10.0
_top_high_action = None
for _a in report.actions:
    if (_a.confidence or "").lower() != "high":
        continue
    if (_a.est_daily_savings_usd or 0) < _ANOM_MIN_SAVINGS:
        continue
    if (
        _top_high_action is None
        or _a.est_daily_savings_usd > _top_high_action.est_daily_savings_usd
    ):
        _top_high_action = _a
if _top_high_action is not None:
    _save = _top_high_action.est_daily_savings_usd
    _save_txt = money(_save) if _save >= 1_000 else f"${_save:,.2f}"
    render_notification_button(
        button_label="Notify anomaly",
        state_key=report_key,
        draft=NotificationDraft(
            title="High-confidence cost anomaly",
            severity="High",
            reason=(
                f"High-confidence recommendation with {_save_txt}/day potential "
                f"savings: {_plain(_top_high_action.issue) or 'see scan results'}."
            ),
            recipient="finops-team@example.com",
            subject=(
                f"[CostSense] Anomaly — {_save_txt}/day savings "
                f"({active.profile})"
            ),
            body=(
                f"CostSense Anomalies scan found a high-confidence action.\n\n"
                f"Account: {active.profile}\n"
                f"Issue: {_plain(_top_high_action.issue)}\n"
                f"Reason: {_plain(_top_high_action.reason)}\n"
                f"Recommendation: {_plain(_top_high_action.recommendation)}\n"
                f"Potential savings: {_save_txt}/day\n"
                f"Total scan savings: {savings_display}/day\n\n"
                f"Please review the ranked actions in CostSense Anomalies."
            ),
            source_page="Anomalies",
            source_type="high_confidence_anomaly",
        ),
    )

st.divider()

# Category filter
if report.actions:
    section(
        "Filter actions",
        "Narrow the list by recommendation category.",
        kicker="Filters",
    )
    cats = sorted({a.category or "other" for a in report.actions})
    picked_cats = st.multiselect(
        "Filter by category", cats, default=cats,
        label_visibility="collapsed",
    )
    filtered = [a for a in report.actions
                if (a.category or "other") in picked_cats]

    # Friendly labels for the category codes.
    CATEGORY_LABEL = {
        "idle": "Idle resource",
        "oversized": "Oversized resource",
        "log-inefficiency": "Log/metric inefficiency",
        "missing-lifecycle": "Missing lifecycle policy",
        "risky-upcoming-pr": "Risky upcoming PR",
        "trending-up": "Cost trending up",
        "other": "Other",
    }

    section(
        "Recommended actions",
        f"{len(filtered)} action(s) shown"
        + (f" of {len(report.actions)} total." if len(filtered) != len(report.actions)
           else "."),
        kicker="Actions",
    )

    for display_i, a in enumerate(filtered, start=1):
        action_idx = report.actions.index(a)
        cat_label = CATEGORY_LABEL.get(
            a.category or "other", (a.category or "Other").title()
        )
        save_amt = a.est_daily_savings_usd
        save_txt = money(save_amt) if save_amt >= 1_000 else f"${save_amt:,.2f}"
        with st.container(border=True):
            head_l, head_r = st.columns([5, 1], gap="medium")
            with head_l:
                st.markdown(
                    f"**#{display_i} · {cat_label}**"
                    f"   ·   Save **{_plain(save_txt)}/day**"
                )
            with head_r:
                st.markdown(
                    f"<div style='text-align:right;'>"
                    f"{confidence_pill(a.confidence)}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"- **Issue:** {_plain(a.issue) or '_—_'}\n"
                f"- **Reason:** {_plain(a.reason) or '_—_'}\n"
                f"- **Recommendation:** {_plain(a.recommendation) or '_—_'}"
            )

            # Single-approach mode: pick the first LLM approach that has a
            # distinct title (not just repeating the recommendation). If none
            # qualifies, skip the "Ways to fix it" section entirely — the
            # 3-bullet Issue/Reason/Recommendation above already tells the
            # user what to do.
            approaches = getattr(a, "approaches", None) or []
            best = None
            rec_lower = (a.recommendation or "").strip().lower()
            for ap in approaches:
                title = (getattr(ap, "title", "") or "").strip()
                desc = (getattr(ap, "description", "") or "").strip()
                if not title and not desc:
                    continue
                # Skip approaches that are just the recommendation restated
                if title.lower() == "recommended fix":
                    continue
                if desc.lower() == rec_lower:
                    continue
                best = ap
                break

            if best is not None:
                st.markdown("**How to fix it:**")
                title = getattr(best, "title", "") or "Fix"
                desc = getattr(best, "description", "") or ""
                code = getattr(best, "code", "") or ""
                lang = getattr(best, "language", "") or "text"
                with st.expander(f"{_plain(title)}", expanded=True):
                    if desc:
                        st.markdown(_plain(desc))
                    if code:
                        st.code(code, language=lang)

                # --- Draft PR flow (gated: IaC code + scanned repos) ---
                if _approach_pr_eligible(best) and selected_repos:
                    st.caption(
                        f"Draft PRs are limited to the {len(selected_repos)} "
                        f"repo(s) selected for this scan on `{active.profile}`."
                    )
                    gh_ready, gh_msg = github_write_auth_status()
                    plan_key = _pr_plan_key(report_key, action_idx)
                    result_key = _pr_result_key(report_key, action_idx)
                    pr_result = st.session_state.get(result_key)

                    if pr_result:
                        st.success(f"Draft PR opened: {pr_result}")
                    else:
                        if not gh_ready:
                            st.caption(f"GitHub write: {gh_msg}")

                        plan = st.session_state.get(plan_key)
                        prepare_key = f"anom_prepare_pr_{report_key}_{action_idx}"
                        confirm_key = f"anom_confirm_pr_{report_key}_{action_idx}"
                        cancel_key = f"anom_cancel_pr_{report_key}_{action_idx}"

                        if st.button(
                            "Prepare draft PR",
                            key=prepare_key,
                            disabled=not gh_ready,
                            type="secondary",
                        ):
                            with st.spinner(
                                "Locating file in repo and building preview…"
                            ):
                                try:
                                    plan = plan_pr_fix(
                                        action=a,
                                        approach=best,
                                        allowed_repos=selected_repos,
                                        model_id=model_id,
                                    )
                                    st.session_state[plan_key] = plan
                                except Exception as e:  # noqa: BLE001
                                    callout(f"PR planning failed: {e}", tone="error")
                                    st.code(traceback.format_exc())

                        plan = st.session_state.get(plan_key)
                        if plan is not None:
                            if plan.error:
                                callout(
                                    f"Could not prepare PR: {plan.error}",
                                    tone="warning",
                                )
                            elif plan.repo and plan.files:
                                if plan.repo not in selected_repos:
                                    callout(
                                        f"PR targets `{plan.repo}` which is not in "
                                        "the repos selected for this scan.",
                                        tone="error",
                                    )
                                else:
                                    st.markdown("**PR preview**")
                                    st.markdown(
                                        f"- **Repo:** `{plan.repo}`\n"
                                        f"- **Branch:** `{plan.branch}`\n"
                                        f"- **Title:** {_plain(plan.title)}"
                                    )
                                    try:
                                        diff_text = build_diff_preview(
                                            plan.repo,
                                            [{"path": f.path, "content": f.content}
                                             for f in plan.files],
                                        )
                                    except Exception as e:  # noqa: BLE001
                                        diff_text = f"(diff preview failed: {e})"
                                    st.code(diff_text or "(no changes)", language="diff")

                                    btn_l, btn_r = st.columns(2)
                                    with btn_l:
                                        if st.button(
                                            "Open draft PR",
                                            key=confirm_key,
                                            type="primary",
                                        ):
                                            if plan.repo not in selected_repos:
                                                callout(
                                                    f"PR targets `{plan.repo}` which is "
                                                    "not in the repos selected for this scan.",
                                                    tone="error",
                                                )
                                            else:
                                                with st.spinner(
                                                    "Pushing branch and opening draft PR…"
                                                ):
                                                    try:
                                                        url = apply_pr_plan(
                                                            repo=plan.repo,
                                                            branch=plan.branch,
                                                            title=plan.title,
                                                            body=plan.body,
                                                            files=[
                                                                {"path": f.path,
                                                                 "content": f.content}
                                                                for f in plan.files
                                                            ],
                                                        )
                                                        st.session_state[result_key] = url
                                                        del st.session_state[plan_key]
                                                        st.rerun()
                                                    except Exception as e:  # noqa: BLE001
                                                        callout(
                                                            f"Failed to open draft PR: {e}",
                                                            tone="error",
                                                        )
                                                        st.code(traceback.format_exc())
                                    with btn_r:
                                        if st.button("Discard preview", key=cancel_key):
                                            del st.session_state[plan_key]
                                            st.rerun()
