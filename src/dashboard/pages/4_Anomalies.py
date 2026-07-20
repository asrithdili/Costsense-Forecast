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
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from src.ai_agent.anomaly_agent import analyze_anomalies
from src.ai_agent.aws_sweep import sweep_account
from src.ai_agent.aws_sweep import sweep_to_summary as aws_summary
from src.ai_agent.repo_sweep import sweep_repos
from src.ai_agent.repo_sweep import sweep_to_summary as repo_summary
from src.aws.profiles import resolve_all
from src.pr_scanner.repos import gh_login, gh_orgs, repos_with_user_prs


st.set_page_config(page_title="CostSense · Anomalies", layout="wide",
                   page_icon="🚨")

st.title("Anomalies & Recommendations")
st.caption("Full-repo + full-AWS sweep. Ranked list of concrete cost-cutting "
           "actions with $/day savings and confidence, grounded in real "
           "AWS + GitHub data.")


# ---------- sidebar ----------

with st.sidebar:
    st.header("AWS account")
    profiles = [p for p in resolve_all() if p.account_id]
    if not profiles:
        st.error("No AWS profiles reachable.")
        st.stop()
    labels = [p.label for p in profiles]
    pick = st.selectbox("Profile", labels)
    active = profiles[labels.index(pick)]

    st.divider()
    st.header("Repos to scan")
    try:
        orgs = list(gh_orgs())
        _gh_user = gh_login()
    except Exception:  # noqa: BLE001
        orgs = []
        _gh_user = "?"
    # Default the org to DiligentCorp when it's in the user's org list.
    default_org_idx = 0
    if orgs:
        for i, o in enumerate(orgs):
            if o == "DiligentCorp":
                default_org_idx = i
                break
        gh_org = st.selectbox("GitHub org", orgs, index=default_org_idx)
    else:
        gh_org = st.text_input("GitHub org", value="DiligentCorp")

    try:
        suggested_full = list(repos_with_user_prs(gh_org)) if gh_org else []
    except Exception:  # noqa: BLE001
        suggested_full = []

    # Show only the short repo name in the multi-select; keep the full
    # `org/name` internally for the scanner.
    short_names = [r.split("/", 1)[-1] for r in suggested_full]

    # Default to the repo whose name matches the chosen AWS profile
    # (e.g. dil-data-platform-dev -> data-platform). Falls back to all
    # repos when no match (team / shared / control-tower profiles).
    from src.pr_scanner.profile_repo_match import match_repos as _match
    default_selection = _match(active.profile, short_names) or short_names

    picked_short = st.multiselect(
        "Repos", options=short_names, default=default_selection,
    )
    selected_repos = [f"{gh_org}/{n}" for n in picked_short] if gh_org else []

    st.divider()
    model_id = st.selectbox(
        "Bedrock model",
        index=1,
        options=[
            "anthropic.claude-3-haiku-20240307-v1:0",
            "us.anthropic.claude-sonnet-4-6",
        ],
    )

    st.divider()
    run_btn = st.button("Analyze", type="primary")


# ---------- session cache ----------

# Bump this whenever the Action / AnomalyReport schema changes so we don't
# render stale cached objects that are missing new fields (e.g. `approaches`).
_SCHEMA_VERSION = "v4-schema-guard"
report_key = (f"anom::{_SCHEMA_VERSION}::{active.profile}::"
              f"{','.join(sorted(selected_repos))}")
report = st.session_state.get(report_key)

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

# Every Analyze click nukes prior anomaly caches — prevents "half-populated
# from an earlier schema" artifacts. Also clears any siblings (aws/repo).
if run_btn:
    stale = [k for k in list(st.session_state.keys())
             if isinstance(k, str) and k.startswith("anom::")]
    for k in stale:
        del st.session_state[k]
    report = None
    with st.spinner("Sweeping AWS (~30s) — Cost Explorer, Compute Optimizer, "
                    "resource inventory…"):
        try:
            aws_raw = sweep_account(active.profile)
            aws_sum = aws_summary(aws_raw)
        except Exception as e:  # noqa: BLE001
            st.error(f"AWS sweep failed: {e}")
            st.code(traceback.format_exc())
            st.stop()
    with st.spinner(f"Sweeping {len(selected_repos)} repo(s) via GitHub…"):
        try:
            repo_raw = sweep_repos(selected_repos) if selected_repos else []
            repo_sum = repo_summary(repo_raw)
        except Exception as e:  # noqa: BLE001
            st.error(f"Repo sweep failed: {e}")
            st.code(traceback.format_exc())
            st.stop()
    with st.spinner("Analyzing with Claude…"):
        try:
            report = analyze_anomalies(
                aws_summary=aws_sum, repo_summary=repo_sum,
                profile=active.profile, model_id=model_id,
            )
            st.session_state[report_key] = report
            st.session_state[report_key + "::aws"] = aws_sum
            st.session_state[report_key + "::repo"] = repo_sum
        except Exception as e:  # noqa: BLE001
            st.error(f"anomaly agent failed: {e}")
            st.code(traceback.format_exc())


# ---------- render ----------

if report is None:
    st.info("Pick an AWS profile and repos in the sidebar, then click "
            "**Analyze**.")
    st.stop()

if report.error:
    st.error(f"Agent error: {report.error}")
    st.stop()

def _plain(text: str) -> str:
    """Escape dollar signs so Streamlit's markdown doesn't render $..$ as
    LaTeX math. Also strip leading/trailing whitespace."""
    if not text:
        return ""
    return text.replace("$", "\\$").strip()


# KPI row first — clean numbers at a glance.
kpis = st.columns(3)
kpis[0].metric("Recommended actions", len(report.actions))
kpis[1].metric("Potential savings / day",
               f"${report.total_daily_savings_usd:,.2f}")
kpis[2].metric("AWS tool calls (drill-down)", report.tool_calls)

# Summary rendered as a subtle info block instead of a competing header.
# This gives the page clear visual hierarchy:
#   Page title (h1)  →  KPI numbers  →  Summary blurb  →  Card list.
if report.summary:
    st.info(f"**Summary** — {_plain(report.summary)}")

st.divider()

# Category filter
if report.actions:
    cats = sorted({a.category or "other" for a in report.actions})
    picked_cats = st.multiselect(
        "Filter by category", cats, default=cats,
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

    for i, a in enumerate(filtered, start=1):
        conf_lower = (a.confidence or "medium").lower()
        conf_bg = {
            "high": "#16a34a",
            "medium": "#eab308",
            "low": "#6b7280",
        }.get(conf_lower, "#6b7280")
        cat_label = CATEGORY_LABEL.get(
            a.category or "other", (a.category or "Other").title()
        )
        with st.container(border=True):
            head_l, head_r = st.columns([5, 1])
            head_l.markdown(
                f"**#{i} · {cat_label}**"
                f"   ·  Save **\\${a.est_daily_savings_usd:,.2f}/day**"
            )
            head_r.markdown(
                f"<div style='text-align:right;'>"
                f"<span style='background:{conf_bg};color:white;"
                f"padding:2px 10px;border-radius:12px;"
                f"font-size:0.85em;font-weight:600;'>"
                f"{a.confidence.title()}"
                f"</span></div>",
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



