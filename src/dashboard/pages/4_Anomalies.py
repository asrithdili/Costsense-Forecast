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
from src.dashboard.nav import inject_css, top_bar


st.set_page_config(page_title="CostSense · Anomalies", layout="wide",
                   page_icon="🚨")
inject_css()

st.title("Anomalies & Recommendations")
st.caption("Full-repo + full-AWS sweep. Ranked list of concrete cost-cutting "
           "actions with $/day savings and confidence, grounded in real "
           "AWS + GitHub data.")


# ---------- top control bar ----------

with st.spinner("Resolving profiles…"):
    profiles = [p for p in resolve_all() if p.account_id]
if not profiles:
    st.error("No AWS profiles reachable.")
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

picked_label = st.session_state.get("anom_profile", labels[0])
if picked_label not in labels:
    picked_label = labels[0]
picked_model_idx = st.session_state.get("anom_model_idx", 0)
if not (0 <= picked_model_idx < len(model_ids)):
    picked_model_idx = 0

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
    c1, c2 = st.columns([3, 3])
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

    c3, c4 = st.columns([2, 5])
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
        default_selection = _match(active_preview.profile, short_names) or short_names
        picked_short = st.multiselect(
            "Repos to scan", options=short_names,
            default=default_selection,
            key="anom_repos",
        )

    run_btn = st.button("Analyze", type="primary",
                        use_container_width=True)

active = profiles[labels.index(picked_label)]
model_id = model_ids[picked_model_idx]
selected_repos = [f"{gh_org}/{n}" for n in picked_short] if gh_org else []


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



