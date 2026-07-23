"""CostSense — shared design system.

One import to give every widget the same visual language. Fixes three things
that currently drift across widgets:

  1. Colour chaos — three greens, a red doing triple duty, the default Office
     chart palette. Everything now comes from ONE token table below.
  2. Delta semantics — cost-up-is-bad vs savings-are-good was applied
     inconsistently. ``metric()`` takes an explicit ``good=`` flag so the
     colour always matches the meaning.
  3. Emoji carrying the hierarchy — headings now have real typographic
     structure (kicker + title + subtitle), so emoji become optional garnish,
     not the load-bearing wall.

Usage (top of a page, once)::

    from src.dashboard.costsense_theme import inject_css
    inject_css()

Then anywhere::

    from src.dashboard.costsense_theme import section, metric, pill, money, C
    section("Cost impact", "Live burn rate vs. daily threshold")
    metric("Monthly savings", money(50_000), delta="-49%", good=True)
    st.markdown(pill("Critical"), unsafe_allow_html=True)

Note: a couple of the CSS selectors target Streamlit's internal test-ids
(e.g. stMetric). Those are stable across recent 1.3x–1.4x releases but can
move between major versions — if a rule stops biting after an upgrade, the
fix is almost always a renamed data-testid here, nothing structural.
"""
from __future__ import annotations

from string import Template

import streamlit as st


# ============================================================================
# TOKENS — the single source of truth for colour. Import as C.INK, C.SEV, etc.
# ============================================================================
class C:
    # neutrals
    INK = "#14181F"       # primary text
    MUTED = "#5A6472"     # secondary text / captions (passes contrast on canvas)
    FAINT = "#8A93A2"     # tertiary / axis labels
    HAIRLINE = "#E6E8EC"  # borders, dividers
    CANVAS = "#F6F7F9"    # app background
    CARD = "#FFFFFF"      # surfaces
    NEUTRAL_SOFT = "#F1F3F5"  # neutral pill / badge fills

    # brand
    BRAND = "#0C7C74"       # teal — primary accent, key numbers, active nav
    BRAND_DARK = "#0A5F59"  # hover / pressed
    BRAND_SOFT = "#E1F1EF"  # tinted fills, selected rows
    INFO = "#3B5BDB"        # secondary accent (links to "expected"/baseline)

    # semantic deltas — meaning, not vibe
    GOOD = "#2B8A3E"  # savings, reductions, "after fix"
    BAD = "#C92A2A"   # cost increases, breaches

    # severity ramp (use ONLY for severity; keeps red meaningful)
    SEV = {
        "Low": "#2F9E44",
        "Medium": "#F59F00",
        "High": "#E8590C",
        "Critical": "#C92A2A",
    }
    SEV_SOFT = {
        "Low": "#EBFBEE",
        "Medium": "#FFF4E0",
        "High": "#FFEEE3",
        "Critical": "#FCEBEB",
    }

    # status (for the Diligent-wide map: Deployed / Pilot / Backlog)
    STATUS = {
        "Deployed": "#2B8A3E",
        "Pilot": "#E8590C",
        "Backlog": "#8A93A2",
    }


def plotly_layout(height: int = 400) -> dict:
    """Consistent Plotly layout dict. ``fig.update_layout(**plotly_layout())``."""
    return dict(
        height=height,
        template="plotly_white",
        margin=dict(l=16, r=24, t=28, b=16),
        font=dict(
            family="Inter, -apple-system, Segoe UI, Roboto, sans-serif",
            color=C.INK,
            size=13,
        ),
        colorway=[C.BRAND, C.INFO, C.SEV["Medium"], C.SEV["High"], C.FAINT],
        xaxis=dict(gridcolor=C.HAIRLINE, zerolinecolor=C.HAIRLINE),
        yaxis=dict(gridcolor=C.HAIRLINE, zerolinecolor=C.HAIRLINE),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=C.MUTED),
        ),
        # Soft card-white tooltip with a hairline border. The old
        # bgcolor=INK / white-text combo looked like a black overlay
        # slap on pale charts; this reads as a floating card that
        # belongs to the same visual system as everything else.
        hoverlabel=dict(
            bgcolor=C.CARD,
            bordercolor=C.HAIRLINE,
            font=dict(color=C.INK, size=12,
                      family="Inter, -apple-system, Segoe UI, Roboto, sans-serif"),
            align="left",
        ),
    )


# ============================================================================
# GLOBAL CSS — call inject_css() once at app start
# ============================================================================
_CSS = Template("""
<style>
:root{
--ink:$INK; --muted:$MUTED; --faint:$FAINT; --line:$HAIRLINE;
--canvas:$CANVAS; --card:$CARD; --brand:$BRAND; --brand-dark:$BRAND_DARK;
--brand-soft:$BRAND_SOFT;
}

/* ---- Type: real font stack + tabular numerals everywhere numbers live ---- */
html, body, [class*="css"]{
  font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
}
h1,h2,h3,h4{ color:var(--ink); letter-spacing:-0.01em; font-weight:650; }
h1{ font-size:1.55rem; } h2{ font-size:1.2rem; } h3{ font-size:1.02rem; }

/* Tighten the default top padding so the app doesn't open with a void */
.block-container{ padding-top:2.2rem; padding-bottom:3rem; max-width:1180px; }

/* ---- Metrics: turn the bare Streamlit metric into a real card ---- */
[data-testid="stMetric"]{
  background:var(--card); border:1px solid var(--line); border-radius:12px;
  padding:14px 16px;
}
[data-testid="stMetricLabel"] p{
  color:var(--muted); font-size:0.72rem; font-weight:600;
  text-transform:uppercase; letter-spacing:0.04em;
}
[data-testid="stMetricValue"]{
  color:var(--ink); font-weight:680; font-variant-numeric:tabular-nums;
  letter-spacing:-0.01em;
}
[data-testid="stMetricDelta"]{ font-variant-numeric:tabular-nums; font-weight:600; }

/* ---- Bordered containers -> soft cards ---- */
[data-testid="stVerticalBlockBorderWrapper"]{
  border-color:var(--line)!important; border-radius:14px!important;
  box-shadow:0 1px 2px rgba(20,24,31,0.04);
}

/* ---- Primary buttons: brand, not alarm-red default ---- */
.stButton>button[kind="primary"]{
  background:var(--brand); border:1px solid var(--brand);
  color:#fff; font-weight:600; border-radius:10px;
}
.stButton>button[kind="primary"]:hover{
  background:var(--brand-dark); border-color:var(--brand-dark);
}
.stButton>button[kind="secondary"]{
  border-radius:10px; border:1px solid var(--line); font-weight:550;
  color:var(--ink);
}
.stButton>button[kind="secondary"]:hover{
  border-color:var(--brand); color:var(--brand);
}

/* ---- Sidebar: hairline separation, calmer ---- */
[data-testid="stSidebar"]{ border-right:1px solid var(--line); }

/* ---- Chat bubbles: lose the grey blocks, use hairline cards ---- */
[data-testid="stChatMessage"]{
  background:var(--card); border:1px solid var(--line); border-radius:12px;
}

/* ---- Markdown body: consistent sizing for AI output and chat ---- */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
  font-size:1rem !important;
  line-height:1.6 !important;
  font-style:normal !important;
}

/* ---- Custom components: section header, kicker, severity pill ---- */
.cs-section{ margin:0.2rem 0 0.9rem 0; }
.cs-kicker{
  color:var(--brand); font-size:0.72rem; font-weight:700;
  text-transform:uppercase; letter-spacing:0.08em; margin-bottom:2px;
}
.cs-title{ color:var(--ink); font-size:1.22rem; font-weight:680;
  letter-spacing:-0.01em; line-height:1.2; }
.cs-sub{ color:var(--muted); font-size:0.9rem; margin-top:3px; }

.cs-pill{
  display:inline-flex; align-items:center; gap:6px; padding:3px 10px;
  border-radius:999px; font-size:0.78rem; font-weight:600; line-height:1;
}
.cs-dot{ width:7px; height:7px; border-radius:50%; display:inline-block; }

.cs-num{ font-variant-numeric:tabular-nums; letter-spacing:-0.01em; }

/* Card used by metric() helper */
.cs-card{
  background:var(--card); border:1px solid var(--line); border-radius:12px;
  padding:14px 16px;
  min-height:104px; /* keep tile rows visually aligned even when one has no delta */
  display:flex; flex-direction:column; justify-content:center;
  transition:border-color 120ms ease, box-shadow 120ms ease;
}
.cs-card:hover{
  border-color:var(--brand-soft); box-shadow:0 1px 2px rgba(20,24,31,0.04);
}
.cs-card .lbl{ color:var(--muted); font-size:0.72rem; font-weight:600;
  text-transform:uppercase; letter-spacing:0.04em;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.cs-card .val{ color:var(--ink); font-size:1.55rem; font-weight:680;
  font-variant-numeric:tabular-nums; letter-spacing:-0.01em; margin-top:4px;
  line-height:1.15;
}
.cs-card .dlt{ font-size:0.82rem; font-weight:600; margin-top:6px;
  font-variant-numeric:tabular-nums;
}

/* Metadata strip row (Basis / Confidence style) — used under tile grids
   for grounding transparency. Consistent look across pages. */
.cs-meta-row{
  display:flex; flex-wrap:wrap; align-items:center; gap:14px 24px;
  padding:10px 14px; margin:8px 0 4px 0;
  border:1px solid var(--line); border-radius:10px;
  background:var(--card);
}
.cs-meta-row .cs-meta-item{
  display:inline-flex; align-items:center; gap:8px;
}
.cs-meta-row .cs-meta-label{
  color:var(--muted); font-size:0.72rem; font-weight:600;
  text-transform:uppercase; letter-spacing:0.04em;
}
.cs-meta-row .cs-meta-text{
  color:var(--ink); font-size:0.9rem;
}

/* Tighten the built-in st.caption spacing so it sits close to the tile
   row it annotates instead of drifting into the next block. */
[data-testid="stCaptionContainer"]{
  margin-top:-4px; margin-bottom:12px;
}
</style>
""")


def inject_css() -> None:
    """Call once, right after ``st.set_page_config()``."""
    st.markdown(
        _CSS.substitute(
            INK=C.INK,
            MUTED=C.MUTED,
            FAINT=C.FAINT,
            HAIRLINE=C.HAIRLINE,
            CANVAS=C.CANVAS,
            CARD=C.CARD,
            BRAND=C.BRAND,
            BRAND_DARK=C.BRAND_DARK,
            BRAND_SOFT=C.BRAND_SOFT,
        ),
        unsafe_allow_html=True,
    )


# ============================================================================
# RENDER HELPERS
# ============================================================================
def section(title: str, subtitle: str | None = None, kicker: str = "") -> None:
    """Consistent section header: small brand kicker, strong title, quiet sub."""
    html = ['<div class="cs-section">']
    if kicker:
        html.append(f'<div class="cs-kicker">{kicker}</div>')
    html.append(f'<div class="cs-title">{title}</div>')
    if subtitle:
        html.append(f'<div class="cs-sub">{subtitle}</div>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def pill(level: str) -> str:
    """Return HTML for a severity/status pill. Wrap in ``st.markdown(..., True)``."""
    color = C.SEV.get(level) or C.STATUS.get(level) or C.MUTED
    soft = C.SEV_SOFT.get(level, C.NEUTRAL_SOFT)
    return (
        f'<span class="cs-pill" style="background:{soft};color:{color};">'
        f'<span class="cs-dot" style="background:{color};"></span>{level}</span>'
    )


def metric(
    label: str,
    value,
    delta: str | None = None,
    good: bool | None = None,
) -> None:
    """
    A metric card with EXPLICIT delta colour.

    ``good=True``  -> green (a savings, a reduction, an improvement)
    ``good=False`` -> red   (a cost increase, a breach)
    ``good=None``  -> neutral grey
    """
    dlt_html = ""
    if delta is not None:
        col = C.GOOD if good else C.BAD if good is False else C.MUTED
        dlt_html = f'<div class="dlt" style="color:{col};">{delta}</div>'
    st.markdown(
        f'<div class="cs-card"><div class="lbl">{label}</div>'
        f'<div class="val">{value}</div>{dlt_html}</div>',
        unsafe_allow_html=True,
    )


def money(x: float, decimals: int = 0) -> str:
    """$-format with K/M abbreviation and tabular-friendly output."""
    a = abs(x)
    if a >= 1_000_000:
        return f"${x / 1_000_000:,.2f}M"
    if a >= 1_000:
        return f"${x / 1_000:,.0f}K"
    return f"${x:,.{decimals}f}"


def severity_color(level: str) -> str:
    """Single source of truth for severity colour (charts + pills agree)."""
    return C.SEV.get(level, C.SEV["Low"])


def callout(body: str, *, tone: str = "info", title: str = "") -> None:
    """Themed notice card — drop-in for st.info / warning / success / error."""
    styles: dict[str, tuple[str, str, str]] = {
        "info": (C.INFO, C.BRAND_SOFT, "Notice"),
        "warning": (C.SEV["Medium"], C.SEV_SOFT["Medium"], "Warning"),
        "success": (C.GOOD, C.SEV_SOFT["Low"], "Success"),
        "error": (C.BAD, C.SEV_SOFT["Critical"], "Error"),
    }
    color, soft, label = styles.get(tone, styles["info"])
    with st.container(border=True):
        st.markdown(
            f'<span class="cs-pill" style="background:{soft};color:{color};">'
            f'<span class="cs-dot" style="background:{color};"></span>'
            f"{label}</span>",
            unsafe_allow_html=True,
        )
        if title:
            st.markdown(f"**{title}**")
        st.markdown(body)


def meta_row(items: list[tuple[str, str, str | None]]) -> None:
    """Render a horizontal strip of (label, pill_level, description) entries.

    Used under tile grids for grounding transparency (e.g. "Basis: Measured",
    "Confidence: Medium"). ``pill_level`` is one of the SEV keys (Low /
    Medium / High / Critical) OR any STATUS key — anything ``pill()`` accepts.
    Pass ``description=None`` to render just the pill with no trailing text.

    Wraps to a new line at narrow widths and keeps consistent gap + padding
    so the row reads as one status bar rather than a caption soup.
    """
    parts = ['<div class="cs-meta-row">']
    for label, pill_level, description in items:
        parts.append(
            '<span class="cs-meta-item">'
            f'<span class="cs-meta-label">{label}</span>'
            f"{pill(pill_level)}"
        )
        if description:
            parts.append(f'<span class="cs-meta-text">{description}</span>')
        parts.append("</span>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def confidence_pill(confidence: str) -> str:
    """Confidence badge HTML for ranked recommendations."""
    conf_lower = (confidence or "medium").lower()
    label = conf_lower.title()
    color, soft = {
        "high": (C.GOOD, C.SEV_SOFT["Low"]),
        "medium": (C.SEV["Medium"], C.SEV_SOFT["Medium"]),
        "low": (C.MUTED, C.NEUTRAL_SOFT),
    }.get(conf_lower, (C.MUTED, C.NEUTRAL_SOFT))
    return (
        f'<span class="cs-pill" style="background:{soft};color:{color};">'
        f'<span class="cs-dot" style="background:{color};"></span>{label}</span>'
    )
