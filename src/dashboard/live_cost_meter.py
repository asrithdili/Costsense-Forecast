"""Live Cost Impact Meter — hero instrument for the Dashboard.

Renders via ``streamlit.components.v1.html``. Static display only — no
simulated counters or client-side animation loops.

Data binding (presentation-only):
  * **Primary** — trailing 7-day average daily burn (``last_7_actual / 7``)
  * **Threshold** — next 7-day forecast average daily (``next_7_total / 7``)
  * **Gauge** — burn vs forecast daily (capped visually at 100%+)
  * **Direction** — forecast vs last-7d percent change
"""
from __future__ import annotations

import html
import json
from typing import Any

import streamlit.components.v1 as components

from src.dashboard.costsense_theme import C


_METER_HEIGHT = 188


def _meter_config(
    *,
    daily_burn_usd: float | None,
    forecast_daily_usd: float | None,
    delta_pct: float | None,
    account_label: str,
) -> dict[str, Any]:
    burn = float(daily_burn_usd) if daily_burn_usd is not None else None
    forecast = (
        float(forecast_daily_usd)
        if forecast_daily_usd is not None
        else burn
    )
    pct = float(delta_pct) if delta_pct is not None else None
    ratio = None
    if burn is not None and forecast and forecast > 0:
        ratio = burn / forecast
    return {
        "burn": burn,
        "forecast": forecast,
        "deltaPct": pct,
        "ratio": ratio,
        "account": account_label or "",
        "colors": {
            "ink": C.INK,
            "muted": C.MUTED,
            "faint": C.FAINT,
            "line": C.HAIRLINE,
            "card": C.CARD,
            "brand": C.BRAND,
            "brandDark": C.BRAND_DARK,
            "brandSoft": C.BRAND_SOFT,
            "good": C.GOOD,
            "bad": C.BAD,
        },
    }


def _empty_meter_html(cfg: dict[str, Any]) -> str:
    c = cfg["colors"]
    account = html.escape(cfg.get("account") or "this account")
    return f"""
<div class="cs-meter cs-meter--empty" style="
  --ink:{c['ink']};--muted:{c['muted']};--line:{c['line']};
  --card:{c['card']};--brand:{c['brand']};--brand-soft:{c['brandSoft']};
">
  <div class="cs-meter-kicker">Live impact</div>
  <div class="cs-meter-title">Cost impact meter</div>
  <p class="cs-meter-empty-msg">
    No recent spend data for <strong>{account}</strong> yet.
    Fetch history or run a forecast to populate the meter.
  </p>
</div>
<style>
.cs-meter {{
  font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: linear-gradient(135deg, var(--brand-soft) 0%, var(--card) 55%);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 18px 22px;
  box-shadow: 0 1px 2px rgba(20,24,31,0.04);
  color: var(--ink);
}}
.cs-meter-kicker {{
  color: var(--brand); font-size: 0.72rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;
}}
.cs-meter-title {{
  font-size: 1.05rem; font-weight: 680; letter-spacing: -0.01em; margin-bottom: 8px;
}}
.cs-meter-empty-msg {{ color: var(--muted); font-size: 0.9rem; margin: 0; line-height: 1.5; }}
</style>
"""


def _live_meter_html(cfg: dict[str, Any]) -> str:
    cfg_json = json.dumps(cfg)
    return f"""
<div id="cs-meter-root" class="cs-meter"></div>
<style>
.cs-meter {{
  font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: linear-gradient(135deg, var(--brand-soft) 0%, var(--card) 52%);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 18px 22px 16px;
  box-shadow: 0 1px 2px rgba(20,24,31,0.04);
  color: var(--ink);
  overflow: hidden;
}}
.cs-meter-kicker {{
  color: var(--brand); font-size: 0.72rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 2px;
}}
.cs-meter-head {{ margin-bottom: 14px; }}
.cs-meter-primary-label {{
  color: var(--muted); font-size: 0.72rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.04em;
}}
.cs-meter-primary-value {{
  font-size: 2.35rem; font-weight: 700; letter-spacing: -0.03em;
  font-variant-numeric: tabular-nums; line-height: 1;
  color: var(--ink); margin-top: 4px;
}}
.cs-meter-primary-unit {{
  font-size: 1rem; font-weight: 600; color: var(--muted); margin-left: 4px;
}}
.cs-meter-gauge-wrap {{ margin-bottom: 12px; }}
.cs-meter-gauge-meta {{
  display: flex; justify-content: space-between; gap: 8px;
  font-size: 0.78rem; color: var(--muted); margin-bottom: 6px;
}}
.cs-meter-gauge-track {{
  height: 10px; background: var(--line); border-radius: 999px; overflow: hidden;
}}
.cs-meter-gauge-fill {{
  height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, var(--brand) 0%, var(--brand-dark) 100%);
}}
.cs-meter-gauge-fill.is-over {{
  background: var(--bad);
}}
.cs-meter-foot {{
  display: flex; justify-content: space-between; align-items: center;
  gap: 12px; flex-wrap: wrap; font-size: 0.82rem; color: var(--muted);
}}
.cs-meter-direction {{
  font-weight: 600; font-variant-numeric: tabular-nums;
}}
.cs-meter-direction.up {{ color: var(--bad); }}
.cs-meter-direction.down {{ color: var(--good); }}
.cs-meter-direction.flat {{ color: var(--muted); }}
.cs-meter-threshold {{ font-variant-numeric: tabular-nums; }}
</style>
<script>
(function() {{
  const cfg = {cfg_json};
  const c = cfg.colors;
  const root = document.getElementById('cs-meter-root');
  if (!root || cfg.burn == null || cfg.burn <= 0) return;

  const fmtMoney0 = (n) => {{
    const a = Math.abs(n);
    if (a >= 1e6) return '$' + (n / 1e6).toFixed(2) + 'M';
    if (a >= 1e3) return '$' + Math.round(n / 1e3) + 'K';
    return '$' + Math.round(n);
  }};

  const burn = cfg.burn;
  const forecast = cfg.forecast || burn;
  const ratio = cfg.ratio != null ? cfg.ratio : (forecast > 0 ? burn / forecast : 0);
  const pctFill = Math.min(100, Math.max(0, ratio * 100));
  const isOver = ratio > 1.001;
  const delta = cfg.deltaPct;
  let dirClass = 'flat';
  let dirText = 'Forecast flat vs last 7d';
  if (delta != null) {{
    if (delta > 5) {{ dirClass = 'up'; dirText = 'Forecast up ' + delta.toFixed(1) + '% vs last 7d'; }}
    else if (delta < -5) {{ dirClass = 'down'; dirText = 'Forecast down ' + Math.abs(delta).toFixed(1) + '% vs last 7d'; }}
    else {{ dirText = 'Forecast near last 7d (' + (delta >= 0 ? '+' : '') + delta.toFixed(1) + '%)'; }}
  }}

  root.style.setProperty('--ink', c.ink);
  root.style.setProperty('--muted', c.muted);
  root.style.setProperty('--faint', c.faint);
  root.style.setProperty('--line', c.line);
  root.style.setProperty('--card', c.card);
  root.style.setProperty('--brand', c.brand);
  root.style.setProperty('--brand-dark', c.brandDark);
  root.style.setProperty('--brand-soft', c.brandSoft);
  root.style.setProperty('--good', c.good);
  root.style.setProperty('--bad', c.bad);

  root.innerHTML = `
    <div class="cs-meter-kicker">Live impact</div>
    <div class="cs-meter-head">
      <div class="cs-meter-primary-label">Trailing 7-day burn rate</div>
      <div>
        <span class="cs-meter-primary-value">${{fmtMoney0(burn)}}</span>
        <span class="cs-meter-primary-unit">/ day</span>
      </div>
    </div>
    <div class="cs-meter-gauge-wrap">
      <div class="cs-meter-gauge-meta">
        <span>Burn vs forecast daily</span>
        <span>${{Math.round(ratio * 100)}}% of forecast</span>
      </div>
      <div class="cs-meter-gauge-track">
        <div class="cs-meter-gauge-fill${{isOver ? ' is-over' : ''}}" style="width:${{pctFill.toFixed(2)}}%"></div>
      </div>
    </div>
    <div class="cs-meter-foot">
      <span class="cs-meter-direction ${{dirClass}}">${{dirText}}</span>
      <span class="cs-meter-threshold">Forecast daily: ${{fmtMoney0(forecast)}}</span>
    </div>
  `;
}})();
</script>
"""


def render_live_cost_meter(
    *,
    daily_burn_usd: float | None,
    forecast_daily_usd: float | None = None,
    delta_pct: float | None = None,
    account_label: str = "",
) -> None:
    """Render the hero live cost impact meter (static, data-driven display)."""
    cfg = _meter_config(
        daily_burn_usd=daily_burn_usd,
        forecast_daily_usd=forecast_daily_usd,
        delta_pct=delta_pct,
        account_label=account_label,
    )
    if cfg["burn"] is None or cfg["burn"] <= 0:
        html = _empty_meter_html(cfg)
    else:
        html = _live_meter_html(cfg)
    components.html(html, height=_METER_HEIGHT, scrolling=False)
