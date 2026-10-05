"""
FII x Nifty 50 - live market dashboard.
A project by Akhil Madhu.

Layout: a card-based "bento" dashboard - hero Nifty chart with range selector,
a synthesized market read, KPI cards (gauge / split bars), signals, a live
markets watchlist, NSE session clock, and research charts. Panels can be
switched on/off or swapped by template from the sidebar.
"""
from __future__ import annotations

import html
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import streamlit as st  # noqa: E402  (needed early for the secrets shim below)

# Streamlit Cloud has no .env file - secrets go in its own "Secrets" panel instead, exposed as
# st.secrets. Everything in this app reads credentials via os.getenv(...), so this one shim makes
# both local .env and Streamlit Cloud secrets work through the exact same code path everywhere
# else, with no changes needed to the broker adapters, alerts, etc.
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(str(_k), str(_v))
except Exception:
    pass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components
from scipy import stats as sps
from streamlit_autorefresh import st_autorefresh

from src.alerts.notifier import AlertManager
from src.analytics.fii_nifty import (_prepare_merged, dii_absorption, fii_streak,
                                     fo_positioning, full_fii_nifty_analysis, regime_from_flows)
from src.database.store import DataStore
from src.ingestion.real_sources import RealDataProvider, get_nifty_live

# ---------------------------------------------------------------------------
# Theme - written once so native widgets (pills, selects) match the palette
# ---------------------------------------------------------------------------
THEME_TOML = """[theme]
base = "dark"
primaryColor = "#B8F23B"
backgroundColor = "#06100B"
secondaryBackgroundColor = "#0D1B14"
textColor = "#E8F0E9"

[browser]
gatherUsageStats = false
"""


def _ensure_theme_file() -> None:
    cfg = ROOT / ".streamlit" / "config.toml"
    if cfg.exists():
        return
    try:
        cfg.parent.mkdir(exist_ok=True)
        cfg.write_text(THEME_TOML, encoding="utf-8")
    except OSError:
        pass


_ensure_theme_file()

st.set_page_config(page_title="FII x Nifty 50", page_icon="\u25c8", layout="wide",
                  initial_sidebar_state="expanded")
_refresh_sec = st.session_state.get("refresh_sec", 5)
st_autorefresh(interval=_refresh_sec * 1000, key="refresh")

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
THEMES = {
    "dark": dict(bg="#06100B", text="#E8F0E9", muted="#7F9488", line="rgba(255,255,255,0.07)",
                 lime="#B8F23B", red="#FF5C6C", amber="#F2C14E",
                 card1="rgba(22,42,31,.80)", card2="rgba(10,22,16,.94)",
                 glow1="rgba(184,242,59,0.09)", glow2="rgba(30,120,70,0.18)",
                 hover="#13261C", scroll="rgba(255,255,255,.06)"),
    "light": dict(bg="#F3F7F1", text="#152018", muted="#5C6F63", line="rgba(0,0,0,0.09)",
                  lime="#4F8A22", red="#C43E4C", amber="#A9781F",
                  card1="rgba(255,255,255,.92)", card2="rgba(235,244,236,.96)",
                  glow1="rgba(79,138,34,0.07)", glow2="rgba(120,180,120,0.14)",
                  hover="#FFFFFF", scroll="rgba(0,0,0,.08)"),
}
# The sidebar toggle uses key="theme"; Streamlit keeps that key's value in
# session_state across reruns even before the widget itself runs again
# later in the script, so it is safe to read here, at the top.
_theme = THEMES[st.session_state.get("theme", "dark")]
BG, TEXT, MUTED, LINE = _theme["bg"], _theme["text"], _theme["muted"], _theme["line"]
LIME, RED, AMBER = _theme["lime"], _theme["red"], _theme["amber"]
CARD1, CARD2, GLOW1, GLOW2, HOVER = (_theme["card1"], _theme["card2"], _theme["glow1"],
                                     _theme["glow2"], _theme["hover"])
IST = timezone(timedelta(hours=5, minutes=30))

# The four phases of the SIP research. Edit these dates if any boundary differs in your report.
SIP_PHASES = [
    ("Baseline", "2022-01-01", "2023-12-31", "#6F8FA3"),
    ("Accumulation", "2024-01-01", "2024-09-30", "#B8F23B"),
    ("FII exodus", "2024-10-01", "2025-03-31", "#FF5C6C"),
    ("Stabilisation", "2025-04-01", "2026-03-31", "#F2C14E"),
]

CSS_VARS = (f":root{{--bg:{BG};--text:{TEXT};--muted:{MUTED};--lime:{LIME};--red:{RED};--amber:{AMBER};"
            f"--line:{LINE};--card1:{CARD1};--card2:{CARD2};--glow1:{GLOW1};--glow2:{GLOW2};--hover:{HOVER};}}")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

.stApp, .stApp p, .stApp label, .stApp li, .stApp button, .stApp input,
.stApp span:not([data-testid="stIconMaterial"]) { font-family: 'Manrope', sans-serif; }

.stApp {
  background:
    radial-gradient(900px 520px at 12% -8%, var(--glow1), transparent 60%),
    radial-gradient(800px 520px at 100% 0%, var(--glow2), transparent 55%),
    var(--bg) !important;
  color: var(--text);
}
#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 1.4rem 2rem 3rem; max-width: 1340px; }
section[data-testid="stSidebar"] { background: #040A07; border-right: 1px solid var(--line); }
div[data-stale="true"] { opacity: 1 !important; transition: none !important; }

/* ---------- cards ---------- */
.card, div[class*="st-key-card-"] {
  background: linear-gradient(180deg, var(--card1) 0%, var(--card2) 100%) !important;
  border: 1px solid var(--line) !important;
  border-radius: 20px !important;
  padding: 18px 20px !important;
  box-shadow: 0 12px 32px rgba(0,0,0,.30), inset 0 1px 0 rgba(255,255,255,.04);
}
.card-title { font-size: 15px; font-weight: 700; letter-spacing: .01em; color: var(--text); }
.card-sub { font-size: 12px; color: var(--muted); margin-top: 2px; }
.card-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
.mut { color: var(--muted); }
.num { font-variant-numeric: tabular-nums; }

/* ---------- top bar ---------- */
.topbar { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; margin: 0 0 18px 0; }
.brand { display: flex; align-items: center; gap: 12px; margin-right: 8px; }
.logo { width: 40px; height: 40px; border-radius: 12px; display: grid; place-items: center;
  background: linear-gradient(135deg, #B8F23B, #4CAF7D); color: #06100B; font-weight: 800; font-size: 18px; }
.brand-name { font-weight: 800; letter-spacing: .06em; font-size: 16px; }
.brand-sub { font-size: 11.5px; color: var(--muted); margin-top: 1px; }
.chip { background: rgba(255,255,255,.035); border: 1px solid var(--line); border-radius: 14px; padding: 8px 14px; min-width: 150px; }
.chip-label { font-size: 11px; color: var(--muted); }
.chip-row { display: flex; align-items: center; gap: 8px; margin-top: 2px; }
.chip-val { font-size: 16px; font-weight: 700; font-variant-numeric: tabular-nums; }
.status { margin-left: auto; display: flex; align-items: center; gap: 8px; font-size: 12.5px; color: var(--muted); }
.pulse { width: 8px; height: 8px; border-radius: 50%; background: var(--lime); animation: pulse 2.2s infinite; }
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(184,242,59,.55); }
  70% { box-shadow: 0 0 0 8px rgba(184,242,59,0); }
  100% { box-shadow: 0 0 0 0 rgba(184,242,59,0); }
}
.pill { display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 700; font-variant-numeric: tabular-nums; }
.pill.up { background: rgba(184,242,59,.14); color: var(--lime); }
.pill.down { background: rgba(255,92,108,.14); color: var(--red); }
.pill.flat { background: rgba(242,193,78,.14); color: var(--amber); }

/* ---------- hero ---------- */
.hero-val { font-size: 40px; font-weight: 800; letter-spacing: -.01em; font-variant-numeric: tabular-nums; line-height: 1.05; }
.hero-row { display: flex; align-items: center; gap: 12px; margin-top: 4px; }
.hero-sub { font-size: 12px; color: var(--muted); margin-top: 6px; }
div[class*="st-key-range"] { display: flex; justify-content: flex-end; }

/* ---------- market read ---------- */
.read-headline { font-size: 15.5px; line-height: 1.55; font-weight: 600; margin: 4px 0 14px 0; }
.bullet { display: flex; gap: 10px; font-size: 13px; line-height: 1.55; color: #B7C7BC; margin-bottom: 9px; }
.bullet .dot { flex: 0 0 6px; height: 6px; border-radius: 50%; margin-top: 8px; background: var(--muted); }
.conf { margin-top: 16px; }
.conf-label { display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); margin-bottom: 6px; }
.conf-bar { height: 6px; border-radius: 6px; background: rgba(255,255,255,.07); overflow: hidden; }
.conf-bar span { display: block; height: 100%; border-radius: 6px; }
.read-foot { font-size: 11.5px; color: var(--muted); margin-top: 14px; line-height: 1.5; }

/* ---------- kpi ---------- */
.kpi { min-height: 168px; }
.kpi-label { font-size: 12.5px; color: var(--muted); font-weight: 600; margin-bottom: 8px; }
.kpi-val { font-size: 30px; font-weight: 800; font-variant-numeric: tabular-nums; line-height: 1.1; }
.kpi-sub { font-size: 12px; color: var(--muted); margin-top: 8px; line-height: 1.45; }
.gauge-wrap { position: relative; width: 150px; margin: 2px auto 0 auto; }
.gauge { width: 150px; height: auto; display: block; }
.gauge-fill { animation: gaugeIn 1.1s ease-out; }
@keyframes gaugeIn { from { stroke-dasharray: 0 188.5; } }
.gauge-val { position: absolute; left: 0; right: 0; bottom: 0; text-align: center; font-size: 24px; font-weight: 800; font-variant-numeric: tabular-nums; }
.splitbar { display: flex; height: 14px; border-radius: 999px; overflow: hidden; background: rgba(255,255,255,.06); margin-top: 14px; gap: 2px; }
.splitbar span { display: block; height: 100%; }
.legend { display: flex; gap: 14px; font-size: 11.5px; color: var(--muted); margin-top: 8px; }
.legend i { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }
.dir-badge { width: 34px; height: 34px; border-radius: 50%; display: grid; place-items: center; font-size: 15px; font-weight: 800; }

/* ---------- rows (signals / markets) ---------- */
.row { display: flex; align-items: center; gap: 12px; padding: 11px 0; border-top: 1px solid var(--line); }
.row:first-of-type { border-top: none; }
.row-main { flex: 1; min-width: 0; }
.row-title { font-size: 13.5px; font-weight: 600; }
.row-sub { font-size: 11.5px; color: var(--muted); margin-top: 1px; }
.row-val { text-align: right; font-size: 13.5px; font-weight: 700; font-variant-numeric: tabular-nums; }
.row-chg { font-size: 11.5px; font-weight: 700; margin-top: 1px; }
.row-chg.up { color: var(--lime); } .row-chg.down { color: var(--red); } .row-chg.flat { color: var(--muted); }
.sdot { width: 8px; height: 8px; border-radius: 50%; flex: 0 0 8px; }
.badge { width: 34px; height: 34px; border-radius: 50%; flex: 0 0 34px; display: grid; place-items: center;
  background: rgba(184,242,59,.10); color: var(--lime); font-size: 10.5px; font-weight: 800; letter-spacing: .02em; }

/* ---------- market hours ---------- */
.hours-status { display: flex; align-items: center; gap: 10px; margin: 6px 0 4px 0; }
.hours-big { font-size: 26px; font-weight: 800; }
.track { position: relative; height: 8px; border-radius: 8px; background: rgba(255,255,255,.07); margin: 16px 0 8px 0; }
.track span { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 8px; background: linear-gradient(90deg, rgba(184,242,59,.35), #B8F23B); }
.track-labels { display: flex; justify-content: space-between; font-size: 11.5px; color: var(--muted); }

/* ---------- stats tables ---------- */
.tgroup { font-size: 12px; font-weight: 700; color: var(--muted); margin: 18px 0 4px 0; }
.trow { display: grid; grid-template-columns: 1fr 92px 70px 120px; gap: 10px; align-items: center; padding: 9px 0; border-top: 1px solid var(--line); }
.trow .tname { font-size: 13.5px; font-weight: 600; }
.trow .tsub { font-size: 11.5px; color: var(--muted); font-weight: 400; margin-top: 1px; }
.trow .tstat { text-align: right; font-weight: 700; font-variant-numeric: tabular-nums; font-size: 13.5px; }
.trow .tp { text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; font-size: 12.5px; }
.trow .tv { text-align: right; }
.chips { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0 6px 0; }
.mini { background: rgba(255,255,255,.04); border: 1px solid var(--line); border-radius: 10px; padding: 4px 10px; font-size: 12px; color: var(--muted); }
.mini b { color: var(--text); font-variant-numeric: tabular-nums; margin-left: 6px; }

/* ---------- phases table ---------- */
.prow { display: grid; grid-template-columns: 1.3fr 78px 78px 88px 100px; gap: 10px; align-items: center; padding: 11px 0; border-top: 1px solid var(--line); }
.prow.head { border-top: none; font-size: 11.5px; color: var(--muted); padding: 2px 0 6px 0; }
.prow > div:not(:first-child) { text-align: right; font-variant-numeric: tabular-nums; font-size: 13.5px; font-weight: 700; }
.prow.head > div:not(:first-child) { font-weight: 500; font-size: 11.5px; }
.pname { display: flex; align-items: center; gap: 10px; }
.pdot { width: 10px; height: 10px; border-radius: 3px; flex: 0 0 10px; }
div[class*="st-key-show_phases"] { display: flex; justify-content: flex-end; }

/* ---------- ticker ---------- */
.ticker-wrap { overflow: hidden; white-space: nowrap; border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line); padding: 9px 0; margin: 2px 0 20px 0; -webkit-mask-image:
  linear-gradient(90deg, transparent, #000 4%, #000 96%, transparent); mask-image:
  linear-gradient(90deg, transparent, #000 4%, #000 96%, transparent); }
.ticker-track { display: inline-block; animation: ticker 30s linear infinite; }
.ticker-track:hover { animation-play-state: paused; }
@keyframes ticker { from { transform: translateX(0); } to { transform: translateX(-50%); } }
.ticker-item { display: inline-flex; align-items: center; gap: 7px; margin-right: 34px; font-size: 13px; }
.ticker-item b { font-weight: 700; font-variant-numeric: tabular-nums; }

/* ---------- what-if ---------- */
.whatif-out { text-align: center; padding: 10px 0 4px 0; }
.whatif-big { font-size: 30px; font-weight: 800; font-variant-numeric: tabular-nums; }
.whatif-sub { font-size: 12px; color: var(--muted); margin-top: 4px; }

/* ---------- big-number pair (daily vs monthly) ---------- */
.pair { display: flex; gap: 18px; }
.pair > div { flex: 1; text-align: center; padding: 14px 10px; border-radius: 14px; background: rgba(255,255,255,.03); }
.pair .pv { font-size: 28px; font-weight: 800; font-variant-numeric: tabular-nums; }
.pair .pl { font-size: 12px; color: var(--muted); margin-top: 4px; }

/* ---------- plain-language explanations ---------- */
.explain { margin-top: 12px; padding: 10px 13px; background: rgba(255,255,255,.035);
  border-radius: 10px; font-size: 12.5px; line-height: 1.65; color: #C3D1C8; }
.explain-tag { display: block; font-size: 10.5px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .05em; color: var(--muted); margin-bottom: 4px; }
.explain b { color: var(--text); font-weight: 700; }

/* ---------- native widgets ---------- */
[data-testid="stExpander"] { border: 1px solid var(--line) !important; border-radius: 16px !important; background: rgba(10,22,16,.6); }
.stButton > button { border-radius: 12px; border: 1px solid var(--line); }
"""

st.markdown("<style>" + CSS_VARS + CSS + "</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def H(s: str) -> str:
    """Collapse whitespace so Markdown never mistakes indented HTML for code."""
    s = " ".join(s.split())
    return s.replace("> <", "><")


def E(x) -> str:
    return html.escape(str(x))


def fnum(x, dp: int = 0, signed: bool = False, default: str = "\u2014") -> str:
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
        return f"{x:+,.{dp}f}" if signed else f"{x:,.{dp}f}"
    except (TypeError, ValueError):
        return default


def tone_cls(v) -> str:
    if v is None or v == 0:
        return "flat"
    return "up" if v > 0 else "down"


def plot(fig: go.Figure, key: str) -> None:
    cfg = {"displayModeBar": False}
    try:
        st.plotly_chart(fig, width="stretch", config=cfg, key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=cfg, key=key)


def card(name: str):
    try:
        return st.container(key=f"card-{name}")
    except TypeError:
        return st.container(border=True)


def _hm(td: timedelta) -> str:
    mins = max(int(td.total_seconds() // 60), 0)
    if mins >= 24 * 60:
        return f"{mins // (24 * 60)}d {(mins % (24 * 60)) // 60}h"
    return f"{mins // 60}h {mins % 60:02d}m"


def approx_span(n_sessions: int) -> str:
    """Turn a count of trading days into an everyday-length phrase, e.g. 47 -> 'about 2 months'."""
    if n_sessions is None or n_sessions <= 0:
        return "no time at all"
    if n_sessions < 5:
        return f"the last {n_sessions} trading day{'s' if n_sessions != 1 else ''}"
    weeks = n_sessions / 5
    if weeks < 8:
        wr = round(weeks)
        return f"about {wr} week{'s' if wr != 1 else ''}"
    months = n_sessions / 21
    if months < 20:
        mr = round(months)
        return f"about {mr} month{'s' if mr != 1 else ''}"
    years = n_sessions / 252
    return f"about {years:.1f} year{'s' if years >= 1.05 else ''}"


RANGE_PHRASE = {"1M": "the past month", "3M": "the past 3 months", "6M": "the past 6 months",
               "1Y": "the past year", "5Y": "the past 5 years"}


def explain(text: str) -> None:
    """One consistent, quiet box for a plain-English read of whatever chart sits above it.
    Every number in `text` should already be computed from real data - this never guesses."""
    st.markdown(H(f'<div class="explain"><span class="explain-tag">in simple words</span>{text}</div>'),
               unsafe_allow_html=True)


def mini_sparkline(values, color, w=124, h=28) -> str:
    """A tiny inline SVG trend line, no JS, no chart library - safe to embed anywhere."""
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or (abs(hi) or 1.0)
    n = len(vals)
    pts = " ".join(f"{(i / (n - 1)) * w:.1f},{h - ((v - lo) / span) * h:.1f}" for i, v in enumerate(vals))
    return (f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="display:block;margin:10px auto 0 auto">'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round" opacity="0.85"/></svg>')


def market_status(now: datetime):
    """NSE cash session (Mon-Fri). Exchange holidays are not modelled."""
    d = now.date()
    pre = datetime.combine(d, dtime(9, 0), tzinfo=IST)
    opn = datetime.combine(d, dtime(9, 15), tzinfo=IST)
    cls = datetime.combine(d, dtime(15, 30), tzinfo=IST)
    weekday = now.weekday() < 5
    if weekday and pre <= now < opn:
        return "Pre-open", AMBER, f"opens in {_hm(opn - now)}", 0.0
    if weekday and opn <= now < cls:
        return "Open", LIME, f"closes in {_hm(cls - now)}", (now - opn) / (cls - opn)
    nxt = opn
    while nxt <= now or nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    done = 1.0 if (weekday and now >= cls) else 0.0
    return "Closed", MUTED, f"opens in {_hm(nxt - now)}", done


# ==== STATS BEGIN ====
# Small, dependency-light statistical tests (numpy + scipy only). Verified against statsmodels.
ADF_CRIT = {"1%": -3.43, "5%": -2.86, "10%": -2.57}  # asymptotic, constant only


def pearson_test(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return None
    r, p = sps.pearsonr(x, y)
    return dict(stat=float(r), p=float(p), n=len(x))


def spearman_test(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return None
    r, p = sps.spearmanr(x, y)
    return dict(stat=float(r), p=float(p), n=len(x))


def ols_test(x, y):
    """y = a + b*x. Returns slope, intercept, R2, t, two-sided p."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 5 or np.std(x) == 0:
        return None
    res = sps.linregress(x, y)
    t = res.slope / res.stderr if res.stderr else float("nan")
    return dict(slope=float(res.slope), intercept=float(res.intercept), r2=float(res.rvalue ** 2),
                t=float(t), p=float(res.pvalue), n=len(x))


def granger_test(y, x, lag):
    """F-test: do lags of x add predictive power for y beyond y's own lags?"""
    y, x = np.asarray(y, float), np.asarray(x, float)
    T = len(y)
    n = T - lag
    df2 = n - 2 * lag - 1
    if df2 < 10:
        return None
    Y = y[lag:]
    ylags = np.column_stack([y[lag - k: T - k] for k in range(1, lag + 1)])
    xlags = np.column_stack([x[lag - k: T - k] for k in range(1, lag + 1)])
    ones = np.ones((n, 1))

    def rss(X):
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        e = Y - X @ beta
        return float(e @ e)

    rss_r, rss_u = rss(np.hstack([ones, ylags])), rss(np.hstack([ones, ylags, xlags]))
    if rss_u <= 0:
        return None
    F = ((rss_r - rss_u) / lag) / (rss_u / df2)
    return dict(stat=float(F), p=float(sps.f.sf(F, lag, df2)), n=n, lag=lag)


def adf_test(y, lags=1):
    """Augmented Dickey-Fuller (constant, fixed lag). Compare tau with ADF_CRIT."""
    y = np.asarray(pd.Series(y).dropna(), float)
    N = len(y)
    if N - 1 - lags < 20:
        return None
    dy = np.diff(y)
    Y = dy[lags:]
    cols = [np.ones(len(Y)), y[lags: N - 1]] + [dy[lags - j: N - 1 - j] for j in range(1, lags + 1)]
    X = np.column_stack(cols)
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    e = Y - X @ beta
    dof = len(Y) - X.shape[1]
    cov = (e @ e / dof) * np.linalg.inv(X.T @ X)
    tau = float(beta[1] / np.sqrt(cov[1, 1]))
    level = next((k for k in ("1%", "5%", "10%") if tau < ADF_CRIT[k]), None)
    return dict(stat=tau, n=len(Y), level=level)


def strength(r):
    a = abs(r)
    return "very weak" if a < 0.2 else "weak" if a < 0.4 else "moderate" if a < 0.6 else "strong" if a < 0.8 else "very strong"


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"
# ==== STATS END ====


# ==== ANALYTICS BEGIN ====
def hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def phase_stats(hist, start, end):
    """Nifty behaviour inside one phase: price return, annualised volatility, max drawdown."""
    if hist is None or len(hist) == 0:
        return None
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    seg = hist[(hist["date"] >= s) & (hist["date"] <= e)]
    if len(seg) < 5:
        return None
    c = seg["close"].astype(float).reset_index(drop=True)
    daily = c.pct_change().dropna()
    return dict(
        ret=float((c.iloc[-1] / c.iloc[0] - 1) * 100),
        vol=float(daily.std() * np.sqrt(252) * 100),
        dd=float((c / c.cummax() - 1).min() * 100),
        n=len(seg),
        partial=bool(seg["date"].min() > s + pd.Timedelta(days=10) or seg["date"].max() < e - pd.Timedelta(days=10)),
    )


def phase_corr(monthly, start, end):
    """FII vs Nifty correlation inside a phase, from the user's own monthly SIP file."""
    if monthly is None:
        return None
    seg = monthly[(monthly["date"] >= pd.Timestamp(start)) & (monthly["date"] <= pd.Timestamp(end))]
    seg = seg.dropna(subset=["fii_net_cr", "nifty_return_pct"])
    if len(seg) < 4 or seg["fii_net_cr"].std() == 0 or seg["nifty_return_pct"].std() == 0:
        return None
    r, p = sps.pearsonr(seg["fii_net_cr"], seg["nifty_return_pct"])
    return dict(r=float(r), p=float(p), n=len(seg))


def calendar_grid(df, col):
    """Weekday x week grid (Mon-Fri rows) of one flow column, for a GitHub-style calendar."""
    d = df[["date", col]].dropna().copy()
    d = d[d["date"].dt.weekday < 5]
    if d.empty:
        return None
    first = d["date"].min().normalize()
    start = first - pd.Timedelta(days=int(first.weekday()))
    d["week"] = ((d["date"].dt.normalize() - start).dt.days // 7).astype(int)
    d["dow"] = d["date"].dt.weekday.astype(int)
    nweeks = int(d["week"].max()) + 1
    z = np.full((5, nweeks), np.nan)
    txt = np.full((5, nweeks), "", dtype=object)
    for row in d.itertuples(index=False):
        z[row.dow, row.week] = getattr(row, col)
        txt[row.dow, row.week] = row.date.strftime("%d %b %Y")
    tickvals, ticktext, prev = [], [], None
    for w in range(nweeks):
        m = (start + pd.Timedelta(days=7 * w)).month
        if m != prev:
            tickvals.append(w)
            ticktext.append((start + pd.Timedelta(days=7 * w)).strftime("%b"))
            prev = m
    return dict(z=z, txt=txt, nweeks=nweeks, tickvals=tickvals, ticktext=ticktext)


def corr_matrix(df, cols, min_n=20):
    """Pairwise Pearson r and p-values (NaN where fewer than min_n paired observations)."""
    k = len(cols)
    r, p = np.full((k, k), np.nan), np.full((k, k), np.nan)
    for i in range(k):
        for j in range(k):
            if i == j:
                if df[cols[i]].notna().sum() >= min_n:
                    r[i, j], p[i, j] = 1.0, 0.0
                continue
            pair = df[[cols[i], cols[j]]].dropna()
            if len(pair) >= min_n and pair.iloc[:, 0].std() > 0 and pair.iloc[:, 1].std() > 0:
                rr, pp = sps.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
                r[i, j], p[i, j] = rr, pp
    return r, p
# ==== ANALYTICS END ====


REGIME_LABEL = {
    "FII_SELL_PRESSURE": "FII sell pressure",
    "DII_SUPPORTED": "DII supported",
    "FII_LED_RALLY": "FII-led rally",
    "MILD_FII_OUTFLOW": "Mild FII outflow",
    "MILD_FII_INFLOW": "Mild FII inflow",
    "NEUTRAL": "Neutral",
}
TONE_COLOR = {"bearish": RED, "bullish": LIME, "neutral": AMBER}
TONE_CLS = {"bearish": "down", "bullish": "up", "neutral": "flat"}


# ---------------------------------------------------------------------------
# Data (all network calls cached so widget clicks stay instant)
# ---------------------------------------------------------------------------
def live_tick_cached():
    """No caching here on purpose - a 'live' price is supposed to be fresh on every single
    refresh, so caching it would just recreate the stuck-clock problem this is meant to fix."""
    return get_nifty_live()


@st.cache_data(ttl=900, show_spinner=False)
def nifty_long_history() -> pd.DataFrame:
    try:
        import yfinance as yf
        h = yf.Ticker("^NSEI").history(period="5y", interval="1d")
        if h.empty:
            return pd.DataFrame()
        h = h.reset_index()
        h["date"] = pd.to_datetime(h["Date"]).dt.tz_localize(None)
        return h[["date", "Close"]].rename(columns={"Close": "close"}).dropna()
    except Exception:
        return pd.DataFrame()


# name, yahoo symbol, badge, decimals, prefix, suffix, change mode
WATCH = [
    ("Bank Nifty", "^NSEBANK", "BN", 2, "", "", "pct"),
    ("India VIX", "^INDIAVIX", "VIX", 2, "", "", "pct"),
    ("USD / INR", "INR=X", "\u20b9", 3, "", "", "pct"),
    ("US 10Y yield", "^TNX", "10Y", 3, "", "%", "bp"),
    ("Brent crude", "BZ=F", "OIL", 2, "$", "", "pct"),
    ("Gold", "GC=F", "AU", 1, "$", "", "pct"),
]


def _fetch_one(item):
    name, sym, badge, dp, pre, suf, mode = item
    last = prev = None
    try:
        import yfinance as yf
        closes = yf.Ticker(sym).history(period="5d", interval="1d")["Close"].dropna()
        if len(closes) >= 1:
            last = float(closes.iloc[-1])
        if len(closes) >= 2:
            prev = float(closes.iloc[-2])
    except Exception:
        pass
    chg = None
    if last is not None and prev:
        chg = (last - prev) * 100 if mode == "bp" else (last / prev - 1) * 100
    return dict(name=name, badge=badge, dp=dp, pre=pre, suf=suf, mode=mode, last=last, chg=chg)


@st.cache_data(ttl=90, show_spinner=False)
def fetch_watchlist():
    with ThreadPoolExecutor(max_workers=len(WATCH)) as ex:
        return list(ex.map(_fetch_one, WATCH))


# name, yahoo symbol, change type. Yields are compared in basis points, everything else in percent.
MKT_VARS = [
    ("Nifty", "^NSEI", "pct"), ("Bank Nifty", "^NSEBANK", "pct"), ("India VIX", "^INDIAVIX", "pct"),
    ("USD/INR", "INR=X", "pct"), ("US 10Y", "^TNX", "bp"), ("Brent", "BZ=F", "pct"), ("Gold", "GC=F", "pct"),
]


def _daily_change(item):
    name, sym, mode = item
    try:
        import yfinance as yf
        c = yf.Ticker(sym).history(period="1y", interval="1d")["Close"].dropna()
        c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
        c = c[~c.index.duplicated(keep="last")]
        ch = c.diff() * 100 if mode == "bp" else c.pct_change() * 100
        return ch.rename(name)
    except Exception:
        return pd.Series(dtype=float, name=name)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_changes() -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=len(MKT_VARS)) as ex:
        parts = list(ex.map(_daily_change, MKT_VARS))
    parts = [p for p in parts if len(p) > 0]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, axis=1)
    df.index.name = "date"
    return df.reset_index()


# Best-known Yahoo Finance tickers for NSE sectoral indices. Unlike ^NSEI/^NSEBANK (already
# confirmed working elsewhere in this file), these could not be verified against a live feed
# from this build environment - Yahoo occasionally drops or renames a sectoral ticker. Any
# sector that comes back empty is simply skipped below rather than shown as an error, and is
# easy to swap here if one stops working.
SECTOR_TICKERS = [
    ("Bank", "^NSEBANK"), ("IT", "^CNXIT"), ("Auto", "^CNXAUTO"), ("Pharma", "^CNXPHARMA"),
    ("FMCG", "^CNXFMCG"), ("Metal", "^CNXMETAL"), ("Energy", "^CNXENERGY"),
    ("Realty", "^CNXREALTY"), ("Media", "^CNXMEDIA"), ("PSU Bank", "^CNXPSUBANK"),
]


def _sector_return(item):
    name, sym = item
    try:
        import yfinance as yf
        h = yf.Ticker(sym).history(period="6mo", interval="1d")["Close"].dropna()
        if len(h) < 6:
            return None
        h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
        return dict(name=name, series=h)
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_sector_series():
    with ThreadPoolExecutor(max_workers=len(SECTOR_TICKERS)) as ex:
        results = list(ex.map(_sector_return, SECTOR_TICKERS))
    return [r for r in results if r is not None]


@st.cache_data(ttl=900, show_spinner=False)
def load_fpi_sector_csv():
    """Optional: NSDL's official fortnightly sector-wise FPI investment data, saved by hand as
    data/fpi_sector.csv (columns: sector, net_invest_cr, and optionally period).
    NSDL publishes this free at fpi.nsdl.co.in; there is no free live/API version of it, so this
    is a manual-refresh file, the same pattern as data/sip_monthly.csv."""
    path = ROOT / "data" / "fpi_sector.csv"
    if not path.exists():
        return None
    try:
        d = pd.read_csv(path)
        if not {"sector", "net_invest_cr"}.issubset(d.columns):
            return None
        return d
    except Exception:
        return None


@st.cache_data(ttl=300, show_spinner=False)
def load_sip_monthly():
    """Optional: your own monthly SIP data at data/sip_monthly.csv (date, fii_net_cr, nifty_return_pct)."""
    path = ROOT / "data" / "sip_monthly.csv"
    if not path.exists():
        return None
    try:
        m = pd.read_csv(path)
        if not {"date", "fii_net_cr", "nifty_return_pct"}.issubset(m.columns):
            return None
        m["date"] = pd.to_datetime(m["date"], errors="coerce")
        return m.dropna(subset=["date"])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "provider" not in st.session_state:
    st.session_state.provider = RealDataProvider()
    st.session_state.store = DataStore()
    st.session_state.provider.refresh_history_if_needed(force=True)
    st.session_state["flows_refresh_ts"] = time.time()
    st.session_state.alerter = AlertManager(
        trigger_regimes={"FII_SELL_PRESSURE", "DII_SUPPORTED"},
        cooldown_minutes=240,
        channels=["email"],
    )

provider: RealDataProvider = st.session_state.provider
store: DataStore = st.session_state.store
alerter: AlertManager = st.session_state.alerter

live = live_tick_cached()
if live and st.session_state.get("last_tick_ts") != live.ts:
    store.realtime.push([live])  # session tick buffer grows with each new tick
    st.session_state["last_tick_ts"] = live.ts

now_ist = datetime.now(IST)

# ---------------------------------------------------------------------------
# Sidebar: templates + panel switches
# ---------------------------------------------------------------------------
PAGE_OVERVIEW = [
    (["Nifty chart", "Market read"], [2.2, 1]),
    (["Signals", "Markets", "Market hours"], [1, 1, 1]),
    (["Client update"], [1]),
]
PAGE_FLOWS = [
    (["DII absorption", "FII sell days", "FII streak", "F&O positioning"], [1, 1, 1, 1]),
    (["FII flow chart", "Cumulative chart"], [1.5, 1]),
    (["FII flow calendar"], [1]),
]
PAGE_STATISTICS = [
    (["FII vs DII scatter", "Rolling correlation"], [1, 1.35]),
    (["Statistical tests", "Lead-lag chart"], [1.35, 1]),
    (["Correlation heatmap"], [1]),
    (["What if", "Daily vs monthly"], [1.3, 1]),
    (["Regime track record"], [1]),
]
PAGE_RESEARCH = [
    (["SIP phases"], [1]),
    (["Sector flows"], [1]),
]
PAGE_HEALTH = [
    (["Market mood", "Data health"], [1, 1.2]),
]
PAGE_REPORTS = [
    (["Generate report", "Methodology"], [1, 1.2]),
    (["Downloads"], [1]),
    (["Raw data"], [1]),
]

st.sidebar.markdown(
    H('<div class="brand"><div class="logo">\u25c8</div><div><div class="brand-name">FII \u00d7 NIFTY</div>'
      '<div class="brand-sub">a project by Akhil Madhu</div></div></div>'),
    unsafe_allow_html=True,
)
theme_options = ["Dark", "Light"]
_default_theme_label = "Light" if st.session_state.get("theme") == "light" else "Dark"
if hasattr(st, "segmented_control"):
    _theme_choice = st.sidebar.segmented_control("Appearance", theme_options,
                                                 default=_default_theme_label, key="theme_choice")
else:
    _theme_choice = st.sidebar.radio("Appearance", theme_options, horizontal=True, key="theme_choice",
                                     index=theme_options.index(_default_theme_label))
_new_theme = "light" if _theme_choice == "Light" else "dark"
if _new_theme != st.session_state.get("theme", "dark"):
    st.session_state["theme"] = _new_theme
    st.rerun()

st.sidebar.markdown("---")
_refresh_options = {"2 seconds": 2, "3 seconds": 3, "5 seconds": 5, "10 seconds": 10, "15 seconds": 15, "30 seconds": 30, "60 seconds": 60}
_refresh_label = st.sidebar.selectbox("Refresh every", list(_refresh_options),
                                     index=list(_refresh_options).index("5 seconds"), key="refresh_label")
st.session_state["refresh_sec"] = _refresh_options[_refresh_label]
st.sidebar.caption("Faster refresh = closer to real time, but more requests to your data sources.")
st.sidebar.markdown("---")
st.sidebar.selectbox("Statistics window", ["All sessions", "Last 60", "Last 30"], key="stat_window")
st.sidebar.slider("Granger lag (sessions)", 1, 5, 1, key="stat_lag")
st.sidebar.markdown("---")
st.sidebar.caption("Custom alert thresholds (needs Gmail/SMTP keys in .env)")
st.sidebar.number_input("Alert if FII sells more than (\u20b9 cr)", min_value=0, value=5000, step=500, key="alert_fii_sell")
st.sidebar.number_input("Alert if India VIX moves more than (%)", min_value=0.0, value=10.0, step=1.0, key="alert_vix_move")
st.sidebar.markdown("---")
force_alert = st.sidebar.button("Send test alert")
st.sidebar.caption("Set your Gmail (SMTP) keys in .env to enable alerts.")

# ---------------------------------------------------------------------------
# Analysis (recomputed at most once a minute)
# ---------------------------------------------------------------------------
# Evening: pick up freshly published FII/DII figures within ~10 minutes of release
if now_ist.weekday() < 5 and 17 <= now_ist.hour < 23:
    if time.time() - st.session_state.get("flows_refresh_ts", 0) > 600:
        provider.refresh_history_if_needed(force=True)
        st.session_state["flows_refresh_ts"] = time.time()
        st.session_state["analysis_ts"] = 0

if "analysis" not in st.session_state or time.time() - st.session_state.get("analysis_ts", 0) > 60:
    st.session_state["analysis"] = full_fii_nifty_analysis(provider.fii_history, provider.nifty_history)
    st.session_state["fii_latest"] = provider.get_fii_latest()
    st.session_state["analysis_ts"] = time.time()

analysis = st.session_state["analysis"]
fii_latest = st.session_state.get("fii_latest")

if not analysis.get("ok"):
    st.error(analysis.get("message", "Analysis failed"))
    st.stop()

latest = analysis["latest"]
regime = analysis["regime"]
corr = analysis["correlation"]
streak = analysis["streak"]
absorp = analysis["dii_absorption"]
cum = analysis["cumulative_20d"]
fo = analysis.get("fo", {}) or {}
merged_tail = analysis.get("merged_tail")
cum_series = analysis.get("cumulative_series")

alert_results = alerter.check_and_alert(analysis, force=force_alert)
if alert_results:
    st.sidebar.write("alert:", alert_results)


def _custom_alert_cooldown_ok(key: str, minutes: int = 240) -> bool:
    last = st.session_state.get("custom_alert_sent", {}).get(key, 0)
    return (time.time() - last) >= minutes * 60


def _custom_alert_mark(key: str) -> None:
    d = st.session_state.setdefault("custom_alert_sent", {})
    d[key] = time.time()


def check_custom_alerts(fii_net_val, vix_pct_change) -> None:
    """User-set numeric thresholds, independent of the regime-based AlertManager above."""
    from src.alerts.notifier import send_email
    fii_thresh = float(st.session_state.get("alert_fii_sell", 5000) or 0)
    vix_thresh = float(st.session_state.get("alert_vix_move", 10.0) or 0)

    if fii_thresh > 0 and fii_net_val is not None and fii_net_val <= -fii_thresh:
        key = f"fii_sell_{int(fii_thresh)}"
        if _custom_alert_cooldown_ok(key):
            msg = (f"FII sold more than your Rs {fii_thresh:,.0f} cr threshold: "
                  f"net Rs {fii_net_val:,.0f} cr on {flows_date_txt}.")
            if send_email("FII sell threshold hit", msg):
                _custom_alert_mark(key)

    if vix_thresh > 0 and vix_pct_change is not None and abs(vix_pct_change) >= vix_thresh:
        key = f"vix_move_{int(vix_thresh)}"
        if _custom_alert_cooldown_ok(key):
            msg = f"India VIX moved {vix_pct_change:+.1f}%, past your {vix_thresh:.0f}% threshold."
            if send_email("VIX threshold hit", msg):
                _custom_alert_mark(key)

# ---------------------------------------------------------------------------
# Derived values used across panels
# ---------------------------------------------------------------------------
hist = nifty_long_history()
if hist.empty and not provider.nifty_history.empty:
    hist = provider.nifty_history[["date", "close"]].copy()
    hist["date"] = pd.to_datetime(hist["date"])

def build_flows_frame() -> pd.DataFrame:
    """One row per session with FII, DII and Nifty return (%), used by all statistics panels."""
    try:
        nf = hist[["date", "close"]] if not hist.empty else provider.nifty_history
        merged = _prepare_merged(provider.fii_history, nf)
    except Exception:
        return pd.DataFrame()
    if merged.empty:
        return merged
    fr = merged.dropna(subset=["fii_net", "dii_net", "nifty_ret"]).reset_index(drop=True)
    fr["nifty_ret_pct"] = fr["nifty_ret"] * 100
    return fr


if "flows_frame" not in st.session_state or st.session_state.get("frame_ts") != st.session_state["analysis_ts"]:
    st.session_state["flows_frame"] = build_flows_frame()
    st.session_state["frame_ts"] = st.session_state["analysis_ts"]
flows_all = st.session_state["flows_frame"]


def build_regime_backtest(merged: pd.DataFrame, warmup: int = 15, horizons=(1, 3, 5)) -> pd.DataFrame:
    """Replays the live regime logic day by day on an expanding window (only using data that
    would have been available up to that point), then checks what Nifty actually did over the
    following 1/3/5 sessions. Same functions the live dashboard calls - nothing re-derived."""
    if merged is None or len(merged) < warmup + max(horizons) + 5:
        return pd.DataFrame()
    closes = merged["nifty_close"].to_numpy(float)
    n = len(merged)
    rows = []
    for t in range(warmup, n - 1):
        window = merged.iloc[: t + 1]
        try:
            streak_t = fii_streak(window)
            abs_t = dii_absorption(window, lookback=15)
            fo_t = fo_positioning(window)
            reg_t = regime_from_flows(window, streak_t, abs_t, fo_t)
        except Exception:
            continue
        row = {"date": merged["date"].iloc[t], "regime": reg_t.get("regime", "NEUTRAL")}
        for h in horizons:
            if t + h < n:
                row[f"fwd_{h}d"] = (closes[t + h] / closes[t] - 1) * 100
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_backtest(bt: pd.DataFrame, horizons=(1, 3, 5)) -> pd.DataFrame:
    if bt is None or bt.empty:
        return pd.DataFrame()
    out = []
    for reg_name, g in bt.groupby("regime"):
        row = {"regime": reg_name, "n": len(g)}
        for h in horizons:
            col = f"fwd_{h}d"
            if col in g.columns and g[col].notna().sum() >= 5:
                vals = g[col].dropna()
                row[f"avg_{h}d"] = float(vals.mean())
                row[f"hit_{h}d"] = float((vals > 0).mean() * 100)
                row[f"n_{h}d"] = int(len(vals))
            else:
                row[f"avg_{h}d"] = None
                row[f"hit_{h}d"] = None
                row[f"n_{h}d"] = int(g[col].notna().sum()) if col in g.columns else 0
        out.append(row)
    return pd.DataFrame(out)


if "regime_bt" not in st.session_state or st.session_state.get("bt_ts") != st.session_state["analysis_ts"]:
    _merged_full = analysis.get("merged_tail")
    # merged_tail is only the last 30 rows; the backtest wants the full overlap, so rebuild it
    try:
        _bt_merged = _prepare_merged(provider.fii_history,
                                     hist[["date", "close"]] if not hist.empty else provider.nifty_history)
    except Exception:
        _bt_merged = pd.DataFrame()
    st.session_state["regime_bt"] = build_regime_backtest(_bt_merged)
    st.session_state["bt_ts"] = st.session_state["analysis_ts"]
regime_bt = st.session_state["regime_bt"]
_win = {"All sessions": None, "Last 60": 60, "Last 30": 30}[st.session_state.get("stat_window", "All sessions")]
flows_win = flows_all.tail(_win) if (_win and not flows_all.empty) else flows_all
stat_lag = int(st.session_state.get("stat_lag", 1))

status_label, status_color, status_note, status_prog = market_status(now_ist)

price = live.price if live else latest.get("nifty_close")
source = live.source if live else "history"
day_pct = None
if not hist.empty and price:
    closes = hist["close"].astype(float)
    if status_label == "Open":
        last_is_today = hist["date"].iloc[-1].date() == now_ist.date()
        base = closes.iloc[-2] if (last_is_today and len(closes) > 1) else closes.iloc[-1]
        day_pct = (price / base - 1) * 100
    elif len(closes) > 1:
        day_pct = (closes.iloc[-1] / closes.iloc[-2] - 1) * 100
if day_pct is None:
    day_pct = latest.get("nifty_ret_pct")

fii_net = latest.get("fii_net_cr")
dii_net = latest.get("dii_net_cr")
flows_date = latest.get("date")
try:
    flows_date_txt = datetime.strptime(flows_date, "%Y-%m-%d").strftime("%d %b")
except (TypeError, ValueError):
    flows_date_txt = "latest"

_vix_chg = next((w["chg"] for w in fetch_watchlist() if w["name"] == "India VIX"), None)
check_custom_alerts(fii_net, _vix_chg)

reg = regime.get("regime", "NEUTRAL")
reg_conf = float(regime.get("confidence") or 0)
s_dir = streak.get("direction", "NONE")
s_len = int(streak.get("streak", 0) or 0)
fo_bias = (fo.get("fo_bias") or "NEUTRAL").upper()
same_day = corr.get("same_day")
abs_pct = absorp.get("avg_absorption_pct")


def build_insight():
    plural = "s" if s_len != 1 else ""
    tone = "neutral"
    if reg in ("FII_SELL_PRESSURE", "MILD_FII_OUTFLOW"):
        tone = "bearish"
    elif reg in ("FII_LED_RALLY", "MILD_FII_INFLOW"):
        tone = "bullish"

    if reg == "FII_SELL_PRESSURE":
        head = (f"FIIs have sold for {s_len} straight session{plural}, and index-futures positioning is "
                f"leaning {fo_bias.lower()} too. The setup currently favours caution over fresh longs.")
    elif reg == "DII_SUPPORTED":
        head = ("FIIs are selling, but domestic funds are absorbing enough of it that the index is "
                "holding up. It is a tug-of-war rather than a clear trend.")
    elif reg == "FII_LED_RALLY":
        head = (f"FIIs have been net buyers for {s_len} session{plural}, and futures positioning "
                f"supports it. Conditions currently favour the upside.")
    elif reg == "MILD_FII_OUTFLOW":
        head = ("FII selling looks mild rather than a sustained streak. Worth watching for whether it "
                "extends, but not yet a strong signal either way.")
    elif reg == "MILD_FII_INFLOW":
        head = "FII buying looks mild: a modest tailwind, not yet a strong signal."
    else:
        head = ("Flows are mixed with no dominant push in either direction. Price structure and "
                "volatility likely matter more than flows right now.")

    sup = []
    if same_day is not None:
        if same_day > 0.25:
            sup.append(f"Same-day FII flow and Nifty returns are correlated at {same_day:.2f} recently, "
                       f"so flow direction has tended to line up with the day's move.")
        else:
            sup.append(f"Same-day correlation is only {same_day:.2f} recently, so flows alone are not "
                       f"explaining much of the daily move.")
    if abs_pct is not None:
        sup.append(f"On FII sell days over the last {absorp.get('lookback')} sessions, DIIs absorbed about "
                   f"{abs_pct:.0f}% of the selling on average.")
    if fo_bias != "NEUTRAL":
        agrees = (fo_bias == "BEARISH" and tone == "bearish") or (fo_bias == "BULLISH" and tone == "bullish")
        sup.append(f"FII index-futures positioning is net {fo_bias.lower()}, which tends to "
                   f"{'reinforce' if agrees else 'complicate'} the cash-market picture.")
    return tone, head, sup


tone, headline, support = build_insight()

# ---------------------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------------------
def chip(label: str, value: str, pill: str = "") -> str:
    return (f'<div class="chip"><div class="chip-label">{E(label)}</div>'
            f'<div class="chip-row"><span class="chip-val">{value}</span>{pill}</div></div>')


def pill(v, text: str) -> str:
    arrow = "\u2191" if (v or 0) > 0 else ("\u2193" if (v or 0) < 0 else "")
    return f'<span class="pill {tone_cls(v)}">{arrow} {text}</span>'


nifty_chip = chip("Nifty 50", f'<span data-animate="an-nifty">{fnum(price, 2)}</span>',
                  pill(day_pct, f"{abs(day_pct):.2f}%") if day_pct is not None else "")
fii_chip = chip(f"FII net \u00b7 {flows_date_txt}",
               f'<span style="color:{LIME if (fii_net or 0) >= 0 else RED}">\u20b9'
               f'<span data-animate="an-fii">{fnum(fii_net, 0, True)}</span> cr</span>')
dii_chip = chip(f"DII net \u00b7 {flows_date_txt}",
               f'<span style="color:{LIME if (dii_net or 0) >= 0 else RED}">\u20b9'
               f'<span data-animate="an-dii">{fnum(dii_net, 0, True)}</span> cr</span>')

st.markdown(H(f"""
<div class="topbar">
  <div class="brand"><div class="logo">\u25c8</div>
    <div><div class="brand-name">FII \u00d7 NIFTY</div><div class="brand-sub">a project by Akhil Madhu</div></div></div>
  {nifty_chip}{fii_chip}{dii_chip}
  <div class="status"><span class="pulse"></span>live \u00b7 {E(source)} \u00b7 {now_ist.strftime('%H:%M:%S')} IST</div>
</div>"""), unsafe_allow_html=True)

# ---- Count-up: animate each number from its previous displayed value to
# its current one. First render of the session has no "previous" value,
# so nothing animates until the number actually changes on a refresh.
_anim_targets = [
    dict(id="an-nifty", to=float(price or 0), dp=2, signed=False),
    dict(id="an-fii", to=float(fii_net or 0), dp=0, signed=True),
    dict(id="an-dii", to=float(dii_net or 0), dp=0, signed=True),
]
_anim_prev = st.session_state.get("anim_prev", {})
for t in _anim_targets:
    t["from"] = _anim_prev.get(t["id"], t["to"])
st.session_state["anim_prev"] = {t["id"]: t["to"] for t in _anim_targets}
try:
    components.html(f"""
<script>
(function() {{
  var targets = {json.dumps(_anim_targets)};
  function fmt(v, dp, signed) {{
    var s = Math.abs(v).toLocaleString('en-US', {{minimumFractionDigits: dp, maximumFractionDigits: dp}});
    var sign = v < 0 ? '-' : (signed ? '+' : '');
    return sign + s;
  }}
  function animate(el, from, to, dp, signed) {{
    var start = null, ms = 650;
    function step(ts) {{
      if (!start) start = ts;
      var p = Math.min((ts - start) / ms, 1);
      el.textContent = fmt(from + (to - from) * p, dp, signed);
      if (p < 1) window.requestAnimationFrame(step);
    }}
    window.requestAnimationFrame(step);
  }}
  try {{
    var doc = window.parent.document;
    targets.forEach(function(t) {{
      var el = doc.querySelector('[data-animate="' + t.id + '"]');
      if (el) animate(el, t.from, t.to, t.dp, t.signed);
    }});
  }} catch (e) {{ /* fine - the plain numbers above are already correct */ }}
}})();
</script>
""", height=0)
except AttributeError:
    pass  # older/newer Streamlit without components.v1.html - numbers above are already correct, just no animation

# ---- Scrolling ticker of live-ish prices ----
_tick_items = []
for w in fetch_watchlist():
    if w["last"] is None:
        continue
    val = f'{w["pre"]}{w["last"]:,.{w["dp"]}f}{w["suf"]}'
    chg = w["chg"]
    c = LIME if (chg or 0) > 0 else (RED if (chg or 0) < 0 else MUTED)
    unit = " bp" if w["mode"] == "bp" else "%"
    dp2 = 1 if w["mode"] == "bp" else 2
    chg_txt = f'{chg:+.{dp2}f}{unit}' if chg is not None else "n/a"
    _tick_items.append(f'<span class="ticker-item">{E(w["name"])} <b>{E(val)}</b> '
                       f'<span style="color:{c}">{E(chg_txt)}</span></span>')
if price:
    nifty_txt = f'{fnum(price, 2)}' + (f' <span style="color:{LIME if day_pct >= 0 else RED}">{day_pct:+.2f}%</span>' if day_pct is not None else "")
    _tick_items.insert(0, f'<span class="ticker-item">Nifty 50 <b>{nifty_txt}</b></span>')
if _tick_items:
    _track = "".join(_tick_items) * 2
    st.markdown(H(f'<div class="ticker-wrap"><div class="ticker-track">{_track}</div></div>'),
               unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def hero_figure(x, y, phases: bool = False) -> go.Figure:
    y = pd.Series(list(y), dtype=float)
    x = pd.Series(list(x))
    lo, hi = float(y.min()), float(y.max())
    pad = (hi - lo) * 0.14 or hi * 0.002
    y0, y1 = lo - pad, hi + pad
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=[y0] * len(x), mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    common = dict(x=x, y=y, mode="lines", name="Nifty 50",
                  line=dict(color=LIME, width=2.6, shape="spline", smoothing=0.6),
                  hovertemplate="<b>\u20b9%{y:,.2f}</b><extra></extra>")
    try:
        trace = go.Scatter(fill="tonexty", fillgradient=dict(
            type="vertical", start=y0, stop=y1,
            colorscale=[[0, "rgba(184,242,59,0.0)"], [1, "rgba(184,242,59,0.34)"]]), **common)
    except (ValueError, TypeError):
        trace = go.Scatter(fill="tonexty", fillcolor="rgba(184,242,59,0.12)", **common)
    fig.add_trace(trace)
    for size, op, edge in ((26, 0.12, 0), (15, 0.25, 0), (8, 1.0, 2)):
        fig.add_trace(go.Scatter(x=[x.iloc[-1]], y=[y.iloc[-1]], mode="markers", showlegend=False,
                                 hoverinfo="skip",
                                 marker=dict(size=size, color=LIME, opacity=op, line=dict(color=BG, width=edge))))
    if phases:
        xmin, xmax = pd.Timestamp(x.min()), pd.Timestamp(x.max())
        for pname, ps, pe, pc in SIP_PHASES:
            x0, x1 = max(pd.Timestamp(ps), xmin), min(pd.Timestamp(pe), xmax)
            if x0 >= x1:
                continue
            fig.add_shape(type="rect", xref="x", yref="paper", x0=x0.to_pydatetime(), x1=x1.to_pydatetime(),
                          y0=0, y1=1, fillcolor=hex_rgba(pc, 0.10), line_width=0, layer="below")
            fig.add_annotation(x=x0.to_pydatetime(), y=1, xref="x", yref="paper", text=pname, showarrow=False,
                               xanchor="left", yanchor="top", xshift=6, font=dict(size=11, color=pc))
    spike = dict(showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dash",
                 spikecolor="rgba(184,242,59,.5)", spikethickness=1)
    fig.update_layout(
        height=330, margin=dict(l=4, r=4, t=6, b=4), showlegend=False, hovermode="x",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif", color=MUTED, size=12),
        hoverlabel=dict(bgcolor="#13261C", bordercolor=LIME,
                        font=dict(family="Manrope, sans-serif", color=TEXT, size=13)),
        xaxis=dict(showgrid=False, zeroline=False, showline=False, **spike),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", zeroline=False, showline=False,
                   range=[y0, y1], tickformat=",.0f", **spike),
    )
    return fig


def panel_nifty_chart():
    with card("hero"):
        top_l, top_r = st.columns([1.5, 1])
        with top_l:
            pill_html = pill(day_pct, f"{abs(day_pct):.2f}%") if day_pct is not None else ""
            st.markdown(H(f"""
            <div class="card-title">Nifty 50</div>
            <div class="hero-row"><span class="hero-val">{fnum(price, 2)}</span>{pill_html}</div>
            <div class="hero-sub">{E(source)} \u00b7 {'live tick' if live else 'last close'}</div>"""),
                        unsafe_allow_html=True)
        show_phases = False
        with top_r:
            options = ["Live", "1M", "3M", "6M", "1Y", "5Y"]
            if hasattr(st, "segmented_control"):
                rng = st.segmented_control("Range", options, default="6M", key="range",
                                           label_visibility="collapsed")
            else:
                rng = st.radio("Range", options, index=3, horizontal=True, key="range",
                               label_visibility="collapsed")
            if hasattr(st, "toggle"):
                show_phases = st.toggle("Show my SIP phases", key="show_phases")
            else:
                show_phases = st.checkbox("Show my SIP phases", key="show_phases")
        rng = rng or "6M"

        if rng == "Live":
            intraday = store.realtime.to_dataframe("NIFTY50")
            if intraday.empty or len(intraday) < 2:
                st.markdown(H('<div class="card-sub" style="padding:80px 0;text-align:center">'
                              'Collecting live ticks. A new point arrives about every 30 seconds while this page '
                              'is open, so the line builds up in front of you.</div>'), unsafe_allow_html=True)
                return
            idx = pd.to_datetime(intraday.index)
            if idx.tz is not None:
                idx = idx.tz_convert(IST).tz_localize(None)
            plot(hero_figure(idx, intraday["price"]), "fig-hero")
            st.caption(f"{len(intraday)} ticks captured this session")
            first_p, last_p = float(intraday["price"].iloc[0]), float(intraday["price"].iloc[-1])
            chg = last_p - first_p
            chg_pct = (chg / first_p * 100) if first_p else 0.0
            move = "gone up" if chg > 0 else ("gone down" if chg < 0 else "stayed flat")
            explain(f"Right now, Nifty 50 is at <b>{last_p:,.2f}</b>. Since you opened this page, it has "
                   f"{move} by about <b>{abs(chg):,.1f} points</b> ({chg_pct:+.2f}%). This updates every "
                   f"30 seconds on its own.")
        else:
            if hist.empty:
                st.warning("Price history is unavailable right now.")
                return
            days = {"1M": 31, "3M": 92, "6M": 183, "1Y": 366, "5Y": 1830}[rng]
            sub = hist[hist["date"] >= hist["date"].max() - timedelta(days=days)]
            plot(hero_figure(sub["date"], sub["close"], phases=bool(show_phases)), "fig-hero")
            if len(sub) >= 2:
                c0, c1 = float(sub["close"].iloc[0]), float(sub["close"].iloc[-1])
                hi, lo = float(sub["close"].max()), float(sub["close"].min())
                chg_pct = (c1 / c0 - 1) * 100 if c0 else 0.0
                move = "went up" if chg_pct > 0 else ("went down" if chg_pct < 0 else "stayed about the same")
                explain(f"Over {RANGE_PHRASE.get(rng, 'this period')}, Nifty 50 {move} by "
                       f"<b>{abs(chg_pct):.1f}%</b> - it started around <b>{c0:,.0f}</b> and is now around "
                       f"<b>{c1:,.0f}</b>. The highest point in this time was about <b>{hi:,.0f}</b>, and the "
                       f"lowest was about <b>{lo:,.0f}</b>.")


def panel_market_read():
    color = TONE_COLOR[tone]
    bullets = "".join(f'<div class="bullet"><span class="dot" style="background:{color}"></span><div>{E(s)}</div></div>'
                      for s in support)
    st.markdown(H(f"""
    <div class="card" style="min-height:470px">
      <div class="card-head"><div class="card-title">Market read</div>
        <span class="pill {TONE_CLS[tone]}">{tone}</span></div>
      <div class="read-headline">{E(headline)}</div>
      {bullets}
      <div class="conf"><div class="conf-label"><span>Signal confidence \u00b7 {E(REGIME_LABEL.get(reg, reg))}</span>
        <b style="color:var(--text)">{reg_conf * 100:.0f}%</b></div>
        <div class="conf-bar"><span style="width:{reg_conf * 100:.0f}%;background:{color}"></span></div></div>
      <div class="read-foot">{E(regime.get('reason') or '')}<br>Descriptive read of flow data, not investment advice.</div>
    </div>"""), unsafe_allow_html=True)


def panel_absorption():
    p = abs_pct
    L = 188.5
    frac = 0 if p is None else max(0.0, min(100.0, p)) / 100
    fill = (f'<path d="M10 70 A60 60 0 0 1 130 70" fill="none" stroke="{LIME}" stroke-width="12" '
            f'stroke-linecap="round" stroke-dasharray="{frac * L:.1f} {L}" class="gauge-fill"/>') if frac > 0 else ""
    st.markdown(H(f"""
    <div class="card kpi">
      <div class="kpi-label">DII absorption</div>
      <div class="gauge-wrap">
        <svg viewBox="0 0 140 80" class="gauge"><path d="M10 70 A60 60 0 0 1 130 70" fill="none"
          stroke="{RED}" stroke-opacity=".8" stroke-width="12" stroke-linecap="round"/>{fill}</svg>
        <div class="gauge-val">{'n/a' if p is None else f'{p:.0f}%'}</div>
      </div>
      <div class="kpi-sub" style="text-align:center">of FII selling absorbed by DIIs, last {E(absorp.get('lookback', '\u2014'))} sessions</div>
      {mini_sparkline(flows_all["dii_net"].tail(int(absorp.get("lookback") or 20)), LIME) if not flows_all.empty else ""}
    </div>"""), unsafe_allow_html=True)


def panel_sell_days():
    total = int(absorp.get("lookback") or 0)
    sell = int(absorp.get("fii_sell_days") or 0)
    sell_w = (sell / total * 100) if total else 0
    st.markdown(H(f"""
    <div class="card kpi">
      <div class="kpi-label">FII sell days</div>
      <div class="kpi-val">{sell} <span class="mut" style="font-size:18px;font-weight:600">/ {total}</span></div>
      <div class="splitbar"><span style="width:{sell_w:.1f}%;background:{RED}"></span>
        <span style="width:{100 - sell_w:.1f}%;background:{LIME}"></span></div>
      <div class="legend"><span><i style="background:{RED}"></i>selling</span><span><i style="background:{LIME}"></i>buying</span></div>
      <div class="kpi-sub">FII net cash flow, last {total} sessions</div>
      {mini_sparkline(flows_all["fii_net"].tail(total or 20), RED) if not flows_all.empty else ""}
    </div>"""), unsafe_allow_html=True)


def panel_streak():
    color = RED if s_dir == "SELL" else (LIME if s_dir == "BUY" else MUTED)
    arrow = "\u2193" if s_dir == "SELL" else ("\u2191" if s_dir == "BUY" else "\u2013")
    label = f"{s_dir} \u00d7 {s_len}d" if s_dir in ("SELL", "BUY") else "No streak"
    st.markdown(H(f"""
    <div class="card kpi">
      <div class="kpi-label">FII streak</div>
      <div style="display:flex;align-items:center;gap:12px">
        <div class="dir-badge" style="background:{color}22;color:{color}">{arrow}</div>
        <div class="kpi-val" style="color:{color}">{E(label)}</div></div>
      <div class="kpi-sub">Cumulative \u20b9{fnum(streak.get('cum_flow_cr'), 0, True)} cr<br>
        Last session \u20b9{fnum(streak.get('last_net_cr'), 0, True)} cr</div>
      {mini_sparkline(flows_all["fii_net"].tail(20), color) if not flows_all.empty else ""}
    </div>"""), unsafe_allow_html=True)


def panel_fo():
    lo, sh = fo.get("fii_idx_fut_long"), fo.get("fii_idx_fut_short")
    ratio = fo.get("fii_ls_ratio")
    long_w = None
    try:
        if lo is not None and sh is not None and (lo + sh) > 0:
            long_w = lo / (lo + sh) * 100
    except TypeError:
        pass
    bar = (f'<div class="splitbar"><span style="width:{long_w:.1f}%;background:{LIME}"></span>'
           f'<span style="width:{100 - long_w:.1f}%;background:{RED}"></span></div>'
           f'<div class="legend"><span><i style="background:{LIME}"></i>long</span>'
           f'<span><i style="background:{RED}"></i>short</span></div>') if long_w is not None else ""
    st.markdown(H(f"""
    <div class="card kpi">
      <div class="kpi-label">FII index futures \u00b7 long/short</div>
      <div class="kpi-val">{fnum(ratio, 2)}</div>
      {bar}
      <div class="kpi-sub">{E(fo_bias.title())} \u00b7 net {fnum(fo.get('fii_idx_fut_net'), 0, True)} contracts</div>
      {mini_sparkline(merged_tail["fii_ls_ratio"].tail(20), AMBER) if (merged_tail is not None and "fii_ls_ratio" in merged_tail.columns) else ""}
    </div>"""), unsafe_allow_html=True)


def sig_row(title, sub, val, color):
    return (f'<div class="row"><span class="sdot" style="background:{color}"></span>'
            f'<div class="row-main"><div class="row-title">{E(title)}</div><div class="row-sub">{E(sub)}</div></div>'
            f'<div class="row-val">{E(val)}</div></div>')


def panel_signals():
    dir_color = RED if s_dir == "SELL" else (LIME if s_dir == "BUY" else MUTED)
    fo_color = RED if fo_bias == "BEARISH" else (LIME if fo_bias == "BULLISH" else AMBER)
    cf = cum.get("cum_fii_cr")
    rows = [
        sig_row("FII streak", f"cumulative \u20b9{fnum(streak.get('cum_flow_cr'), 0, True)} cr",
                f"{s_dir} \u00d7 {s_len}d" if s_len else "none", dir_color),
        sig_row("F&O bias", f"long/short {fnum(fo.get('fii_ls_ratio'), 2)}", fo_bias.lower(), fo_color),
        sig_row("Same-day correlation", "FII net vs Nifty return", fnum(same_day, 2),
                LIME if (same_day or 0) > 0.25 else AMBER),
        sig_row("Best lag", f"correlation {fnum(corr.get('best_corr'), 2)}",
                f"t+{corr.get('best_lag')}" if corr.get("best_lag") is not None else "\u2014", AMBER),
        sig_row("20-day FII flow", f"Nifty 20d {fnum(cum.get('nifty_return_pct'), 2, True)}%",
                f"\u20b9{fnum(cf, 0, True)} cr", RED if (cf or 0) < 0 else LIME),
        sig_row("DII cover", "avg on FII sell days", "n/a" if abs_pct is None else f"{abs_pct:.0f}%",
                LIME if (abs_pct or 0) >= 80 else AMBER),
    ]
    st.markdown(H(f'<div class="card" style="min-height:420px"><div class="card-head"><div class="card-title">Signals</div>'
                  f'<span class="card-sub">from flow data</span></div>{"".join(rows)}</div>'), unsafe_allow_html=True)



def _percentile_rank(value, series) -> float:
    """% of the series at or below `value`. 50.0 is returned when there isn't enough history to
    judge - a neutral fallback rather than a misleadingly confident number."""
    s = pd.Series(series).dropna() if series is not None else pd.Series(dtype=float)
    if value is None or len(s) < 5:
        return 50.0
    return float((s <= value).mean() * 100)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_vix_history() -> pd.Series:
    try:
        import yfinance as yf
        h = yf.Ticker("^INDIAVIX").history(period="6mo", interval="1d")["Close"].dropna()
        return h
    except Exception:
        return pd.Series(dtype=float)


MOOD_WEIGHTS = {"fii": 0.35, "dii": 0.15, "vix": 0.30, "fo": 0.20}


def compute_mood_score() -> dict:
    fii_pct = _percentile_rank(fii_net, flows_all["fii_net"] if not flows_all.empty else None)
    dii_pct = _percentile_rank(dii_net, flows_all["dii_net"] if not flows_all.empty else None)
    vix_hist = fetch_vix_history()
    vix_now = float(vix_hist.iloc[-1]) if len(vix_hist) else None
    vix_pct_raw = _percentile_rank(vix_now, vix_hist)
    vix_component = 100.0 - vix_pct_raw  # inverted: a low VIX versus its own recent history scores high
    fo_component = {"BEARISH": 20.0, "NEUTRAL": 50.0, "BULLISH": 80.0}.get(fo_bias, 50.0)
    w = MOOD_WEIGHTS
    score = (fii_pct * w["fii"] + dii_pct * w["dii"] + vix_component * w["vix"] + fo_component * w["fo"])
    return dict(score=score, fii_pct=fii_pct, dii_pct=dii_pct, vix_component=vix_component,
               vix_now=vix_now, fo_component=fo_component)


def mood_label(score: float):
    if score < 25:
        return "Extreme Fear", RED
    if score < 45:
        return "Fear", RED
    if score < 56:
        return "Neutral", AMBER
    if score < 76:
        return "Greed", LIME
    return "Extreme Greed", LIME


def panel_market_mood():
    m = compute_mood_score()
    label, color = mood_label(m["score"])
    frac = max(0.0, min(1.0, m["score"] / 100))
    L = 188.5
    with card("mood"):
        st.markdown(H('<div class="card-title">Market mood score</div>'
                      '<div class="card-sub">one combined read of FII flow, DII flow, India VIX, and F&amp;O positioning</div>'),
                    unsafe_allow_html=True)
        st.markdown(H(f'''
        <div class="gauge-wrap" style="width:210px">
          <svg viewBox="0 0 140 80" class="gauge" style="width:210px">
            <path d="M10 70 A60 60 0 0 1 130 70" fill="none" stroke="{MUTED}" stroke-opacity=".3" stroke-width="14" stroke-linecap="round"/>
            <path d="M10 70 A60 60 0 0 1 130 70" fill="none" stroke="{color}" stroke-width="14" stroke-linecap="round"
                  stroke-dasharray="{frac * L:.1f} {L}" class="gauge-fill"/>
          </svg>
          <div class="gauge-val" style="font-size:28px">{m["score"]:.0f}</div>
        </div>
        <div style="text-align:center;margin-top:2px">
          <span class="pill" style="background:{color}22;color:{color};font-size:13px;padding:4px 14px">{label}</span>
        </div>'''), unsafe_allow_html=True)

        st.markdown(H(f'''<div class="tgroup">What feeds into this score</div>
        <div class="trow"><div><div class="tname">FII flow</div><div class="tsub">today vs. recent history, weight 35%</div></div>
          <div class="tstat">{m["fii_pct"]:.0f}/100</div><div class="tp"></div><div class="tv"></div></div>
        <div class="trow"><div><div class="tname">DII flow</div><div class="tsub">today vs. recent history, weight 15%</div></div>
          <div class="tstat">{m["dii_pct"]:.0f}/100</div><div class="tp"></div><div class="tv"></div></div>
        <div class="trow"><div><div class="tname">India VIX</div><div class="tsub">{fnum(m["vix_now"], 1)} today, inverted vs. recent history, weight 30%</div></div>
          <div class="tstat">{m["vix_component"]:.0f}/100</div><div class="tp"></div><div class="tv"></div></div>
        <div class="trow"><div><div class="tname">F&amp;O positioning</div><div class="tsub">{E(fo_bias.title())}, weight 20%</div></div>
          <div class="tstat">{m["fo_component"]:.0f}/100</div><div class="tp"></div><div class="tv"></div></div>
        '''), unsafe_allow_html=True)

        drivers = sorted(
            [("FII flow", m["fii_pct"], MOOD_WEIGHTS["fii"]), ("DII flow", m["dii_pct"], MOOD_WEIGHTS["dii"]),
             ("India VIX", m["vix_component"], MOOD_WEIGHTS["vix"]), ("F&O positioning", m["fo_component"], MOOD_WEIGHTS["fo"])],
            key=lambda t: abs(t[1] - 50) * t[2], reverse=True)
        top_driver = drivers[0][0]
        st.markdown(H(f'<div class="tgroup">Interpretation</div>'
                      f'<div class="read-foot">The score is {m["score"]:.0f}, in the "{label}" range. {top_driver} is '
                      f'currently the largest contributor. Each input is a percentile rank against its own recent '
                      f'history, so this reflects how today compares to the recent past, not a fixed threshold. The '
                      f'weights (FII 35%, DII 15%, VIX 30%, F&amp;O 20%) are a disclosed starting point, not a validated '
                      f'model - treat this as one more data point alongside the rest of the dashboard, not a signal to '
                      f'act on by itself.</div>'), unsafe_allow_html=True)


def panel_data_health():
    with card("health"):
        st.markdown(H('<div class="card-title">Data health</div>'
                      '<div class="card-sub">status of each data source behind this dashboard</div>'),
                    unsafe_allow_html=True)
        nifty_ok = live is not None
        fii_ok = not provider.fii_history.empty
        last_fii = provider.fii_history["date"].max().strftime("%d %b %Y") if fii_ok else None
        sect_ok = len(fetch_sector_series()) > 0
        fpi_csv = load_fpi_sector_csv()
        fpi_ok = fpi_csv is not None and not fpi_csv.empty
        sip_csv = load_sip_monthly()
        sip_ok = sip_csv is not None and not sip_csv.empty
        alerts_ok = bool(os.getenv("SMTP_USER") and os.getenv("SMTP_PASS"))

        rows = [
            ("Nifty 50 price", LIME if nifty_ok else RED,
             f"live via {source}" if nifty_ok else "no live tick right now - showing last close"),
            ("FII / DII flow data", LIME if fii_ok else RED,
             f"latest session {last_fii}" if fii_ok else "unavailable - check the terminal for a 403 or connection error"),
            ("Sector price data", LIME if sect_ok else AMBER,
             "live via Yahoo Finance" if sect_ok else "unavailable right now (see the Sector flows page)"),
            ("NSDL sector-wise FPI file", LIME if fpi_ok else AMBER,
             "data/fpi_sector.csv loaded" if fpi_ok else "optional file not added yet"),
            ("Monthly SIP research file", LIME if sip_ok else AMBER,
             "data/sip_monthly.csv loaded" if sip_ok else "optional file not added yet"),
            ("Alerts (email)", LIME if alerts_ok else AMBER,
             "configured" if alerts_ok else "not configured - set keys in .env or Secrets"),
        ]
        rows_html = "".join(
            f'<div class="row"><span class="sdot" style="background:{c}"></span>'
            f'<div class="row-main"><div class="row-title">{E(n)}</div><div class="row-sub">{E(d)}</div></div></div>'
            for n, c, d in rows)
        st.markdown(H(rows_html), unsafe_allow_html=True)

        n_ok = sum(1 for _, c, _ in rows if c == LIME)
        critical_ok = nifty_ok and fii_ok
        st.markdown(H(
            f'<div class="tgroup">Interpretation</div>'
            f'<div class="read-foot">{n_ok} of {len(rows)} data sources are fully live. ' +
            ('Both critical feeds - Nifty price and FII/DII flow - are working, so the core dashboard is reliable '
             'right now.' if critical_ok else
             'At least one critical feed (Nifty price or FII/DII flow) is currently down, so figures elsewhere in '
             'the dashboard may be stale or running on a fallback source.') +
            ' Sector data and the two optional CSV files only affect specific panels, not the core numbers.</div>'),
            unsafe_allow_html=True)


def panel_markets():
    rows = []
    for w in fetch_watchlist():
        if w["last"] is None:
            val, chg_html = "\u2014", '<div class="row-chg flat">n/a</div>'
        else:
            val = f'{w["pre"]}{w["last"]:,.{w["dp"]}f}{w["suf"]}'
            if w["chg"] is None:
                chg_html = '<div class="row-chg flat">n/a</div>'
            else:
                unit = " bp" if w["mode"] == "bp" else "%"
                dp = 1 if w["mode"] == "bp" else 2
                chg_html = f'<div class="row-chg {tone_cls(w["chg"])}">{w["chg"]:+.{dp}f}{unit}</div>'
        rows.append(f'<div class="row"><div class="badge">{E(w["badge"])}</div>'
                    f'<div class="row-main"><div class="row-title">{E(w["name"])}</div></div>'
                    f'<div class="row-val">{E(val)}{chg_html}</div></div>')
    st.markdown(H(f'<div class="card" style="min-height:420px"><div class="card-head"><div class="card-title">Markets</div>'
                  f'<span class="card-sub">delayed \u00b7 vs prev close</span></div>{"".join(rows)}</div>'),
                unsafe_allow_html=True)


def panel_hours():
    st.markdown(H(f"""
    <div class="card" style="min-height:420px">
      <div class="card-head"><div class="card-title">Market hours</div><span class="card-sub">NSE \u00b7 IST</span></div>
      <div class="hours-status"><span class="sdot" style="background:{status_color};width:10px;height:10px"></span>
        <span class="hours-big" style="color:{status_color}">{status_label}</span></div>
      <div class="card-sub">{E(status_note)}</div>
      <div class="track"><span style="width:{status_prog * 100:.1f}%"></span></div>
      <div class="track-labels"><span>9:15</span><span>Regular session</span><span>15:30</span></div>
      <div class="row-sub" style="margin-top:22px;line-height:1.6">Pre-open 9:00\u20139:15.<br>
        FII and DII flow figures are published after the close, so they update in the evening, not live.<br>
        Weekday schedule only; exchange holidays are not included.</div>
    </div>"""), unsafe_allow_html=True)


def _flow_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=4, r=4, t=4, b=4), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
        font=dict(family="Manrope, sans-serif", color=MUTED, size=12),
        legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#13261C", bordercolor=LIME, font=dict(color=TEXT)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.04)", zeroline=False)
    return fig


def panel_flow_chart():
    with card("flows"):
        st.markdown(H('<div class="card-title">FII net flow vs Nifty 50</div>'
                      '<div class="card-sub">last 30 sessions \u00b7 bars are \u20b9 crore</div>'), unsafe_allow_html=True)
        if merged_tail is None or merged_tail.empty:
            st.caption("No flow history available.")
            return
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=merged_tail["date"], y=merged_tail["fii_net"], name="FII net (\u20b9 cr)",
                             marker_color=[LIME if v >= 0 else RED for v in merged_tail["fii_net"]],
                             marker_line_width=0, opacity=0.9), secondary_y=False)
        fig.add_trace(go.Scatter(x=merged_tail["date"], y=merged_tail["nifty_close"], name="Nifty 50",
                                 line=dict(color=TEXT, width=2, shape="spline", smoothing=0.5)), secondary_y=True)
        plot(_flow_layout(fig, 320), "fig-flow")
        buy_d = int((merged_tail["fii_net"] > 0).sum())
        sell_d = int((merged_tail["fii_net"] < 0).sum())
        total_fii = float(merged_tail["fii_net"].sum())
        c0n, c1n = float(merged_tail["nifty_close"].iloc[0]), float(merged_tail["nifty_close"].iloc[-1])
        nifty_chg = (c1n / c0n - 1) * 100 if c0n else 0.0
        fii_word = "bought" if total_fii >= 0 else "sold"
        explain(f"Each green bar is a day foreign investors (FIIs) bought more shares than they sold; each red "
               f"bar is a day they sold more than they bought. Over the last 30 trading days, FIIs were net "
               f"sellers on <b>{sell_d}</b> days and net buyers on <b>{buy_d}</b> days. Added up, they "
               f"{fii_word} about <b>\u20b9{abs(total_fii):,.0f} crore</b> worth of shares overall. Nifty "
               f"{'rose' if nifty_chg >= 0 else 'fell'} by <b>{abs(nifty_chg):.1f}%</b> over the same days.")


def panel_cum_chart():
    with card("cum"):
        st.markdown(H('<div class="card-title">20-day cumulative flows</div>'
                      '<div class="card-sub">FII vs DII against Nifty</div>'), unsafe_allow_html=True)
        if cum_series is None or len(cum_series) == 0:
            st.caption("Not enough history yet.")
            return
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=cum_series["date"], y=cum_series["cum_fii"], name="FII 20d",
                                 line=dict(color=RED, width=2, shape="spline", smoothing=0.4)), secondary_y=False)
        fig.add_trace(go.Scatter(x=cum_series["date"], y=cum_series["cum_dii"], name="DII 20d",
                                 line=dict(color=LIME, width=2, shape="spline", smoothing=0.4)), secondary_y=False)
        fig.add_trace(go.Scatter(x=cum_series["date"], y=cum_series["nifty_close"], name="Nifty",
                                 line=dict(color=MUTED, width=1.4, dash="dot")), secondary_y=True)
        plot(_flow_layout(fig, 320), "fig-cum")
        last_row = cum_series.iloc[-1]
        cf, cd = float(last_row.get("cum_fii", 0) or 0), float(last_row.get("cum_dii", 0) or 0)
        explain(f"This adds up FII and DII buying or selling over a rolling 20-trading-day window, like a "
               f"running total that always looks back at the most recent month. Right now, that running "
               f"total is <b>\u20b9{cf:+,.0f} crore</b> for FIIs and <b>\u20b9{cd:+,.0f} crore</b> for DIIs. "
               f"A positive number means more buying than selling in that window; negative means the opposite.")


def panel_raw():
    with st.expander("Raw data and detail"):
        d1, d2, d3 = st.columns(3)
        d1.write("**F&O positioning**")
        d1.write(f"long: {fnum(fo.get('fii_idx_fut_long'))}")
        d1.write(f"short: {fnum(fo.get('fii_idx_fut_short'))}")
        d1.write(f"net: {fnum(fo.get('fii_idx_fut_net'), 0, True)}")
        d1.write(f"long/short: {fnum(fo.get('fii_ls_ratio'), 2)}")
        d2.write("**Correlation**")
        d2.write(f"same-day: {fnum(same_day, 3, default='n/a')}")
        d2.write(f"best lag: t+{corr.get('best_lag')} (corr {fnum(corr.get('best_corr'), 3)})")
        d3.write("**DII absorption**")
        d3.write(f"lookback: {absorp.get('lookback')} sessions")
        d3.write(f"FII sell days: {absorp.get('fii_sell_days')}")
        d3.write(f"FII sold: \u20b9{fnum(absorp.get('total_fii_sold_cr'), 0)} cr")
        d3.write(f"DII bought: \u20b9{fnum(absorp.get('total_dii_bought_cr'), 0)} cr")
        st.write("**Merged, last 15 sessions**")
        table = merged_tail.tail(15) if merged_tail is not None else pd.DataFrame()
        try:
            st.dataframe(table, width="stretch")
        except TypeError:
            st.dataframe(table, use_container_width=True)
        st.write("**Latest FII/DII snapshot**")
        if fii_latest:
            st.json(fii_latest)
        else:
            st.write("No live snapshot.")



def _need(frame: pd.DataFrame, min_n: int) -> bool:
    if frame is None or frame.empty or len(frame) < min_n:
        have = 0 if frame is None else len(frame)
        st.markdown(H(f'<div class="card"><div class="card-title">Not enough data yet</div>'
                      f'<div class="card-sub" style="margin-top:6px">These statistics need at least {min_n} matched '
                      f'FII/DII/Nifty sessions and there are {have}. Try "All sessions" in the sidebar.</div></div>'),
                    unsafe_allow_html=True)
        return False
    return True


def _stat_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=4, r=4, t=6, b=4), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Manrope, sans-serif", color=MUTED, size=12),
        hoverlabel=dict(bgcolor="#13261C", bordercolor=LIME, font=dict(color=TEXT)),
        legend=dict(orientation="h", y=1.14, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.04)", zeroline=False)
    return fig


def _span(fr: pd.DataFrame) -> str:
    return f"{fr['date'].iloc[0].strftime('%d %b %Y')} to {fr['date'].iloc[-1].strftime('%d %b %Y')}"


def panel_scatter():
    fr = flows_win
    if not _need(fr, 15):
        return
    x, y = fr["fii_net"].to_numpy(float), fr["dii_net"].to_numpy(float)
    pr, ols = pearson_test(x, y), ols_test(x, y)
    chips = f'<span class="mini">n<b>{len(fr)}</b></span>'
    if pr:
        chips = f'<span class="mini">r<b>{pr["stat"]:+.2f}</b></span>' + chips
    if ols:
        chips = (f'<span class="mini">R\u00b2<b>{ols["r2"]:.2f}</b></span>'
                 f'<span class="mini">slope<b>{ols["slope"]:+.2f}</b></span>') + chips
    with card("scatter"):
        st.markdown(H(f'<div class="card-title">FII vs DII flows</div>'
                      f'<div class="card-sub">each dot is one session, brighter is more recent</div>'
                      f'<div class="chips">{chips}</div>'), unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_hline(y=0, line_color="rgba(255,255,255,.12)", line_width=1)
        fig.add_vline(x=0, line_color="rgba(255,255,255,.12)", line_width=1)
        if ols:
            xl = np.array([x.min(), x.max()])
            fig.add_trace(go.Scatter(x=xl, y=ols["intercept"] + ols["slope"] * xl, mode="lines",
                                     line=dict(color="rgba(232,240,233,.55)", width=1.6, dash="dash"),
                                     hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", showlegend=False,
            customdata=fr["date"].dt.strftime("%d %b %Y"),
            marker=dict(size=9, color=list(range(len(fr))), line=dict(width=0),
                        colorscale=[[0, "rgba(184,242,59,0.10)"], [1, "rgba(184,242,59,0.95)"]]),
            hovertemplate="<b>%{customdata}</b><br>FII \u20b9%{x:,.0f} cr<br>DII \u20b9%{y:,.0f} cr<extra></extra>"))
        for size, op, edge in ((26, 0.12, 0), (15, 0.25, 0), (8, 1.0, 2)):
            fig.add_trace(go.Scatter(x=[x[-1]], y=[y[-1]], mode="markers", showlegend=False, hoverinfo="skip",
                                     marker=dict(size=size, color=LIME, opacity=op, line=dict(color=BG, width=edge))))
        for tx, ty, xa, ya, txt in ((0.01, 0.99, "left", "top", "FII sells, DII buys"),
                                    (0.99, 0.01, "right", "bottom", "FII buys, DII sells")):
            fig.add_annotation(xref="paper", yref="paper", x=tx, y=ty, xanchor=xa, yanchor=ya,
                               text=txt, showarrow=False, font=dict(size=11, color=MUTED))
        fig.update_xaxes(title_text="FII net (\u20b9 cr)", title_font=dict(size=11))
        fig.update_yaxes(title_text="DII net (\u20b9 cr)", title_font=dict(size=11))
        fig.update_layout(hovermode="closest")
        plot(_stat_layout(fig, 330), "fig-scatter")
        st.caption(f"{_span(fr)} \u00b7 dashed line is the OLS fit")
        q1 = int(((x < 0) & (y > 0)).sum())
        q3 = int(((x > 0) & (y < 0)).sum())
        same_side = int((((x >= 0) & (y >= 0)) | ((x <= 0) & (y <= 0))).sum())
        opp_pct = (q1 + q3) / len(x) * 100 if len(x) else 0
        pattern = ("On most of these days, when FIIs sold, DIIs bought - and the other way round" if opp_pct > 55
                  else ("On most of these days, FIIs and DIIs moved the same way - both buying or both selling"
                        if same_side / len(x) * 100 > 55 else "There is no strong pattern between what FIIs and "
                        "DIIs did on the same day"))
        explain(f"Each dot is one trading day over {approx_span(len(fr))}. The dot's left-right position shows "
               f"how much FIIs bought or sold that day; its up-down position shows the same for DIIs. "
               f"{pattern}.")


def panel_rolling():
    fr = flows_all
    if not _need(fr, 25):
        return
    w = 20 if len(fr) >= 40 else 10
    rc = pd.DataFrame({
        "date": fr["date"],
        "FII vs DII": fr["fii_net"].rolling(w).corr(fr["dii_net"]),
        "FII vs Nifty return": fr["fii_net"].rolling(w).corr(fr["nifty_ret_pct"]),
        "DII vs Nifty return": fr["dii_net"].rolling(w).corr(fr["nifty_ret_pct"]),
    }).dropna()
    colors = {"FII vs DII": LIME, "FII vs Nifty return": TEXT, "DII vs Nifty return": AMBER}
    last = rc.iloc[-1]
    chips = "".join(f'<span class="mini">{E(k)}<b style="color:{colors[k]}">{last[k]:+.2f}</b></span>' for k in colors)
    with card("rolling"):
        st.markdown(H(f'<div class="card-title">Rolling correlation, {w} sessions</div>'
                      f'<div class="card-sub">does the link hold over time? a line that swings widely means an unstable relationship</div>'
                      f'<div class="chips">{chips}</div>'), unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,.25)", line_width=1)
        for k, c in colors.items():
            fig.add_trace(go.Scatter(x=rc["date"], y=rc[k], name=k, mode="lines",
                                     line=dict(color=c, width=2.2, shape="spline", smoothing=0.5),
                                     hovertemplate="%{y:+.2f}<extra>" + k + "</extra>"))
        fig.update_yaxes(range=[-1, 1], tickformat="+.1f")
        fig.update_layout(hovermode="x unified")
        plot(_stat_layout(fig, 330), "fig-rolling")
        st.caption(f"latest window ends {fr['date'].iloc[-1].strftime('%d %b %Y')}")
        fd, fr_, dr = last.get("FII vs DII"), last.get("FII vs Nifty return"), last.get("DII vs Nifty return")
        def _tie(v):
            if v is None or pd.isna(v):
                return "no clear pattern"
            if v > 0.3:
                return "moving in the same direction"
            if v < -0.3:
                return "moving in opposite directions"
            return "no strong pattern"
        explain(f"This checks, every {w} trading days, whether two things have been moving together or apart. "
               f"Right now: FII and DII flows show <b>{_tie(fd)}</b>; FII flow and Nifty's moves show "
               f"<b>{_tie(fr_)}</b>; DII flow and Nifty's moves show <b>{_tie(dr)}</b>. A line that keeps "
               f"swinging up and down means this relationship doesn't stay the same for long.")


def _sig(p: float) -> str:
    return '<span class="pill up">significant</span>' if p < 0.05 else '<span class="pill flat">not significant</span>'


def _trow(name, sub, stat, p_txt, verdict) -> str:
    return (f'<div class="trow"><div><div class="tname">{E(name)}</div><div class="tsub">{E(sub)}</div></div>'
            f'<div class="tstat">{stat}</div><div class="tp">{p_txt}</div><div class="tv">{verdict}</div></div>')


def _na_row(name, sub, why="needs more sessions") -> str:
    return _trow(name, sub, "n/a", "", f'<span class="card-sub">{E(why)}</span>')


def compute_tests(fr: pd.DataFrame, lag: int) -> dict:
    fii, dii, ret = (fr[c].to_numpy(float) for c in ("fii_net", "dii_net", "nifty_ret_pct"))
    return {
        "pear_fd": pearson_test(fii, dii), "spear_fd": spearman_test(fii, dii), "ols_fd": ols_test(fii, dii),
        "pear_fr": pearson_test(fii, ret), "pear_dr": pearson_test(dii, ret),
        "g_f_r": granger_test(ret, fii, lag), "g_r_f": granger_test(fii, ret, lag),
        "g_f_d": granger_test(dii, fii, lag), "g_d_f": granger_test(fii, dii, lag),
        "adf_f": adf_test(fii), "adf_d": adf_test(dii), "adf_r": adf_test(ret),
    }


def super_plain_summary(t: dict) -> str:
    """One beginner sentence per panel - no r=, no p-values, no 'Granger', no 'OLS'."""
    p = t.get("pear_fd")
    if p and p["p"] < 0.05:
        fd_txt = ("tend to move opposite ways - when one buys more, the other tends to sell more"
                  if p["stat"] < 0 else "tend to move the same way - when one buys more, so does the other")
    else:
        fd_txt = "don't show a clear pattern together on the same day"
    pf = t.get("pear_fr")
    if pf and pf["p"] < 0.05:
        fr_txt = (f"FII buying and selling has tended to line up with Nifty's moves on the same day "
                  f"({'both up or both down' if pf['stat'] > 0 else 'moving opposite ways'})")
    else:
        fr_txt = "FII buying and selling hasn't shown a strong same-day link with Nifty's moves"
    return f"FIIs and DIIs {fd_txt}. Separately, {fr_txt}."


def plain_read(t: dict, lag: int) -> list:
    out = []
    p, o = t["pear_fd"], t["ols_fd"]
    if p:
        if p["p"] < 0.05:
            way = "opposite directions" if p["stat"] < 0 else "the same direction"
            out.append(f"FII and DII flows tend to move in {way} (r = {p['stat']:+.2f}, {strength(p['stat'])}).")
            if o and o["p"] < 0.05 and o["slope"] < 0:
                out.append(f"Each extra \u20b91,000 cr of FII selling has coincided with about \u20b9{abs(o['slope']) * 1000:,.0f} cr "
                           f"of DII buying (R\u00b2 = {o['r2']:.2f}).")
        else:
            out.append(f"No statistically significant same-day link between FII and DII flows in this window (p = {fmt_p(p['p'])}).")
    pf = t["pear_fr"]
    if pf:
        if pf["p"] < 0.05:
            out.append(f"Same-day FII flow and Nifty return are {'positively' if pf['stat'] > 0 else 'negatively'} related "
                       f"(r = {pf['stat']:+.2f}). That shows they move together on the day, not which one drives the other.")
        else:
            out.append("No significant same-day link between FII flow and the Nifty return in this window.")
    leads = []
    if t["g_f_r"] and t["g_f_r"]["p"] < 0.05:
        leads.append("past FII flow helps predict Nifty returns")
    if t["g_r_f"] and t["g_r_f"]["p"] < 0.05:
        leads.append("past Nifty returns help predict FII flow (flows following price)")
    if t["g_f_d"] and t["g_f_d"]["p"] < 0.05:
        leads.append("past FII flow helps predict DII flow")
    if t["g_d_f"] and t["g_d_f"]["p"] < 0.05:
        leads.append("past DII flow helps predict FII flow")
    if any(t[k] for k in ("g_f_r", "g_r_f", "g_f_d", "g_d_f")):
        out.append(f"At lag {lag}: " + ("; ".join(leads) + "." if leads else "no lead-lag relationship is statistically significant."))
    return out


def panel_tests():
    fr = flows_win
    if not _need(fr, 20):
        return
    lag = stat_lag
    t = compute_tests(fr, lag)

    def corr_row(key, name, sub, sym):
        r = t[key]
        return (_trow(name, sub, f"{sym} = {r['stat']:+.2f}", fmt_p(r["p"]), _sig(r["p"])) if r else _na_row(name, sub))

    rel = [corr_row("pear_fd", "FII vs DII, Pearson", "linear link, same day", "r"),
           corr_row("spear_fd", "FII vs DII, Spearman", "rank based, robust to outliers", "\u03c1")]
    o = t["ols_fd"]
    rel.append(_trow("DII on FII, OLS", f"DII = a + b\u00b7FII, R\u00b2 {o['r2']:.2f}", f"b = {o['slope']:+.2f}", fmt_p(o["p"]), _sig(o["p"]))
               if o else _na_row("DII on FII, OLS", "DII = a + b\u00b7FII"))
    rel += [corr_row("pear_fr", "FII vs Nifty return", "same day, coincident not predictive", "r"),
            corr_row("pear_dr", "DII vs Nifty return", "same day, coincident not predictive", "r")]

    def g_row(key, name, sub):
        g = t[key]
        return (_trow(name, sub, f"F = {g['stat']:.2f}", fmt_p(g["p"]), _sig(g["p"])) if g
                else _na_row(name, sub, "needs 30+ sessions"))

    lead = [g_row("g_f_r", "FII \u2192 Nifty", f"do {lag} past FII flow(s) predict return"),
            g_row("g_r_f", "Nifty \u2192 FII", f"do {lag} past return(s) predict FII flow"),
            g_row("g_f_d", "FII \u2192 DII", "does FII flow lead DII flow"),
            g_row("g_d_f", "DII \u2192 FII", "does DII flow lead FII flow")]

    def adf_row(key, name):
        a = t[key]
        if not a:
            return _na_row(name, "unit-root test (ADF)")
        ok = a["level"] is not None
        verdict = (f'<span class="pill up">stationary {a["level"]}</span>' if ok
                   else '<span class="pill down">unit root</span>')
        return _trow(name, "ADF, constant, 1 lag", f"\u03c4 = {a['stat']:.2f}", f"5%: {ADF_CRIT['5%']:.2f}", verdict)

    stat_rows = [adf_row("adf_f", "FII net flow"), adf_row("adf_d", "DII net flow"), adf_row("adf_r", "Nifty daily return")]

    st.markdown(H(f"""
    <div class="card">
      <div class="card-head"><div class="card-title">Statistical tests</div>
        <span class="card-sub">n = {len(fr)} sessions, 5% level</span></div>
      <div class="card-sub">{_span(fr)}. Recomputed on every refresh and whenever new FII/DII figures arrive.</div>
      <div class="explain"><span class="explain-tag">in simple words</span>{super_plain_summary(t)}</div>
      <div class="tgroup">Relationship</div>{''.join(rel)}
      <div class="tgroup">Lead-lag, Granger at lag {lag}</div>{''.join(lead)}
      <div class="tgroup">Stationarity check</div>{''.join(stat_rows)}
      <div class="read-foot">Twelve tests are run together, so a p-value just under 0.05 is weak evidence. Granger tests show which series helps predict
        the other, not what causes what. ADF uses asymptotic critical values.</div>
    </div>"""), unsafe_allow_html=True)


def panel_leadlag():
    fr = flows_win
    if not _need(fr, 30):
        return
    fii, dii, ret = (fr[c].to_numpy(float) for c in ("fii_net", "dii_net", "nifty_ret_pct"))
    lags = [1, 2, 3, 4, 5]
    series = {"FII \u2192 Nifty": (ret, fii, LIME), "Nifty \u2192 FII": (fii, ret, TEXT),
              "FII \u2192 DII": (dii, fii, AMBER), "DII \u2192 FII": (fii, dii, RED)}
    with card("leadlag"):
        st.markdown(H('<div class="card-title">Who leads whom?</div>'
                      '<div class="card-sub">Granger p-value by lag. Points below the dashed line are significant at 5%.</div>'),
                    unsafe_allow_html=True)
        fig = go.Figure()
        for name, (y, x, c) in series.items():
            ps = []
            for L in lags:
                g = granger_test(y, x, L)
                ps.append(g["p"] if g else None)
            fig.add_trace(go.Scatter(x=lags, y=ps, name=name, mode="lines+markers", connectgaps=False,
                                     line=dict(color=c, width=2, shape="spline", smoothing=0.3), marker=dict(size=7),
                                     hovertemplate="p = %{y:.3f}<extra>" + name + "</extra>"))
        fig.add_hline(y=0.05, line_dash="dash", line_color=RED, line_width=1)
        fig.update_xaxes(tickvals=lags, title_text="lag (sessions)", title_font=dict(size=11))
        fig.update_yaxes(range=[0, 1.02], title_text="p-value", title_font=dict(size=11))
        fig.update_layout(hovermode="x unified")
        plot(_stat_layout(fig, 300), "fig-leadlag")
    t = compute_tests(fr, stat_lag)
    bullets = "".join(f'<div class="bullet"><span class="dot" style="background:{LIME}"></span><div>{E(s)}</div></div>'
                      for s in plain_read(t, stat_lag))
    leads_any = any(t[k] and t[k]["p"] < 0.05 for k in ("g_f_r", "g_r_f", "g_f_d", "g_d_f"))
    beginner = ("This chart checks whether knowing what FIIs or DIIs did on past days helps guess what happens "
               "a few days later - not just what happened on the same day. " +
               ("At the moment, at least one of those does show up as a useful early signal."
                if leads_any else
                "At the moment, none of those show up as a useful early signal here - past flows don't "
                "appear to predict what comes next, in this data."))
    st.markdown(H(f'<div class="card" style="margin-top:14px"><div class="card-title" style="margin-bottom:10px">In plain words</div>'
                  f'<div class="explain" style="margin-top:0"><span class="explain-tag">in simple words</span>{beginner}</div>'
                  f'{bullets}<div class="read-foot">Lag {stat_lag} chosen in the sidebar.</div></div>'), unsafe_allow_html=True)



def choice(options, key, default=None):
    default = default or options[0]
    if hasattr(st, "segmented_control"):
        v = st.segmented_control("Series", options, default=default, key=key, label_visibility="collapsed")
    else:
        v = st.radio("Series", options, index=options.index(default), horizontal=True, key=key,
                     label_visibility="collapsed")
    return v or default


def panel_calendar():
    fr = flows_all
    if not _need(fr, 10):
        return
    with card("calendar"):
        left, right = st.columns([2.2, 1])
        with right:
            which = choice(["FII", "DII"], "cal_series")
        col = "fii_net" if which == "FII" else "dii_net"
        g = calendar_grid(fr, col)
        vals = fr[col].dropna()
        buy, sell = int((vals > 0).sum()), int((vals < 0).sum())
        big_sell, big_buy = fr.loc[vals.idxmin()], fr.loc[vals.idxmax()]
        chips = (f'<span class="mini">buying days<b style="color:{LIME}">{buy}</b></span>'
                 f'<span class="mini">selling days<b style="color:{RED}">{sell}</b></span>'
                 f'<span class="mini">biggest sale<b>\u20b9{fnum(big_sell[col], 0)} cr</b>'
                 f'<span style="margin-left:6px">{big_sell["date"].strftime("%d %b")}</span></span>'
                 f'<span class="mini">biggest buy<b>\u20b9{fnum(big_buy[col], 0, True)} cr</b>'
                 f'<span style="margin-left:6px">{big_buy["date"].strftime("%d %b")}</span></span>')
        with left:
            st.markdown(H(f'<div class="card-title">{which} flow calendar</div>'
                          f'<div class="card-sub">each square is one session: green is net buying, red is net selling</div>'
                          f'<div class="chips">{chips}</div>'), unsafe_allow_html=True)
        z = g["z"]
        q = float(np.nanpercentile(np.abs(z), 95)) or 1.0
        fig = go.Figure(go.Heatmap(
            z=z, x=list(range(g["nweeks"])), y=["Mon", "Tue", "Wed", "Thu", "Fri"], customdata=g["txt"],
            zmin=-q, zmax=q, colorscale=[[0, RED], [0.5, "#16291F"], [1, LIME]], showscale=False,
            xgap=4, ygap=4, hoverongaps=False,
            hovertemplate="%{customdata}<br>\u20b9%{z:,.0f} cr<extra></extra>"))
        fig.update_xaxes(tickvals=g["tickvals"], ticktext=g["ticktext"], showgrid=False, zeroline=False)
        fig.update_yaxes(autorange="reversed", showgrid=False, zeroline=False)
        fig.update_layout(height=230, margin=dict(l=4, r=4, t=4, b=4), paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Manrope, sans-serif", color=MUTED, size=12),
                          hoverlabel=dict(bgcolor="#13261C", bordercolor=LIME, font=dict(color=TEXT)))
        plot(fig, "fig-calendar")
        st.caption(f"{_span(fr)}. Colours are capped at the 95th percentile so one extreme day does not wash out the rest.")
        explain(f"Each small square is one trading day. A green square means {which}s bought more than they sold "
               f"that day; a red square means they sold more than they bought. Over {approx_span(len(fr))}, "
               f"that happened on <b>{buy}</b> buying days and <b>{sell}</b> selling days. The biggest single "
               f"sale was <b>\u20b9{abs(big_sell[col]):,.0f} crore</b> on {big_sell['date'].strftime('%d %b')}, "
               f"and the biggest single buy was <b>\u20b9{fnum(big_buy[col], 0, True)} crore</b> on "
               f"{big_buy['date'].strftime('%d %b')}.")


def panel_heatmap():
    fr = flows_win
    if not _need(fr, 25):
        return
    mk = fetch_market_changes()
    if mk is None or mk.empty:
        st.markdown(H('<div class="card"><div class="card-title">Market data unavailable</div>'
                      '<div class="card-sub" style="margin-top:6px">Could not load the market series from Yahoo Finance just now. '
                      'It will retry automatically.</div></div>'), unsafe_allow_html=True)
        return
    df = fr[["date", "fii_net", "dii_net"]].rename(columns={"fii_net": "FII net", "dii_net": "DII net"}).merge(
        mk, on="date", how="left")
    cols = [c for c in ["FII net", "DII net", "Nifty", "Bank Nifty", "India VIX", "USD/INR", "US 10Y", "Brent", "Gold"]
            if c in df.columns]
    r, p = corr_matrix(df, cols)
    k = len(cols)
    fig = go.Figure(go.Heatmap(
        z=r, x=cols, y=cols, zmin=-1, zmax=1, showscale=False, xgap=3, ygap=3, hoverongaps=False,
        colorscale=[[0, RED], [0.5, "#122419"], [1, LIME]],
        customdata=p, hovertemplate="%{y} vs %{x}<br>r = %{z:.2f}<br>p = %{customdata:.3f}<extra></extra>"))
    for i in range(k):
        for j in range(k):
            if np.isnan(r[i, j]):
                continue
            strong = i != j and p[i, j] < 0.05
            fig.add_annotation(x=cols[j], y=cols[i], showarrow=False,
                               text=f"<b>{r[i, j]:.2f}</b>" if strong else f"{r[i, j]:.2f}",
                               font=dict(size=11, color=TEXT if (strong or i == j) else "rgba(232,240,233,.45)"))
    fig.update_xaxes(side="top", showgrid=False, zeroline=False, tickfont=dict(size=11))
    fig.update_yaxes(autorange="reversed", showgrid=False, zeroline=False, tickfont=dict(size=11))
    fig.update_layout(height=430, margin=dict(l=4, r=4, t=4, b=4), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Manrope, sans-serif", color=MUTED, size=12),
                      hoverlabel=dict(bgcolor="#13261C", bordercolor=LIME, font=dict(color=TEXT)))
    with card("heatmap"):
        st.markdown(H(f'<div class="card-title">What moves together?</div>'
                      f'<div class="card-sub">correlation of daily changes and daily flows, {len(fr)} sessions. '
                      f'Bold numbers are significant at 5%.</div>'), unsafe_allow_html=True)
        plot(fig, "fig-heatmap")
        st.caption("Yields are in basis points, other series in percent. US markets close after NSE, "
                   "so a one-day shift can fit US series better than the same-date link shown here.")
        best_pair, best_val = None, 0.0
        for a_i in range(len(cols)):
            for b_i in range(a_i + 1, len(cols)):
                if not np.isnan(r[a_i, b_i]) and abs(r[a_i, b_i]) > abs(best_val):
                    best_pair, best_val = (cols[a_i], cols[b_i]), r[a_i, b_i]
        pair_txt = (f"The strongest link here is between <b>{best_pair[0]}</b> and <b>{best_pair[1]}</b> "
                   f"({'moving together' if best_val > 0 else 'moving opposite ways'})." if best_pair else "")
        explain(f"Each square shows whether two things tend to move up and down together, over "
               f"{approx_span(len(fr))}. Bright green near the top means 'almost always move the same way'; "
               f"bright red means 'almost always move opposite ways'. A dull, dark square near the middle "
               f"means there isn't much of a pattern between them. {pair_txt}")


def panel_sip():
    monthly = load_sip_monthly()
    rows = ['<div class="prow head"><div>Phase</div><div>Nifty</div><div>Volatility</div>'
            '<div>Max drawdown</div><div>FII vs Nifty</div></div>']
    for name, s, e, color in SIP_PHASES:
        ps = phase_stats(hist, s, e)
        pc = phase_corr(monthly, s, e)
        label = f'{pd.Timestamp(s).strftime("%b %Y")} to {pd.Timestamp(e).strftime("%b %Y")}'
        if ps and ps["partial"]:
            label += " (partial data)"
        head = (f'<div class="pname"><span class="pdot" style="background:{color}"></span>'
                f'<div><div class="row-title">{E(name)}</div><div class="row-sub">{E(label)}</div></div></div>')
        if ps:
            cells = (f'<div style="color:{LIME if ps["ret"] >= 0 else RED}">{ps["ret"]:+.1f}%</div>'
                     f'<div>{ps["vol"]:.1f}%</div><div style="color:{RED}">{ps["dd"]:.1f}%</div>')
        else:
            cells = '<div>\u2014</div><div>\u2014</div><div>\u2014</div>'
        if pc:
            note = "" if pc["n"] >= 10 else " (few points)"
            corr_cell = f'<div>r = {pc["r"]:+.2f}<div class="row-sub">n = {pc["n"]}{note}</div></div>'
        else:
            corr_cell = '<div>\u2014</div>'
        rows.append(f'<div class="prow">{head}{cells}{corr_cell}</div>')
    hint = ("" if monthly is not None else
            "To fill the last column, save your monthly SIP data as data/sip_monthly.csv with the columns "
            "date, fii_net_cr, nifty_return_pct. ")
    st.markdown(H(f"""
    <div class="card" style="min-height:430px">
      <div class="card-head"><div class="card-title">My SIP phases</div><span class="card-sub">Nifty inside each phase</span></div>
      {''.join(rows)}
      <div class="read-foot">{E(hint)}Phase dates follow your SIP research. Change SIP_PHASES near the top of app.py if a boundary
        differs. Nifty figures use daily closes from Yahoo Finance: return is first to last close in the phase, volatility is annualised,
        drawdown is the worst fall from a peak.</div>
    </div>"""), unsafe_allow_html=True)
    phase_rets = [(name, phase_stats(hist, s, e)) for name, s, e, _ in SIP_PHASES]
    phase_rets = [(n, p["ret"]) for n, p in phase_rets if p]
    if phase_rets:
        best_ph = max(phase_rets, key=lambda t: t[1])
        worst_ph = min(phase_rets, key=lambda t: t[1])
        explain(f"This table splits your SIP research timeframe into 4 chunks. Nifty did best during "
               f"<b>{best_ph[0]}</b> (<b>{best_ph[1]:+.1f}%</b>) and worst during <b>{worst_ph[0]}</b> "
               f"(<b>{worst_ph[1]:+.1f}%</b>). The last column shows whether FII buying/selling and Nifty's "
               f"moves lined up during each chunk - a higher number means they moved together more closely.")



def panel_whatif():
    fr = flows_win
    if not _need(fr, 15):
        return
    o = ols_test(fr["fii_net"].to_numpy(float), fr["nifty_ret_pct"].to_numpy(float))
    if not o:
        return
    resid_std = float(np.std(fr["nifty_ret_pct"] - (o["intercept"] + o["slope"] * fr["fii_net"])))
    lo_x, hi_x = float(fr["fii_net"].min()), float(fr["fii_net"].max())
    pad = (hi_x - lo_x) * 0.4 or 1000.0
    with card("whatif"):
        st.markdown(H('<div class="card-title">What if FII flow were...</div>'
                      '<div class="card-sub">a rough historical-average estimate, not a forecast or advice</div>'),
                    unsafe_allow_html=True)
        slo, shi = float(lo_x - pad), float(hi_x + pad)
        st.session_state.setdefault("whatif_x", 0.0)
        st.session_state["whatif_x"] = min(max(st.session_state["whatif_x"], slo), shi)
        x = st.slider("Hypothetical FII net flow (\u20b9 cr)", slo, shi,
                     step=100.0, key="whatif_x", label_visibility="collapsed")
        pred = o["intercept"] + o["slope"] * x
        band = resid_std
        color = LIME if pred >= 0 else RED
        pts_val = (price or 0) * pred / 100 if price else None
        pts_txt = f' &middot; about {pts_val:+,.0f} Nifty points from here' if pts_val is not None else ""
        st.markdown(H(f'''<div class="whatif-out">
          <div class="whatif-big" style="color:{color}">{pred:+.2f}%</div>
          <div class="whatif-sub">typical range {pred - band:+.2f}% to {pred + band:+.2f}%{pts_txt}</div>
        </div>'''), unsafe_allow_html=True)
        st.caption(f"Based on the OLS fit over {len(fr)} sessions (R\u00b2 = {o['r2']:.2f}). "
                  f"This reads the historical average relationship; it is not a prediction of what will actually happen.")
        explain(f"Drag the slider to pick a hypothetical amount of FII buying or selling. The number above shows "
               f"what Nifty has, <b>on average</b>, done on days when FII flow was around that amount, based on "
               f"{approx_span(len(fr))} of real data. It is a historical pattern, not a prediction - markets "
               f"don't repeat exactly.")


def panel_daily_vs_monthly():
    monthly = load_sip_monthly()
    fr = flows_all
    daily = pearson_test(fr["fii_net"], fr["nifty_ret_pct"]) if not fr.empty else None
    mrow = None
    if monthly is not None:
        m = monthly.dropna(subset=["fii_net_cr", "nifty_return_pct"])
        if len(m) >= 4:
            mrow = pearson_test(m["fii_net_cr"], m["nifty_return_pct"])
    with card("dvm"):
        st.markdown(H('<div class="card-title">Daily vs monthly correlation</div>'
                      '<div class="card-sub">same relationship, FII flow vs Nifty return, at two samplings</div>'),
                    unsafe_allow_html=True)
        d_txt = f"r = {daily['stat']:+.2f}" if daily else "n/a"
        d_sub = f"n = {daily['n']} sessions" if daily else "not enough data"
        m_txt = f"r = {mrow['stat']:+.2f}" if mrow else "n/a"
        m_sub = f"n = {mrow['n']} months" if mrow else ("add data/sip_monthly.csv" if monthly is None else "not enough months")
        st.markdown(H(f'''<div class="pair">
          <div><div class="pv" style="color:{TEXT}">{d_txt}</div><div class="pl">Daily, {E(d_sub)}</div></div>
          <div><div class="pv" style="color:{LIME}">{m_txt}</div><div class="pl">Monthly, {E(m_sub)}</div></div>
        </div>'''), unsafe_allow_html=True)
        st.markdown(H('<div class="read-foot" style="margin-top:14px">Daily noise (a big single-day headline, an options-expiry '
                      'wobble) tends to cancel out once flows are summed into a month, which is usually why the monthly '
                      'correlation reads stronger than the daily one. Neither figure is more "correct" than the other; '
                      'they answer different questions.</div>'), unsafe_allow_html=True)
        def _lvl(r):
            if r is None:
                return "can't be checked yet"
            a = abs(r["stat"])
            word = "barely linked" if a < 0.2 else ("a bit linked" if a < 0.4 else ("linked" if a < 0.6 else "closely linked"))
            return word
        explain(f"If you look day by day, FII buying/selling and Nifty's moves look <b>{_lvl(daily)}</b>. But "
               f"if you add everything up a whole month at a time, the same two things look "
               f"<b>{_lvl(mrow)}</b>. This usually happens because single noisy days cancel each other out "
               f"once you zoom out to a month.")


METHOD_ROWS = [
    ("Nifty 50 price", "Yahoo Finance (delayed) or a connected broker (Upstox/Zerodha/Groww)", "live tick / daily close",
     "Free public data is delayed ~15 min and not licensed for redistribution."),
    ("FII / DII net, gross flows", "NSE cash-market activity report, via a third-party aggregator", "once daily, after market close",
     "Provisional figures; can be revised. Not a licensed feed."),
    ("FII index-futures long/short", "NSE participant-wise open interest", "once daily, after market close",
     "F&O positioning only; does not include options."),
    ("Bank Nifty, India VIX, USD/INR, US 10Y, Brent, Gold", "Yahoo Finance", "delayed, ~daily",
     "For context only; not used in the statistical tests."),
    ("Monthly SIP correlation", "Your own research file, data/sip_monthly.csv", "as you update it",
     "Optional. Falls back to a note if the file is absent."),
]


def panel_methodology():
    with card("method"):
        st.markdown(H('<div class="card-title">Methodology</div>'
                      '<div class="card-sub">what each number is, where it comes from, and its limits</div>'),
                    unsafe_allow_html=True)
        st.markdown("\n".join(
            ["| Variable | Source | Frequency | Notes |", "|---|---|---|---|"] +
            [f"| {n} | {s} | {f} | {nt} |" for n, s, f, nt in METHOD_ROWS]
        ))
        st.markdown(H('<div class="read-foot" style="margin-top:6px">'
                      'Statistical tests (correlation, Granger, ADF) are computed on whatever FII/DII history the app currently '
                      'holds, typically 90-120 sessions - thin for Granger causality by academic convention. For the SIP report '
                      'itself, treat the dashboard as a live demonstration of the method, and the original monthly dataset as '
                      'the record of the finding.</div>'), unsafe_allow_html=True)


def panel_downloads():
    with card("downloads"):
        st.markdown(H('<div class="card-title">Downloads</div>'
                      '<div class="card-sub">export what is on screen right now</div>'), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if merged_tail is not None and not merged_tail.empty:
            csv_bytes = merged_tail.to_csv(index=False).encode("utf-8")
            csv_name = f"fii_nifty_data_{now_ist.strftime('%Y%m%d')}.csv"
            try:
                c1.download_button("Flow data (CSV)", csv_bytes, file_name=csv_name, mime="text/csv", width="stretch")
            except TypeError:
                c1.download_button("Flow data (CSV)", csv_bytes, file_name=csv_name, mime="text/csv", use_container_width=True)
        else:
            c1.caption("No flow data to export yet.")
        try:
            from fpdf import FPDF
            pdf_bytes = _build_pdf_summary()
            c2.download_button("Today's read (PDF)", pdf_bytes,
                               file_name=f"fii_nifty_summary_{now_ist.strftime('%Y%m%d')}.pdf",
                               mime="application/pdf")
        except ImportError:
            c2.caption("Run: pip install fpdf2  for the PDF summary button.")
        except Exception as e:
            c2.caption(f"Could not build the PDF ({e}).")


def _pdf_safe(text: str) -> str:
    """fpdf2's core Helvetica font only supports Latin-1, unlike the browser. Swap the symbols this
    dashboard actually uses, then fall back to '?' for anything else so a PDF can never crash on a
    stray character."""
    text = (text.replace("\u20b9", "Rs ").replace("\u00b2", "^2").replace("\u00d7", "x")
                .replace("\u2192", "->").replace("\u2190", "<-")
                .replace("\u2191", "up").replace("\u2193", "down")
                .replace("\u2013", "-").replace("\u2014", "-")
                .replace("\u2018", "'").replace("\u2019", "'")
                .replace("\u201c", '"').replace("\u201d", '"'))
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _build_pdf_summary() -> bytes:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    NL = dict(new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "FII x Nifty 50 - Daily Summary", **NL)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated {now_ist.strftime('%d %b %Y, %H:%M IST')} - a project by Akhil Madhu", **NL)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, f"Nifty 50: {fnum(price, 2)}  ({day_pct:+.2f}%)" if day_pct is not None else f"Nifty 50: {fnum(price, 2)}", **NL)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"FII net: Rs {fnum(fii_net, 0, True)} cr   |   DII net: Rs {fnum(dii_net, 0, True)} cr", **NL)
    pdf.cell(0, 7, f"Regime: {REGIME_LABEL.get(reg, reg)}  (confidence {reg_conf * 100:.0f}%)", **NL)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Market read", **NL)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, _pdf_safe(headline), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    for s in support:
        pdf.multi_cell(0, 6, _pdf_safe("- " + s), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)
    if merged_tail is not None and not merged_tail.empty:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Last 10 sessions", **NL)
        pdf.set_font("Helvetica", "B", 9)
        for w, t in ((28, "Date"), (34, "FII net (cr)"), (34, "DII net (cr)"), (34, "Nifty close")):
            pdf.cell(w, 6, t, border=1)
        pdf.ln()
        pdf.set_font("Helvetica", "", 9)
        for _, row in merged_tail.tail(10).iterrows():
            pdf.cell(28, 6, str(row["date"])[:10], border=1)
            pdf.cell(34, 6, f"{row.get('fii_net', 0):,.0f}", border=1)
            pdf.cell(34, 6, f"{row.get('dii_net', 0):,.0f}", border=1)
            pdf.cell(34, 6, f"{row.get('nifty_close', 0):,.2f}", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 5, _pdf_safe("Descriptive read of published flow data, not investment advice. FII/DII figures "
                                   "are end-of-day and provisional. See the Methodology panel in the dashboard for "
                                   "sources and limits."), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    return bytes(pdf.output())



def panel_backtest():
    summ = summarize_backtest(regime_bt)
    if summ is None or summ.empty:
        st.markdown(H('<div class="card"><div class="card-title">Regime track record</div>'
                      '<div class="card-sub" style="margin-top:6px">Needs a longer price/flow history than is '
                      'currently loaded (at least ~25 sessions after warmup) to test whether a regime label has '
                      'actually predicted what came next.</div></div>'), unsafe_allow_html=True)
        return
    horizons = (1, 3, 5)
    rows = ['<div class="prow head"><div>Regime</div><div>n</div><div>Avg 1d</div><div>Hit 1d</div><div>Avg 5d</div></div>']
    for _, r in summ.sort_values("n", ascending=False).iterrows():
        reg_name = r["regime"]
        color = {"FII_SELL_PRESSURE": RED, "MILD_FII_OUTFLOW": RED, "DII_SUPPORTED": AMBER,
                "FII_LED_RALLY": LIME, "MILD_FII_INFLOW": LIME, "NEUTRAL": MUTED}.get(reg_name, MUTED)
        head = (f'<div class="pname"><span class="pdot" style="background:{color}"></span>'
               f'<div><div class="row-title">{E(REGIME_LABEL.get(reg_name, reg_name))}</div></div></div>')
        avg1, hit1, avg5 = r.get("avg_1d"), r.get("hit_1d"), r.get("avg_5d")
        c_avg1 = f'<div style="color:{LIME if (avg1 or 0) >= 0 else RED}">{avg1:+.2f}%</div>' if avg1 is not None else '<div>\u2014</div>'
        c_hit1 = f'<div>{hit1:.0f}%</div>' if hit1 is not None else '<div>\u2014</div>'
        c_avg5 = f'<div style="color:{LIME if (avg5 or 0) >= 0 else RED}">{avg5:+.2f}%</div>' if avg5 is not None else '<div>\u2014</div>'
        rows.append(f'<div class="prow"><div class="pname">{head}</div><div>{int(r["n"])}</div>{c_avg1}{c_hit1}{c_avg5}</div>')
    with card("backtest"):
        st.markdown(H(f'<div class="card-title">Regime track record</div>'
                      f'<div class="card-sub">what Nifty actually did after each regime label showed up, '
                      f'{regime_bt["date"].min().strftime("%d %b %Y")} to {regime_bt["date"].max().strftime("%d %b %Y")}, '
                      f'day by day, using only data available up to that day</div>'
                      f'{"".join(rows)}'), unsafe_allow_html=True)
        fig = go.Figure()
        plot_df = summ.dropna(subset=["avg_5d"])
        if not plot_df.empty:
            fig.add_trace(go.Bar(x=[REGIME_LABEL.get(r, r) for r in plot_df["regime"]], y=plot_df["avg_5d"],
                                 marker_color=[LIME if v >= 0 else RED for v in plot_df["avg_5d"]],
                                 text=[f"n={n}" for n in plot_df["n"]], textposition="outside"))
            fig.add_hline(y=0, line_color="rgba(255,255,255,.2)", line_width=1)
            fig.update_yaxes(title_text="avg next-5-session return (%)", title_font=dict(size=11))
            plot(_stat_layout(fig, 280), "fig-backtest")
            top_row = plot_df.sort_values("n", ascending=False).iloc[0]
            top_label = REGIME_LABEL.get(top_row["regime"], top_row["regime"])
            top_avg = top_row["avg_5d"]
            word = "rose" if top_avg >= 0 else "fell"
            explain(f"This looks back through history and checks: after a day was labelled "
                   f"<b>{top_label.lower()}</b> (the situation that has happened most often), what did Nifty "
                   f"usually do over the <b>next 5 trading days</b>? On average, it {word} by "
                   f"<b>{abs(top_avg):.1f}%</b>. This tells you what tended to happen before - it is not a "
                   f"promise of what will happen next time.")
        st.markdown(H('<div class="read-foot">Overlapping windows, not independent draws - a single strong week can '
                      'dominate one regime\'s average. Sample sizes here are small (n in the tens, not thousands), so '
                      'read this as "does the label look directionally sensible", not as a validated trading signal. '
                      'No transaction costs, slippage, or position sizing are modelled.</div>'), unsafe_allow_html=True)


REGIME_SHORT = {
    "FII_SELL_PRESSURE": "FIIs continue to sell; caution favoured over fresh buying.",
    "DII_SUPPORTED": "FIIs selling, but DIIs are cushioning the market.",
    "FII_LED_RALLY": "FIIs buying strongly; sentiment stays constructive.",
    "MILD_FII_OUTFLOW": "Mild FII selling; not yet a clear trend.",
    "MILD_FII_INFLOW": "Mild FII buying; a modest positive sign.",
    "NEUTRAL": "No clear FII/DII trend today.",
}


def panel_client_update():
    with card("client_update"):
        st.markdown(H('<div class="card-title">Client update</div>'
                      '<div class="card-sub">a short, copy-ready message - tap the icon in the corner to copy</div>'),
                    unsafe_allow_html=True)
        day_line = f"Nifty 50: {fnum(price, 2)}" + (f" ({day_pct:+.2f}%)" if day_pct is not None else "")
        msg = (
            f"Market Update - {now_ist.strftime('%d %b, %H:%M')} IST\n"
            f"{day_line}\n"
            f"FII: \u20b9{fnum(fii_net, 0, True)} Cr | DII: \u20b9{fnum(dii_net, 0, True)} Cr\n"
            f"{REGIME_SHORT.get(reg, 'Mixed signals today.')}\n"
            f"(auto-generated from published NSE data - not investment advice)"
        )
        st.code(msg, language=None)
        st.caption("Numbers are the same ones shown above. Edit before sending - this is a draft, not a "
                  "compliance-reviewed client communication.")



def panel_sector_flows():
    with card("sector"):
        st.markdown(H('<div class="card-title">Where is the money going?</div>'
                      '<div class="card-sub">sector performance as a proxy, plus FPI\'s own official sector data '
                      'where it exists</div>'), unsafe_allow_html=True)

        st.markdown('<div class="tgroup" style="margin-top:2px">Sector performance (price-based proxy)</div>',
                   unsafe_allow_html=True)
        st.caption("Nifty sectoral indices, relative to Nifty 50 itself, over the window below. This reflects "
                  "which sectors are outperforming or lagging - not disclosed FII or DII rupee flows, which "
                  "no free source publishes by sector in real time. Read it as \'where relative strength currently "
                  "sits\', not as a report of who bought what.")
        win_label = choice(["1W", "1M", "3M"], "sector_win", default="1M")
        win_days = {"1W": 7, "1M": 30, "3M": 90}[win_label]
        sectors = fetch_sector_series()
        nifty_series = None
        if not hist.empty:
            nifty_series = hist.set_index("date")["close"]
        rows = []
        for s in sectors:
            ser = s["series"]
            cutoff = ser.index.max() - pd.Timedelta(days=win_days)
            recent = ser[ser.index >= cutoff]
            if len(recent) < 3:
                continue
            sec_ret = (recent.iloc[-1] / recent.iloc[0] - 1) * 100
            rel = sec_ret
            if nifty_series is not None:
                nrecent = nifty_series[nifty_series.index >= cutoff]
                if len(nrecent) >= 2:
                    rel = sec_ret - (nrecent.iloc[-1] / nrecent.iloc[0] - 1) * 100
            rows.append((s["name"], sec_ret, rel))
        if rows:
            rows.sort(key=lambda r: r[2], reverse=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[r[2] for r in rows], y=[r[0] for r in rows], orientation="h",
                marker_color=[LIME if r[2] >= 0 else RED for r in rows],
                customdata=[r[1] for r in rows],
                hovertemplate="%{y}: %{x:+.1f}% vs Nifty (sector itself %{customdata:+.1f}%)<extra></extra>"))
            fig.add_vline(x=0, line_color="rgba(255,255,255,.2)", line_width=1)
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(title_text=f"return vs Nifty 50 over {win_label} (%)", title_font=dict(size=11))
            plot(_stat_layout(fig, 320), "fig-sector")
            best, worst = rows[0], rows[-1]
            explain(f"Over {win_label} (compared to Nifty itself), the <b>{best[0]}</b> sector did the best, "
                   f"{'gaining' if best[2] >= 0 else 'losing'} about <b>{abs(best[2]):.1f}%</b> more than the "
                   f"overall market. The <b>{worst[0]}</b> sector did the worst, "
                   f"{'gaining' if worst[2] >= 0 else 'losing'} about <b>{abs(worst[2]):.1f}%</b> "
                   f"{'more' if worst[2] >= 0 else 'less'} than the market. This is based on prices only - it "
                   f"shows which parts of the market did well, not exactly who bought or sold.")
        else:
            st.caption("Sector index data is not available right now (Yahoo Finance may be unreachable, or one of "
                      "the sector tickers may have changed - this is the one part of the dashboard I could not "
                      "verify against a live feed while building it).")

        st.markdown('<div class="tgroup">FPI sector-wise investment (official, from NSDL)</div>', unsafe_allow_html=True)
        fpi_sector = load_fpi_sector_csv()
        if fpi_sector is not None and not fpi_sector.empty:
            d = fpi_sector.copy()
            if "period" in d.columns:
                latest_period = d["period"].iloc[-1]
                d = d[d["period"] == latest_period]
                st.caption(f"Period: {latest_period}")
            d = d.sort_values("net_invest_cr")
            fig2 = go.Figure(go.Bar(x=d["net_invest_cr"], y=d["sector"], orientation="h",
                                    marker_color=[LIME if v >= 0 else RED for v in d["net_invest_cr"]],
                                    hovertemplate="%{y}: \u20b9%{x:,.0f} cr<extra></extra>"))
            fig2.add_vline(x=0, line_color="rgba(255,255,255,.2)", line_width=1)
            fig2.update_xaxes(title_text="net FPI investment (\u20b9 cr)", title_font=dict(size=11))
            plot(_stat_layout(fig2, max(220, 26 * len(d))), "fig-fpi-sector")
            top_fpi, bot_fpi = d.iloc[-1], d.iloc[0]
            explain(f"In the latest period NSDL reported, foreign investors put the most new money into the "
                   f"<b>{top_fpi['sector']}</b> sector (about <b>\u20b9{top_fpi['net_invest_cr']:,.0f} crore</b>) "
                   f"and pulled the most money out of <b>{bot_fpi['sector']}</b> (about "
                   f"<b>\u20b9{abs(bot_fpi['net_invest_cr']):,.0f} crore</b>). So even while headline FII "
                   f"numbers might show overall selling, this shows some sectors can still be getting fresh "
                   f"money.")
        else:
            st.markdown(H(
                '<div class="card-sub">NSDL publishes this free, officially, twice a month - but only as a '
                'downloadable report, not a live feed, so it is not wired in automatically. To add it: open '
                '<a href="https://www.fpi.nsdl.co.in/web/Reports/FPI_Fortnightly_Selection.aspx" target="_blank" '
                'style="color:inherit">NSDL\'s Fortnightly Sector-wise FPI Investment report</a>, save the sector '
                'and net-investment columns as <code>data/fpi_sector.csv</code> (columns: sector, net_invest_cr, '
                'and optionally period), and this chart fills in automatically.</div>'), unsafe_allow_html=True)

        st.markdown('<div class="tgroup">DII sector-wise investment</div>', unsafe_allow_html=True)
        st.markdown(H(
            '<div class="card-sub">No free official source publishes DII investment broken down by equity sector '
            'in the way NSDL does for FPI. AMFI\'s monthly bulletin (free, official) reports mutual-fund '
            '<i>category</i> flows instead - equity, debt, hybrid, and fund types like sectoral/thematic - which is '
            'a different cut of the data, not a sector-of-the-market breakdown. The sector-performance proxy above '
            'is the closest free, real-time-ish signal available for DII activity too.</div>'),
            unsafe_allow_html=True)



REPORT_PERIODS = [
    ("Today", 1), ("This week", 7), ("This month", 31), ("Last 6 months", 183),
    ("Last 1 year", 366), ("Last 5 years", 1830), ("All available data", None),
]


def _period_slice(df: pd.DataFrame, date_col: str, days):
    if df is None or df.empty or days is None:
        return df
    cutoff = df[date_col].max() - pd.Timedelta(days=days)
    return df[df[date_col] >= cutoff]


def build_full_report_pdf(period_label: str, days) -> bytes:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos
    NL = dict(new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def h1(text):
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, _pdf_safe(text), **NL)

    def h2(text):
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, _pdf_safe(text), **NL)

    def body(text):
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5.5, _pdf_safe(text), **NL)

    def small(text):
        pdf.set_font("Helvetica", "I", 8)
        pdf.multi_cell(0, 4.5, _pdf_safe(text), **NL)

    h1("FII x Nifty 50 - Report")
    small(f"Period: {period_label} | Generated {now_ist.strftime('%d %b %Y, %H:%M IST')} | "
         f"a project by Akhil Madhu")
    pdf.ln(2)

    h2("1. Nifty 50 - what happened to the market")
    nifty_slice = _period_slice(hist, "date", days) if (hist is not None and not hist.empty) else pd.DataFrame()
    if len(nifty_slice) >= 2:
        c0, c1_ = float(nifty_slice["close"].iloc[0]), float(nifty_slice["close"].iloc[-1])
        hi, lo = float(nifty_slice["close"].max()), float(nifty_slice["close"].min())
        chg = (c1_ / c0 - 1) * 100 if c0 else 0.0
        move = "went up" if chg > 0 else ("went down" if chg < 0 else "stayed flat")
        body(f"Over {period_label.lower()}, Nifty 50 {move} by {abs(chg):.1f}%. It started around "
             f"{c0:,.0f} and ended around {c1_:,.0f}. The highest point was about {hi:,.0f}, and the "
             f"lowest was about {lo:,.0f}.")
    else:
        body("Not enough price history is loaded to cover this period.")

    h2("2. FII and DII activity - who bought, who sold")
    flow_slice = _period_slice(flows_all, "date", days) if (flows_all is not None and not flows_all.empty) else pd.DataFrame()
    if len(flow_slice) >= 2:
        fii_buy = int((flow_slice["fii_net"] > 0).sum())
        fii_sell = int((flow_slice["fii_net"] < 0).sum())
        dii_buy = int((flow_slice["dii_net"] > 0).sum())
        dii_sell = int((flow_slice["dii_net"] < 0).sum())
        fii_tot = float(flow_slice["fii_net"].sum())
        dii_tot = float(flow_slice["dii_net"].sum())
        body(f"Foreign investors (FIIs) were net buyers on {fii_buy} days and net sellers on {fii_sell} days "
             f"in this period. Added up, they {'bought' if fii_tot >= 0 else 'sold'} about Rs "
             f"{abs(fii_tot):,.0f} crore worth of shares overall.")
        body(f"Domestic investors (DIIs) were net buyers on {dii_buy} days and net sellers on {dii_sell} days. "
             f"Added up, they {'bought' if dii_tot >= 0 else 'sold'} about Rs {abs(dii_tot):,.0f} crore "
             f"worth of shares overall.")
        pr = pearson_test(flow_slice["fii_net"], flow_slice["dii_net"])
        if pr and pr["p"] < 0.05:
            way = ("opposite ways - when one bought more, the other tended to sell more" if pr["stat"] < 0
                  else "the same way - both tended to buy or both tended to sell on the same days")
            body(f"During this period, FIIs and DIIs tended to move {way}.")
        else:
            body("During this period, there was no clear, reliable pattern between what FIIs and DIIs did "
                 "on the same day.")
    else:
        body("Not enough FII/DII history is loaded to cover this period.")

    h2("3. Where the money went, by sector")
    sectors = fetch_sector_series()
    if sectors and hist is not None and not hist.empty:
        win_days_for_report = 7 if (days or 999) <= 7 else (31 if (days or 999) <= 45 else 90)
        nifty_series = hist.set_index("date")["close"]
        srows = []
        for s in sectors:
            ser = s["series"]
            cutoff = ser.index.max() - pd.Timedelta(days=win_days_for_report)
            recent = ser[ser.index >= cutoff]
            if len(recent) < 3:
                continue
            sec_ret = (recent.iloc[-1] / recent.iloc[0] - 1) * 100
            nrecent = nifty_series[nifty_series.index >= cutoff]
            rel = sec_ret - ((nrecent.iloc[-1] / nrecent.iloc[0] - 1) * 100 if len(nrecent) >= 2 else 0.0)
            srows.append((s["name"], rel))
        if srows:
            srows.sort(key=lambda t: t[1], reverse=True)
            best, worst = srows[0], srows[-1]
            body(f"Looking at which parts of the market did better or worse than Nifty overall (a rough "
                 f"stand-in for where money may be flowing, since no free source publishes exact sector-level "
                 f"FII/DII rupee flows in real time): the {best[0]} sector did the best, and the {worst[0]} "
                 f"sector did the worst, over a comparable recent window.")
        else:
            body("Sector-level price data was not available while this report was generated.")
    else:
        body("Sector-level price data was not available while this report was generated.")

    fpi_sector = load_fpi_sector_csv()
    if fpi_sector is not None and not fpi_sector.empty:
        d = fpi_sector.copy()
        if "period" in d.columns:
            d = d[d["period"] == d["period"].iloc[-1]]
        d = d.sort_values("net_invest_cr")
        if not d.empty:
            top_fpi, bot_fpi = d.iloc[-1], d.iloc[0]
            body(f"NSDL's official sector-wise report (the only free, official source of this kind) most "
                 f"recently showed foreign investors putting the most new money into {top_fpi['sector']} "
                 f"and taking the most money out of {bot_fpi['sector']}.")
    else:
        body("No official NSDL sector-wise FPI file has been added to this project yet - this is optional; "
             "see the Sector flows panel in the dashboard for how to add it.")
    body("There is no free, official source that reports DII investment broken down by stock-market sector. "
         "AMFI's monthly mutual-fund data is the closest free source, but it reports fund category - equity, "
         "debt, sectoral funds - not sector of the market.")

    h2("4. What has usually followed this kind of pattern")
    summ = summarize_backtest(regime_bt)
    if summ is not None and not summ.empty:
        top_row = summ.sort_values("n", ascending=False).iloc[0]
        avg5 = top_row.get("avg_5d")
        if avg5 is not None:
            label_txt = REGIME_LABEL.get(top_row["regime"], top_row["regime"]).lower()
            body(f"Looking back through history, after days that looked like '{label_txt}' (the most common "
                 f"pattern seen), Nifty has on average {'risen' if avg5 >= 0 else 'fallen'} by "
                 f"{abs(avg5):.1f}% over the following 5 trading days. This is a historical pattern, not a "
                 f"guarantee of what happens next.")
    else:
        body("Not enough history is loaded yet to look at what has usually followed this kind of pattern.")

    pdf.ln(3)
    small("This report is generated automatically from published NSE/NSDL data and simple statistics. It is a "
         "descriptive summary for learning purposes, not investment advice, and not a substitute for "
         "professional research.")
    return bytes(pdf.output())


def panel_report():
    with card("report"):
        st.markdown(H('<div class="card-title">Generate a report</div>'
                      '<div class="card-sub">pick a time period - every section is explained in plain words</div>'),
                    unsafe_allow_html=True)
        labels = [p[0] for p in REPORT_PERIODS]
        period_label = st.selectbox("Report period", labels, index=2, key="report_period",
                                    label_visibility="collapsed")
        days = dict(REPORT_PERIODS)[period_label]
        if st.button("Generate report", key="gen_report_btn"):
            try:
                from fpdf import FPDF
                pdf_bytes = build_full_report_pdf(period_label, days)
                st.session_state["report_pdf"] = pdf_bytes
                st.session_state["report_pdf_label"] = period_label
            except ImportError:
                st.session_state["report_pdf"] = None
                st.caption("Run: pip install fpdf2 to generate reports.")
            except Exception as e:
                st.session_state["report_pdf"] = None
                st.caption(f"Could not build the report ({e}).")
        if st.session_state.get("report_pdf"):
            safe_label = st.session_state.get("report_pdf_label", "report").replace(" ", "_").lower()
            fname = f"fii_nifty_report_{safe_label}_{now_ist.strftime('%Y%m%d')}.pdf"
            st.download_button("Download report (PDF)", st.session_state["report_pdf"],
                               file_name=fname, mime="application/pdf")
            st.caption(f"Report ready for: {st.session_state.get('report_pdf_label')}.")


RENDER = {
    "Nifty chart": panel_nifty_chart, "Market read": panel_market_read,
    "DII absorption": panel_absorption, "FII sell days": panel_sell_days,
    "FII streak": panel_streak, "F&O positioning": panel_fo,
    "Signals": panel_signals, "Markets": panel_markets, "Market hours": panel_hours,
    "FII flow chart": panel_flow_chart, "Cumulative chart": panel_cum_chart,
    "Raw data": panel_raw,
    "FII vs DII scatter": panel_scatter, "Rolling correlation": panel_rolling,
    "Statistical tests": panel_tests, "Lead-lag chart": panel_leadlag,
    "FII flow calendar": panel_calendar, "SIP phases": panel_sip, "Correlation heatmap": panel_heatmap,
    "What if": panel_whatif, "Daily vs monthly": panel_daily_vs_monthly,
    "Methodology": panel_methodology, "Downloads": panel_downloads,
    "Regime track record": panel_backtest, "Client update": panel_client_update,
    "Sector flows": panel_sector_flows, "Generate report": panel_report,
    "Market mood": panel_market_mood, "Data health": panel_data_health,
}

def render_rows(rows) -> None:
    for names, widths in rows:
        cols = st.columns(widths, gap="medium")
        for col, name in zip(cols, names):
            with col:
                RENDER[name]()


def render_footer() -> None:
    st.markdown(H(f'<div class="card-sub" style="margin-top:22px">Sources: NSE/NSDL FII, DII and F&amp;O data (end of day); '
                  f'Nifty via {E(source)}; watchlist via Yahoo Finance (delayed). A project by Akhil Madhu.</div>'),
               unsafe_allow_html=True)


def page_overview():
    render_rows(PAGE_OVERVIEW)
    render_footer()


def page_flows():
    render_rows(PAGE_FLOWS)
    render_footer()


def page_statistics():
    render_rows(PAGE_STATISTICS)
    render_footer()


def page_research():
    render_rows(PAGE_RESEARCH)
    render_footer()


def page_health():
    render_rows(PAGE_HEALTH)
    render_footer()


def page_reports():
    render_rows(PAGE_REPORTS)
    render_footer()


pg = st.navigation([
    st.Page(page_overview, title="Overview", icon="\U0001F4CA", default=True),
    st.Page(page_flows, title="FII & DII Flows", icon="\U0001F4B0"),
    st.Page(page_statistics, title="Statistics", icon="\U0001F9EE"),
    st.Page(page_research, title="Sector & SIP Research", icon="\U0001F50D"),
    st.Page(page_health, title="Market Health", icon="\U0001FA7A"),
    st.Page(page_reports, title="Reports", icon="\U0001F4C4"),
])
pg.run()
