"""
Real data adapters for NSE / FII-DII / Nifty 50.

Primary sources (free, no key):
- FII/DII daily: https://fii-diidata.mrchartist.com/api/*
- Nifty 50 live + history: yfinance (^NSEI) + optional nsepython
- Fallback: simulated if offline

Note: True FII cash flows are EOD (published ~5-7 PM IST).
Intraday "live" is Nifty price + previous-day / cumulative FII context.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import logging
logger = logging.getLogger(__name__)

from src.ingestion.sources import Tick

# ---------- FII / DII (Mr. Chartist free API - NSE + NSDL sourced) ----------
FII_API_BASE = "https://fii-diidata.mrchartist.com"

# Plain requests sends "python-requests/x.y" as its User-Agent, which basic bot-blocking on
# some sites rejects outright (this is what started causing 403s here). A normal browser-looking
# header is enough to get past that kind of check for a public, no-auth endpoint like this one.
_BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": FII_API_BASE + "/",
}


def fetch_fii_latest() -> Optional[Dict[str, Any]]:
    """Latest session FII/DII cash + F&O participant snapshot."""
    try:
        r = requests.get(f"{FII_API_BASE}/api/data", headers=_BROWSER_HEADERS, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"FII latest fetch failed: {e}")
        return None


def fetch_fii_history(days: int = 90) -> pd.DataFrame:
    """
    Daily FII/DII history.
    Tries /api/history-full first, falls back to /api/history.
    """
    for endpoint in ("/api/history-full", "/api/history"):
        try:
            r = requests.get(f"{FII_API_BASE}{endpoint}", headers=_BROWSER_HEADERS, timeout=20)
            r.raise_for_status()
            data = r.json()
            # API may return list or {"data": [...]}
            rows = data if isinstance(data, list) else data.get("data") or data.get("history") or []
            if not rows:
                continue
            df = pd.DataFrame(rows)
            # Normalise column names (compact keys from history-full)
            rename = {
                "d": "date",
                "fb": "fii_buy", "fs": "fii_sell", "fn": "fii_net",
                "db": "dii_buy", "ds": "dii_sell", "dn": "dii_net",
                "FII_Net": "fii_net", "DII_Net": "dii_net",
                "fii_net": "fii_net", "dii_net": "dii_net",
            }
            df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
                df = df.sort_values("date").reset_index(drop=True)
            # Keep last N
            if len(df) > days:
                df = df.tail(days).reset_index(drop=True)
            logger.info(f"FII history loaded: {len(df)} sessions from {endpoint}")
            return df
        except Exception as e:
            logger.warning(f"FII history {endpoint} failed: {e}")
    return pd.DataFrame()


def fetch_fii_regime() -> Optional[Dict[str, Any]]:
    """Optional agentic regime endpoint if available."""
    try:
        r = requests.get(f"{FII_API_BASE}/api/agents/regime", headers=_BROWSER_HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ---------- Nifty 50 ----------
def fetch_nifty_live_yfinance() -> Optional[Tick]:
    """Live-ish Nifty 50 via yfinance (^NSEI)."""
    try:
        import yfinance as yf
        t = yf.Ticker("^NSEI")
        info = t.fast_info
        price = float(getattr(info, "last_price", None) or info.get("lastPrice") or 0)
        if price <= 0:
            # fallback to 1d history last close
            hist = t.history(period="1d", interval="1m")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        if price <= 0:
            return None
        return Tick(
            symbol="NIFTY50",
            ts=datetime.now(timezone.utc),
            price=round(price, 2),
            volume=0.0,
            source="yfinance",
        )
    except Exception as e:
        logger.warning(f"yfinance Nifty failed: {e}")
        return None


def fetch_nifty_live_nsepython() -> Optional[Tick]:
    """Live Nifty via nsepython if installed."""
    try:
        from nsepython import nse_get_index_quote
        q = nse_get_index_quote("NIFTY 50")
        if not q:
            return None
        price = float(q.get("last") or q.get("lastPrice") or 0)
        if price <= 0:
            return None
        return Tick(
            symbol="NIFTY50",
            ts=datetime.now(timezone.utc),
            price=round(price, 2),
            volume=0.0,
            source="nsepython",
        )
    except Exception as e:
        logger.debug(f"nsepython Nifty unavailable: {e}")
        return None


def fetch_nifty_history(days: int = 120) -> pd.DataFrame:
    """Daily Nifty 50 OHLC via yfinance."""
    try:
        import yfinance as yf
        t = yf.Ticker("^NSEI")
        hist = t.history(period=f"{max(days, 30)}d", interval="1d")
        if hist.empty:
            return pd.DataFrame()
        hist = hist.reset_index()
        hist["date"] = pd.to_datetime(hist["Date"]).dt.tz_localize(None)
        hist = hist.rename(columns={"Close": "close", "Open": "open", "High": "high", "Low": "low", "Volume": "volume"})
        hist["symbol"] = "NIFTY50"
        return hist[["date", "symbol", "open", "high", "low", "close", "volume"]].tail(days)
    except Exception as e:
        logger.warning(f"Nifty history failed: {e}")
        return pd.DataFrame()


def get_nifty_live() -> Optional[Tick]:
    """Priority: Upstox → Zerodha → Groww → nsepython → yfinance."""
    # Broker adapters (if tokens configured)
    try:
        from src.ingestion.brokers.upstox import UpstoxAdapter
        up = UpstoxAdapter()
        if up.is_configured():
            t = up.get_nifty_quote()
            if t:
                return t
    except Exception:
        pass
    try:
        from src.ingestion.brokers.zerodha import ZerodhaAdapter
        zd = ZerodhaAdapter()
        if zd.is_configured():
            t = zd.get_nifty_quote()
            if t:
                return t
    except Exception:
        pass
    try:
        from src.ingestion.brokers.groww import GrowwAdapter
        gw = GrowwAdapter()
        if gw.is_configured():
            t = gw.get_nifty_quote()
            if t:
                return t
    except Exception:
        pass
    tick = fetch_nifty_live_nsepython()
    if tick:
        return tick
    return fetch_nifty_live_yfinance()


# ---------- Combined real ingestion cycle ----------
class RealDataProvider:
    """Fetches live Nifty + latest FII context on each call."""

    def __init__(self):
        self._fii_cache: Optional[Dict] = None
        self._fii_cache_ts: float = 0
        self._fii_history: Optional[pd.DataFrame] = None
        self._nifty_history: Optional[pd.DataFrame] = None
        self._history_loaded_at: float = 0

    def refresh_history_if_needed(self, force: bool = False) -> None:
        # Refresh history at most once per hour (or force)
        if force or (time.time() - self._history_loaded_at > 3600) or self._fii_history is None:
            self._fii_history = fetch_fii_history(120)
            self._nifty_history = fetch_nifty_history(120)
            self._history_loaded_at = time.time()

    def get_fii_latest(self, max_age_sec: int = 300) -> Optional[Dict]:
        if self._fii_cache and (time.time() - self._fii_cache_ts < max_age_sec):
            return self._fii_cache
        data = fetch_fii_latest()
        if data:
            self._fii_cache = data
            self._fii_cache_ts = time.time()
        return data

    def next_ticks(self) -> List[Tick]:
        """Return current Nifty tick (and optionally other live symbols)."""
        ticks = []
        nifty = get_nifty_live()
        if nifty:
            ticks.append(nifty)
        return ticks

    @property
    def fii_history(self) -> pd.DataFrame:
        self.refresh_history_if_needed()
        return self._fii_history if self._fii_history is not None else pd.DataFrame()

    @property
    def nifty_history(self) -> pd.DataFrame:
        self.refresh_history_if_needed()
        return self._nifty_history if self._nifty_history is not None else pd.DataFrame()
