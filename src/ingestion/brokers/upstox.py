"""
Upstox live adapter scaffold.

Setup:
1. Create app at https://account.upstox.com/developer/apps
2. Get API key + secret, generate access token (OAuth or long-lived analytics token)
3. Set env:
   UPSTOX_ACCESS_TOKEN=...
   UPSTOX_API_KEY=...   (optional for some endpoints)

Docs:
- Market quotes: https://upstox.com/developer/api-documentation/get-market-quote-full
- FII data:     https://upstox.com/developer/api-documentation/get-fii-data
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List, Optional

import requests

from src.ingestion.brokers.base import BrokerAdapter
from src.ingestion.sources import Tick

UPSTOX_BASE = "https://api.upstox.com/v2"


class UpstoxAdapter(BrokerAdapter):
    name = "upstox"

    def __init__(self, access_token: Optional[str] = None, api_key: Optional[str] = None):
        self.access_token = (access_token or os.getenv("UPSTOX_ACCESS_TOKEN", "")).strip()
        self.api_key = (api_key or os.getenv("UPSTOX_API_KEY", "")).strip()

    def is_configured(self) -> bool:
        return bool(self.access_token)

    def _headers(self) -> dict:
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }

    def get_nifty_quote(self) -> Optional[Tick]:
        """
        Nifty 50 instrument key on Upstox is typically: NSE_INDEX|Nifty 50
        """
        if not self.is_configured():
            return None
        instrument_key = "NSE_INDEX|Nifty 50"
        url = f"{UPSTOX_BASE}/market-quote/quotes"
        try:
            r = requests.get(
                url,
                headers=self._headers(),
                params={"instrument_key": instrument_key},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json().get("data") or {}
            # Response shape: { "NSE_INDEX:Nifty 50": { last_price, ... } }
            quote = next(iter(data.values()), None) if data else None
            if not quote:
                return None
            price = float(quote.get("last_price") or quote.get("lastPrice") or 0)
            if price <= 0:
                return None
            depth = quote.get("depth") or {}
            buy_levels = depth.get("buy") or []
            sell_levels = depth.get("sell") or []
            bid = buy_levels[0].get("price") if buy_levels else None
            ask = sell_levels[0].get("price") if sell_levels else None
            return Tick(
                symbol="NIFTY50",
                ts=datetime.now(timezone.utc),
                price=round(price, 2),
                volume=float(quote.get("volume") or 0),
                bid=bid,
                ask=ask,
                source="upstox",
            )
        except Exception as e:
            print(f"[upstox] quote error: {e}")
            return None

    def get_fii_data(self, data_type: str = "NSE_CASH", interval: str = "1D") -> Optional[dict]:
        """
        Official Upstox FII endpoint (requires analytics-capable token).
        data_type examples: NSE_CASH, NSE_FO|INDEX_FUTURES, ...
        """
        if not self.is_configured():
            return None
        url = f"{UPSTOX_BASE}/market/fii"
        try:
            r = requests.get(
                url,
                headers=self._headers(),
                params={"data_type": data_type, "interval": interval},
                timeout=15,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[upstox] FII error: {e}")
            return None

    def get_quotes(self, symbols: List[str]) -> List[Tick]:
        # Map common symbols → instrument keys (extend as needed)
        key_map = {
            "NIFTY50": "NSE_INDEX|Nifty 50",
            "BANKNIFTY": "NSE_INDEX|Nifty Bank",
            "RELIANCE": "NSE_EQ|INE002A01018",
        }
        ticks = []
        for sym in symbols:
            ik = key_map.get(sym)
            if not ik:
                continue
            # Reuse single-quote logic by temporarily swapping – simple path
            if sym == "NIFTY50":
                t = self.get_nifty_quote()
                if t:
                    ticks.append(t)
        return ticks
