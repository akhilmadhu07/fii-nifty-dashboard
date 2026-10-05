"""
Groww API adapter (TOTP flow).

Setup:
1. groww.in -> profile -> Settings -> Trading APIs -> API keys
2. Click the dropdown next to "Generate API key" -> "Generate TOTP token"
3. Copy the TOTP Token (long string starting with eyJ...) and the TOTP Secret
4. Set env:
   GROWW_TOTP_TOKEN=...
   GROWW_TOTP_SECRET=...
5. pip install growwapi pyotp

Requires an active Groww Trading API subscription for live market data
(market-data endpoints return 403 without it).

Docs: https://groww.in/trade-api/docs/python-sdk
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import List, Optional

from src.ingestion.brokers.base import BrokerAdapter
from src.ingestion.sources import Tick

# How long we trust a generated access token before regenerating it.
# Groww's daily token-generation endpoint is rate-limited to 150 calls/24h,
# and the token itself is valid until ~6 AM IST the next day, so a coarse
# cache here keeps us far under that limit without needing per-request auth.
_TOKEN_TTL_SEC = 6 * 3600

# Symbol -> Groww "EXCHANGE_SYMBOL" key used by get_ltp / get_ohlc
_SYMBOL_MAP = {
    "NIFTY50": "NSE_NIFTY",
    "BANKNIFTY": "NSE_NIFTYBANK",
    "RELIANCE": "NSE_RELIANCE",
}


class GrowwAdapter(BrokerAdapter):
    name = "groww"

    def __init__(self, totp_token: Optional[str] = None, totp_secret: Optional[str] = None):
        self.totp_token = (totp_token or os.getenv("GROWW_TOTP_TOKEN", "")).strip()
        self.totp_secret = (totp_secret or os.getenv("GROWW_TOTP_SECRET", "")).strip()
        self._client = None
        self._token_generated_at: float = 0.0

    def is_configured(self) -> bool:
        return bool(self.totp_token and self.totp_secret)

    def _ensure_client(self):
        stale = (time.time() - self._token_generated_at) > _TOKEN_TTL_SEC
        if self._client is not None and not stale:
            return self._client
        if not self.is_configured():
            return None
        try:
            import pyotp
            from growwapi import GrowwAPI
        except ImportError:
            print("[groww] install dependencies: pip install growwapi pyotp")
            return None
        try:
            code = pyotp.TOTP(self.totp_secret).now()
            access_token = GrowwAPI.get_access_token(api_key=self.totp_token, totp=code)
            self._client = GrowwAPI(access_token)
            self._token_generated_at = time.time()
            return self._client
        except Exception as e:
            print(f"[groww] auth error: {e}")
            self._client = None
            return None

    def get_nifty_quote(self) -> Optional[Tick]:
        client = self._ensure_client()
        if not client:
            return None
        try:
            ltp = client.get_ltp(segment=client.SEGMENT_CASH, exchange_trading_symbols="NSE_NIFTY")
            price = float(ltp.get("NSE_NIFTY") or 0)
            if price <= 0:
                return None
            return Tick(
                symbol="NIFTY50",
                ts=datetime.now(timezone.utc),
                price=round(price, 2),
                volume=0.0,
                source="groww",
            )
        except Exception as e:
            print(f"[groww] quote error: {e}")
            return None

    def get_quotes(self, symbols: List[str]) -> List[Tick]:
        client = self._ensure_client()
        if not client:
            return []
        keys = [_SYMBOL_MAP[s] for s in symbols if s in _SYMBOL_MAP]
        if not keys:
            return []
        try:
            ltp = client.get_ltp(segment=client.SEGMENT_CASH, exchange_trading_symbols=tuple(keys))
            inv_map = {v: k for k, v in _SYMBOL_MAP.items()}
            ticks = []
            for key, price in ltp.items():
                if price and price > 0:
                    ticks.append(Tick(
                        symbol=inv_map.get(key, key),
                        ts=datetime.now(timezone.utc),
                        price=round(float(price), 2),
                        volume=0.0,
                        source="groww",
                    ))
            return ticks
        except Exception as e:
            print(f"[groww] multi-quote error: {e}")
            return []
