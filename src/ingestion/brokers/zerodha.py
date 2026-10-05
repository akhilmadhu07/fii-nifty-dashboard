"""
Zerodha Kite Connect live adapter scaffold.

Setup:
1. Create app at https://developers.kite.trade/
2. Login flow → request_token → access_token
3. Set env:
   KITE_API_KEY=...
   KITE_ACCESS_TOKEN=...

Install: pip install kiteconnect

Docs: https://kite.trade/docs/connect/v3/
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import List, Optional

from src.ingestion.brokers.base import BrokerAdapter
from src.ingestion.sources import Tick

# Nifty 50 instrument token on NSE (stable for index)
NIFTY50_INSTRUMENT_TOKEN = 256265  # NSE:NIFTY 50


class ZerodhaAdapter(BrokerAdapter):
    name = "zerodha"

    def __init__(
        self,
        api_key: Optional[str] = None,
        access_token: Optional[str] = None,
    ):
        self.api_key = (api_key or os.getenv("KITE_API_KEY", "")).strip()
        self.access_token = (access_token or os.getenv("KITE_ACCESS_TOKEN", "")).strip()
        self._kite = None

    def is_configured(self) -> bool:
        return bool(self.api_key and self.access_token)

    def _client(self):
        if self._kite is not None:
            return self._kite
        if not self.is_configured():
            return None
        try:
            from kiteconnect import KiteConnect
            kite = KiteConnect(api_key=self.api_key)
            kite.set_access_token(self.access_token)
            self._kite = kite
            return kite
        except ImportError:
            print("[zerodha] Install kiteconnect: pip install kiteconnect")
            return None
        except Exception as e:
            print(f"[zerodha] client init error: {e}")
            return None

    def get_nifty_quote(self) -> Optional[Tick]:
        kite = self._client()
        if not kite:
            return None
        try:
            q = kite.quote(["NSE:NIFTY 50"])
            data = q.get("NSE:NIFTY 50") or {}
            price = float(data.get("last_price") or 0)
            if price <= 0:
                return None
            depth = data.get("depth") or {}
            bid = None
            ask = None
            if depth.get("buy"):
                bid = depth["buy"][0].get("price")
            if depth.get("sell"):
                ask = depth["sell"][0].get("price")
            return Tick(
                symbol="NIFTY50",
                ts=datetime.now(timezone.utc),
                price=round(price, 2),
                volume=float(data.get("volume") or 0),
                bid=bid,
                ask=ask,
                source="zerodha",
            )
        except Exception as e:
            print(f"[zerodha] quote error: {e}")
            return None

    def get_quotes(self, symbols: List[str]) -> List[Tick]:
        kite = self._client()
        if not kite:
            return []
        # Map to exchange:symbol
        instruments = []
        sym_map = {}
        for s in symbols:
            if s in ("NIFTY50", "NIFTY 50"):
                key = "NSE:NIFTY 50"
            else:
                key = f"NSE:{s}"
            instruments.append(key)
            sym_map[key] = s if s != "NIFTY 50" else "NIFTY50"
        try:
            q = kite.quote(instruments)
            ticks = []
            for key, data in q.items():
                price = float(data.get("last_price") or 0)
                if price <= 0:
                    continue
                ticks.append(
                    Tick(
                        symbol=sym_map.get(key, key),
                        ts=datetime.now(timezone.utc),
                        price=round(price, 2),
                        volume=float(data.get("volume") or 0),
                        source="zerodha",
                    )
                )
            return ticks
        except Exception as e:
            print(f"[zerodha] multi-quote error: {e}")
            return []

    def ltp(self, exchange_symbol: str = "NSE:NIFTY 50") -> Optional[float]:
        kite = self._client()
        if not kite:
            return None
        try:
            data = kite.ltp([exchange_symbol])
            return float(data[exchange_symbol]["last_price"])
        except Exception:
            return None
