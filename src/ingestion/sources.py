"""
Simulated live data sources.
Replace these classes with real adapters (NSE, Kite, Polygon, FRED, etc.) later.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Generator, List, Optional

import numpy as np


@dataclass
class Tick:
    symbol: str
    ts: datetime
    price: float
    volume: float = 0.0
    bid: Optional[float] = None
    ask: Optional[float] = None
    source: str = "sim"
    meta: Dict = field(default_factory=dict)


class BaseSource:
    """Abstract base for any data feed."""

    def __init__(self, symbols: List[str], seed: Optional[int] = None):
        self.symbols = symbols
        self.rng = np.random.default_rng(seed)
        self._state: Dict[str, float] = {}
        self._init_prices()

    def _init_prices(self) -> None:
        raise NotImplementedError

    def next_ticks(self) -> List[Tick]:
        raise NotImplementedError


class EquitySource(BaseSource):
    """NSE-style equity ticks (GBM with mild mean reversion)."""

    def _init_prices(self) -> None:
        # Realistic starting levels (approx INR)
        bases = {
            "RELIANCE": 2950.0,
            "TCS": 3850.0,
            "INFY": 1850.0,
            "HDFCBANK": 1650.0,
            "SBIN": 820.0,
        }
        for s in self.symbols:
            self._state[s] = bases.get(s, 1000.0 + self.rng.uniform(-50, 50))

    def next_ticks(self) -> List[Tick]:
        ticks = []
        now = datetime.now(timezone.utc)
        for sym in self.symbols:
            price = self._state[sym]
            # Geometric Brownian Motion step
            mu, sigma = 0.00005, 0.0018
            shock = self.rng.normal(mu, sigma)
            new_price = max(price * math.exp(shock), 1.0)
            # occasional larger move
            if self.rng.random() < 0.02:
                new_price *= 1 + self.rng.choice([-1, 1]) * self.rng.uniform(0.003, 0.012)
            self._state[sym] = new_price
            vol = float(self.rng.integers(50, 2500))
            spread = new_price * 0.0004
            ticks.append(
                Tick(
                    symbol=sym,
                    ts=now,
                    price=round(new_price, 2),
                    volume=vol,
                    bid=round(new_price - spread / 2, 2),
                    ask=round(new_price + spread / 2, 2),
                    source="nse_sim",
                )
            )
        return ticks


class FXSource(BaseSource):
    """FX pairs with lower volatility."""

    def _init_prices(self) -> None:
        bases = {"USDINR": 83.45, "EURUSD": 1.085, "GBPUSD": 1.265}
        for s in self.symbols:
            self._state[s] = bases.get(s, 1.0)

    def next_ticks(self) -> List[Tick]:
        ticks = []
        now = datetime.now(timezone.utc)
        for sym in self.symbols:
            price = self._state[sym]
            sigma = 0.00035 if "INR" in sym else 0.00025
            shock = self.rng.normal(0, sigma)
            new_price = price * math.exp(shock)
            self._state[sym] = new_price
            ticks.append(
                Tick(
                    symbol=sym,
                    ts=now,
                    price=round(new_price, 4 if "INR" in sym else 5),
                    volume=0.0,
                    source="fx_sim",
                )
            )
        return ticks


class MacroSource(BaseSource):
    """Slow-moving macro / regime indicators."""

    def _init_prices(self) -> None:
        bases = {"NIFTY_VIX": 13.8, "INDIA_10Y": 7.05, "USD_INDEX": 104.2}
        for s in self.symbols:
            self._state[s] = bases.get(s, 10.0)

    def next_ticks(self) -> List[Tick]:
        ticks = []
        now = datetime.now(timezone.utc)
        for sym in self.symbols:
            price = self._state[sym]
            # very low frequency moves
            if self.rng.random() < 0.15:
                delta = self.rng.normal(0, 0.08 if "VIX" in sym else 0.015)
                new_price = max(price + delta, 0.1)
                self._state[sym] = new_price
            else:
                new_price = price
            ticks.append(
                Tick(
                    symbol=sym,
                    ts=now,
                    price=round(new_price, 2),
                    volume=0.0,
                    source="macro_sim",
                    meta={"type": "macro"},
                )
            )
        return ticks


def create_all_sources(cfg_symbols: Dict[str, List[str]]) -> Dict[str, BaseSource]:
    return {
        "equity": EquitySource(cfg_symbols.get("equity", [])),
        "fx": FXSource(cfg_symbols.get("fx", [])),
        "macro": MacroSource(cfg_symbols.get("macro", [])),
    }
