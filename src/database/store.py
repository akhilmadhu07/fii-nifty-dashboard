"""
Dual storage layer:
- RealTimeStore  : fixed-size in-memory ring buffer (low latency)
- HistoricalStore: SQLAlchemy + SQLite (persistent)
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional

import pandas as pd
import logging
logger = logging.getLogger(__name__)
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import settings
from src.ingestion.sources import Tick


# ---------- Real-time ring buffer ----------
class RealTimeStore:
    def __init__(self, window_size: int = None):
        self.window_size = window_size or settings.database.realtime_window_size
        self._buffers: Dict[str, Deque[Tick]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    def push(self, ticks: List[Tick]) -> None:
        for t in ticks:
            self._buffers[t.symbol].append(t)

    def get(self, symbol: str, n: Optional[int] = None) -> List[Tick]:
        buf = self._buffers.get(symbol, deque())
        if n is None:
            return list(buf)
        return list(buf)[-n:]

    def latest(self, symbol: str) -> Optional[Tick]:
        buf = self._buffers.get(symbol)
        return buf[-1] if buf else None

    def to_dataframe(self, symbol: str) -> pd.DataFrame:
        ticks = self.get(symbol)
        if not ticks:
            return pd.DataFrame()
        rows = [
            {
                "ts": t.ts,
                "symbol": t.symbol,
                "price": t.price,
                "volume": t.volume,
                "bid": t.bid,
                "ask": t.ask,
                "source": t.source,
            }
            for t in ticks
        ]
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"])
        return df.set_index("ts").sort_index()

    def symbols(self) -> List[str]:
        return list(self._buffers.keys())


# ---------- Historical SQL store ----------
class Base(DeclarativeBase):
    pass


class TickRecord(Base):
    __tablename__ = "ticks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), index=True, nullable=False)
    ts = Column(DateTime, index=True, nullable=False)
    price = Column(Float, nullable=False)
    volume = Column(Float, default=0.0)
    bid = Column(Float, nullable=True)
    ask = Column(Float, nullable=True)
    source = Column(String(32), default="sim")


class HistoricalStore:
    def __init__(self, url: str = None):
        url = url or settings.database.historical_url
        # ensure parent dir exists for sqlite
        if url.startswith("sqlite:///"):
            db_path = Path(url.replace("sqlite:///", ""))
            db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        logger.info(f"Historical DB ready → {url}")

    def push(self, ticks: List[Tick]) -> None:
        if not ticks:
            return
        with self.Session() as session:
            records = [
                TickRecord(
                    symbol=t.symbol,
                    ts=t.ts.replace(tzinfo=None) if t.ts.tzinfo else t.ts,
                    price=t.price,
                    volume=t.volume,
                    bid=t.bid,
                    ask=t.ask,
                    source=t.source,
                )
                for t in ticks
            ]
            session.add_all(records)
            session.commit()

    def query(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 5000,
    ) -> pd.DataFrame:
        q = "SELECT ts, symbol, price, volume, bid, ask, source FROM ticks WHERE symbol = :sym"
        params = {"sym": symbol}
        if start:
            q += " AND ts >= :start"
            params["start"] = start
        if end:
            q += " AND ts <= :end"
            params["end"] = end
        q += " ORDER BY ts DESC LIMIT :lim"
        params["lim"] = limit

        with self.engine.connect() as conn:
            df = pd.read_sql(text(q), conn, params=params)
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts"])
        return df.set_index("ts").sort_index()


# Convenience combined store
class DataStore:
    def __init__(self):
        self.realtime = RealTimeStore()
        self.historical = HistoricalStore()

    def push(self, ticks: List[Tick]) -> None:
        self.realtime.push(ticks)
        self.historical.push(ticks)
