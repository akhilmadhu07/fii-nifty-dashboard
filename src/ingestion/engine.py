"""
Data Ingestion Engine.
Pulls from all sources on a configurable cadence and pushes into the database layer.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import logging
logger = logging.getLogger(__name__)

from src.config import settings
from src.ingestion.sources import BaseSource, Tick, create_all_sources


class IngestionEngine:
    def __init__(
        self,
        sources: Optional[Dict[str, BaseSource]] = None,
        on_ticks: Optional[Callable[[List[Tick]], None]] = None,
    ):
        self.sources = sources or create_all_sources(settings.ingestion.symbols)
        self.on_ticks = on_ticks
        self._running = False
        self._tick_count = 0
        self.interval = settings.ingestion.tick_interval_sec

    def start(self, max_ticks: Optional[int] = None) -> None:
        """Blocking loop – use in a background thread or asyncio task in production."""
        self._running = True
        logger.info(f"Ingestion engine started | interval={self.interval}s")
        while self._running:
            if max_ticks and self._tick_count >= max_ticks:
                break
            self._cycle()
            time.sleep(self.interval)

    def stop(self) -> None:
        self._running = False
        logger.info("Ingestion engine stopped")

    def _cycle(self) -> None:
        all_ticks: List[Tick] = []
        for name, src in self.sources.items():
            try:
                ticks = src.next_ticks()
                all_ticks.extend(ticks)
            except Exception as e:
                logger.error(f"Source {name} failed: {e}")
        if all_ticks and self.on_ticks:
            self.on_ticks(all_ticks)
        self._tick_count += 1
        if self._tick_count % 30 == 0:
            logger.debug(f"Ingested {self._tick_count} cycles, last batch size={len(all_ticks)}")

    def single_cycle(self) -> List[Tick]:
        """Useful for dashboard-driven polling."""
        all_ticks: List[Tick] = []
        for src in self.sources.values():
            all_ticks.extend(src.next_ticks())
        if self.on_ticks:
            self.on_ticks(all_ticks)
        self._tick_count += 1
        return all_ticks
