"""Abstract broker live-data adapter."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from src.ingestion.sources import Tick


class BrokerAdapter(ABC):
    name: str = "base"

    @abstractmethod
    def is_configured(self) -> bool:
        ...

    @abstractmethod
    def get_nifty_quote(self) -> Optional[Tick]:
        """Live Nifty 50 (or equivalent index) quote."""
        ...

    def get_quotes(self, symbols: List[str]) -> List[Tick]:
        """Optional multi-symbol quotes."""
        return []
