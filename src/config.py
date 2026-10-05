"""Application configuration loader."""
from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "settings.yaml"


class AppConfig(BaseModel):
    name: str = "Live Data Platform"
    version: str = "0.1.0"
    log_level: str = "INFO"


class IngestionConfig(BaseModel):
    tick_interval_sec: float = 1.0
    symbols: Dict[str, List[str]] = Field(default_factory=dict)


class DatabaseConfig(BaseModel):
    historical_url: str = "sqlite:///data/historical.db"
    realtime_window_size: int = 500


class TechnicalConfig(BaseModel):
    sma_periods: List[int] = [9, 21, 50]
    ema_periods: List[int] = [9, 21]
    rsi_period: int = 14
    macd: List[int] = [12, 26, 9]
    bbands_period: int = 20


class RegimeConfig(BaseModel):
    vol_lookback: int = 20
    risk_off_threshold: float = 1.5


class AnalyticsConfig(BaseModel):
    technical: TechnicalConfig = Field(default_factory=TechnicalConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)


class DashboardConfig(BaseModel):
    refresh_interval_sec: int = 2
    default_symbol: str = "RELIANCE"
    chart_height: int = 450


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)


def load_settings(path: Path = CONFIG_PATH) -> Settings:
    if not path.exists():
        return Settings()
    with open(path, "r") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}
    return Settings(**raw)


# Singleton-style access
settings = load_settings()
