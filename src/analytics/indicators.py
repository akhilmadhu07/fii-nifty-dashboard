"""
Analytics Engine – Technical, Statistical & Macro/Regime models.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
import logging
logger = logging.getLogger(__name__)

from src.config import settings


def compute_technical(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators. Expects columns: price (and optionally volume)."""
    if df.empty or "price" not in df.columns:
        return df

    out = df.copy()
    close = out["price"]

    cfg = settings.analytics.technical

    # SMAs / EMAs
    for p in cfg.sma_periods:
        out[f"sma_{p}"] = ta.sma(close, length=p)
    for p in cfg.ema_periods:
        out[f"ema_{p}"] = ta.ema(close, length=p)

    # RSI
    out["rsi"] = ta.rsi(close, length=cfg.rsi_period)

    # MACD
    macd = ta.macd(close, fast=cfg.macd[0], slow=cfg.macd[1], signal=cfg.macd[2])
    if macd is not None:
        out = out.join(macd)

    # Bollinger Bands
    bb = ta.bbands(close, length=cfg.bbands_period)
    if bb is not None:
        out = out.join(bb)

    # ATR (needs high/low – we approximate with price ± small noise if missing)
    if "high" not in out.columns:
        out["high"] = close * 1.001
        out["low"] = close * 0.999
    out["atr"] = ta.atr(out["high"], out["low"], close, length=14)

    return out


def compute_statistical(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Rolling z-score, realized vol, simple returns."""
    if df.empty or "price" not in df.columns:
        return df

    out = df.copy()
    close = out["price"]
    out["ret"] = close.pct_change()
    out["log_ret"] = np.log(close / close.shift(1))
    out["realized_vol"] = out["log_ret"].rolling(lookback).std() * np.sqrt(252 * 6.5 * 60)  # rough annualised
    out["zscore"] = (close - close.rolling(lookback).mean()) / close.rolling(lookback).std()
    return out


def detect_regime(
    equity_df: pd.DataFrame,
    vix_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Very simple risk-on / risk-off detector.
    - High VIX z-score or elevated ATR → Risk-Off
    - Otherwise Risk-On
    """
    cfg = settings.analytics.regime
    result = {
        "regime": "NEUTRAL",
        "confidence": 0.5,
        "vix_level": None,
        "vix_z": None,
        "equity_vol_z": None,
    }

    # VIX path
    if vix_df is not None and not vix_df.empty and "price" in vix_df.columns:
        vix = vix_df["price"].dropna()
        if len(vix) >= cfg.vol_lookback:
            mean = vix.rolling(cfg.vol_lookback).mean().iloc[-1]
            std = vix.rolling(cfg.vol_lookback).std().iloc[-1]
            z = (vix.iloc[-1] - mean) / std if std and std > 0 else 0.0
            result["vix_level"] = float(vix.iloc[-1])
            result["vix_z"] = float(z)
            if z > cfg.risk_off_threshold:
                result["regime"] = "RISK_OFF"
                result["confidence"] = min(0.95, 0.55 + z * 0.15)
            elif z < -0.8:
                result["regime"] = "RISK_ON"
                result["confidence"] = min(0.9, 0.55 + abs(z) * 0.12)

    # Equity vol fallback / reinforcement
    if equity_df is not None and not equity_df.empty and "price" in equity_df.columns:
        rets = equity_df["price"].pct_change().dropna()
        if len(rets) >= cfg.vol_lookback:
            vol = rets.rolling(cfg.vol_lookback).std().iloc[-1]
            vol_mean = rets.rolling(cfg.vol_lookback * 3).std().mean()
            vol_z = (vol - vol_mean) / (rets.rolling(cfg.vol_lookback * 3).std().std() + 1e-9)
            result["equity_vol_z"] = float(vol_z)
            if result["regime"] == "NEUTRAL":
                if vol_z > 1.2:
                    result["regime"] = "RISK_OFF"
                    result["confidence"] = 0.65
                elif vol_z < -0.7:
                    result["regime"] = "RISK_ON"
                    result["confidence"] = 0.6

    return result


def full_analytics(
    symbol: str,
    price_df: pd.DataFrame,
    vix_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Convenience wrapper used by the dashboard."""
    tech = compute_technical(price_df)
    stats = compute_statistical(tech)
    regime = detect_regime(stats, vix_df)

    latest = {}
    if not stats.empty:
        row = stats.iloc[-1]
        latest = {
            "price": float(row.get("price", np.nan)),
            "rsi": float(row.get("rsi", np.nan)) if pd.notna(row.get("rsi")) else None,
            "sma_21": float(row.get("sma_21", np.nan)) if pd.notna(row.get("sma_21")) else None,
            "ema_9": float(row.get("ema_9", np.nan)) if pd.notna(row.get("ema_9")) else None,
            "zscore": float(row.get("zscore", np.nan)) if pd.notna(row.get("zscore")) else None,
            "realized_vol": float(row.get("realized_vol", np.nan)) if pd.notna(row.get("realized_vol")) else None,
        }

    return {
        "symbol": symbol,
        "df": stats,
        "latest": latest,
        "regime": regime,
    }
