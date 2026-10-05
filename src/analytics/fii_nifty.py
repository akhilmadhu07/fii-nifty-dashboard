"""
FII Outflows ↔ Nifty 50 Influence Analytics
+ F&O index-futures positioning (long/short ratio, net) in regime logic.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _prepare_merged(fii_df: pd.DataFrame, nifty_df: pd.DataFrame) -> pd.DataFrame:
    if fii_df.empty or nifty_df.empty:
        return pd.DataFrame()

    f = fii_df.copy()
    n = nifty_df.copy()

    if "date" not in f.columns:
        return pd.DataFrame()
    f["date"] = pd.to_datetime(f["date"], dayfirst=True, errors="coerce").dt.normalize()
    n["date"] = pd.to_datetime(n["date"]).dt.normalize()
    f = f.dropna(subset=["date"])

    # Cash nets
    for col, alts in {
        "fii_net": ["fn", "FII_Net", "fiiNet"],
        "dii_net": ["dn", "DII_Net", "diiNet"],
    }.items():
        if col not in f.columns:
            for a in alts:
                if a in f.columns:
                    f[col] = f[a]
                    break
            else:
                f[col] = np.nan

    # F&O index futures
    for col in ("fii_idx_fut_long", "fii_idx_fut_short"):
        if col not in f.columns:
            f[col] = np.nan
    f["fii_idx_fut_net"] = f["fii_idx_fut_long"] - f["fii_idx_fut_short"]
    # Long/Short ratio (avoid div0)
    f["fii_ls_ratio"] = np.where(
        f["fii_idx_fut_short"] > 0,
        f["fii_idx_fut_long"] / f["fii_idx_fut_short"],
        np.nan,
    )
    if "pcr" not in f.columns:
        f["pcr"] = np.nan

    n = n.sort_values("date")
    n["nifty_ret"] = n["close"].pct_change()
    n["nifty_close"] = n["close"]

    merged = pd.merge(
        f[
            [
                "date", "fii_net", "dii_net",
                "fii_idx_fut_long", "fii_idx_fut_short", "fii_idx_fut_net",
                "fii_ls_ratio", "pcr",
            ]
        ].dropna(subset=["fii_net"]),
        n[["date", "nifty_close", "nifty_ret"]],
        on="date",
        how="inner",
    )
    return merged.sort_values("date").reset_index(drop=True)


def correlation_analysis(merged: pd.DataFrame, max_lag: int = 5) -> Dict[str, Any]:
    out = {"same_day": None, "lags": {}, "best_lag": None, "best_corr": None}
    if merged.empty or "fii_net" not in merged.columns or "nifty_ret" not in merged.columns:
        return out
    s = merged.dropna(subset=["fii_net", "nifty_ret"])
    if len(s) < 20:
        return out
    same = float(s["fii_net"].corr(s["nifty_ret"]))
    out["same_day"] = round(same, 3)
    best_lag, best_corr = 0, same
    for lag in range(1, max_lag + 1):
        corr = float(s["fii_net"].corr(s["nifty_ret"].shift(-lag)))
        out["lags"][f"fii_t → nifty_t+{lag}"] = round(corr, 3) if pd.notna(corr) else None
        if pd.notna(corr) and abs(corr) > abs(best_corr):
            best_corr, best_lag = corr, lag
    for lag in range(1, 3):
        corr = float(s["nifty_ret"].corr(s["fii_net"].shift(-lag)))
        out["lags"][f"nifty_t → fii_t+{lag}"] = round(corr, 3) if pd.notna(corr) else None
    out["best_lag"] = best_lag
    out["best_corr"] = round(best_corr, 3) if best_corr is not None else None
    return out


def cumulative_impact(merged: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
    if merged.empty:
        return {}
    s = merged.copy()
    s["cum_fii"] = s["fii_net"].rolling(window, min_periods=5).sum()
    s["cum_dii"] = s["dii_net"].rolling(window, min_periods=5).sum()
    s["nifty_ret_w"] = s["nifty_close"].pct_change(window)
    last = s.iloc[-1]
    return {
        "window_days": window,
        "cum_fii_cr": round(float(last["cum_fii"]), 1) if pd.notna(last["cum_fii"]) else None,
        "cum_dii_cr": round(float(last["cum_dii"]), 1) if pd.notna(last["cum_dii"]) else None,
        "nifty_return_pct": round(float(last["nifty_ret_w"]) * 100, 2) if pd.notna(last["nifty_ret_w"]) else None,
        "series": s[["date", "cum_fii", "cum_dii", "nifty_close"]].tail(60),
    }


def dii_absorption(merged: pd.DataFrame, lookback: int = 10) -> Dict[str, Any]:
    if merged.empty:
        return {}
    s = merged.tail(lookback).copy()
    fii_sell_days = s[s["fii_net"] < 0]
    if fii_sell_days.empty:
        return {"lookback": lookback, "fii_sell_days": 0, "avg_absorption_pct": None}
    abs_ratio = (fii_sell_days["dii_net"] / (-fii_sell_days["fii_net"])).clip(-2, 3)
    return {
        "lookback": lookback,
        "fii_sell_days": int(len(fii_sell_days)),
        "avg_absorption_pct": round(float(abs_ratio.mean()) * 100, 1),
        "total_fii_sold_cr": round(float(fii_sell_days["fii_net"].sum()), 1),
        "total_dii_bought_cr": round(float(fii_sell_days["dii_net"].sum()), 1),
    }


def fii_streak(merged: pd.DataFrame) -> Dict[str, Any]:
    if merged.empty or "fii_net" not in merged.columns:
        return {"streak": 0, "direction": "NONE", "cum_flow_cr": 0.0, "last_net_cr": 0.0}
    nets = merged["fii_net"].dropna().values
    if len(nets) == 0:
        return {"streak": 0, "direction": "NONE", "cum_flow_cr": 0.0, "last_net_cr": 0.0}
    last = nets[-1]
    direction = "BUY" if last > 0 else "SELL" if last < 0 else "FLAT"
    streak, cum = 1, float(last)
    for i in range(len(nets) - 2, -1, -1):
        if (direction == "BUY" and nets[i] > 0) or (direction == "SELL" and nets[i] < 0):
            streak += 1
            cum += float(nets[i])
        else:
            break
    return {
        "streak": streak,
        "direction": direction,
        "cum_flow_cr": round(cum, 1),
        "last_net_cr": round(float(last), 1),
    }


def fo_positioning(merged: pd.DataFrame) -> Dict[str, Any]:
    """Latest *non-zero* FII index-futures long/short + PCR (API sometimes zeros recent days)."""
    out = {
        "fii_idx_fut_long": None,
        "fii_idx_fut_short": None,
        "fii_idx_fut_net": None,
        "fii_ls_ratio": None,
        "pcr": None,
        "fo_bias": "NEUTRAL",
        "fo_as_of": None,
    }
    if merged.empty:
        return out
    # Prefer last row with meaningful futures OI
    usable = merged[
        (merged["fii_idx_fut_long"].fillna(0) > 0) | (merged["fii_idx_fut_short"].fillna(0) > 0)
    ]
    last = usable.iloc[-1] if not usable.empty else merged.iloc[-1]
    for k in ("fii_idx_fut_long", "fii_idx_fut_short", "fii_idx_fut_net", "fii_ls_ratio", "pcr"):
        v = last.get(k)
        out[k] = round(float(v), 2) if pd.notna(v) else None
    if "date" in last.index and pd.notna(last["date"]):
        out["fo_as_of"] = str(pd.Timestamp(last["date"]).date())

    # Bias rules of thumb:
    # LS ratio < 0.8 → net short (bearish lean)
    # LS ratio > 1.3 → net long (bullish lean)
    ratio = out["fii_ls_ratio"]
    net = out["fii_idx_fut_net"]
    if ratio is not None:
        if ratio < 0.8 or (net is not None and net < -100000):
            out["fo_bias"] = "BEARISH"
        elif ratio > 1.3 or (net is not None and net > 50000):
            out["fo_bias"] = "BULLISH"
    return out


def regime_from_flows(
    merged: pd.DataFrame,
    streak_info: Dict,
    absorption: Dict,
    fo: Dict,
) -> Dict[str, Any]:
    """
    Regime incorporating cash flows + F&O index futures positioning.
    """
    regime = "NEUTRAL"
    confidence = 0.5
    reason = []

    st = streak_info.get("streak", 0)
    direction = streak_info.get("direction", "NONE")
    abs_pct = absorption.get("avg_absorption_pct")
    fo_bias = fo.get("fo_bias", "NEUTRAL")
    ls = fo.get("fii_ls_ratio")

    # Cash-driven base
    if direction == "SELL" and st >= 3:
        if abs_pct is not None and abs_pct >= 80:
            regime = "DII_SUPPORTED"
            confidence = min(0.85, 0.55 + st * 0.05)
            reason.append(f"FII selling streak {st}d but DII absorbing ~{abs_pct:.0f}%")
        else:
            regime = "FII_SELL_PRESSURE"
            confidence = min(0.9, 0.6 + st * 0.06)
            reason.append(f"FII selling streak {st}d, limited DII support")
    elif direction == "BUY" and st >= 3:
        regime = "FII_LED_RALLY"
        confidence = min(0.88, 0.55 + st * 0.07)
        reason.append(f"FII buying streak {st}d")
    elif direction == "SELL" and st >= 1:
        regime = "MILD_FII_OUTFLOW"
        confidence = 0.55
        reason.append("Recent FII net selling")
    elif direction == "BUY" and st >= 1:
        regime = "MILD_FII_INFLOW"
        confidence = 0.55
        reason.append("Recent FII net buying")

    # F&O overlay – amplify or flip mild regimes
    if fo_bias == "BEARISH":
        reason.append(f"F&O: FII index-fut LS ratio={ls} (net short bias)")
        if regime in ("MILD_FII_OUTFLOW", "NEUTRAL"):
            regime = "FII_SELL_PRESSURE"
            confidence = max(confidence, 0.7)
        elif regime == "DII_SUPPORTED":
            confidence = min(confidence + 0.05, 0.9)
            reason.append("Cash supported by DII but F&O still short")
        elif regime == "FII_LED_RALLY":
            confidence = max(0.5, confidence - 0.15)
            reason.append("Cash buying but F&O still net short → caution")
    elif fo_bias == "BULLISH":
        reason.append(f"F&O: FII index-fut LS ratio={ls} (net long bias)")
        if regime in ("MILD_FII_INFLOW", "NEUTRAL"):
            regime = "FII_LED_RALLY"
            confidence = max(confidence, 0.65)
        elif regime == "FII_SELL_PRESSURE":
            confidence = max(0.5, confidence - 0.1)
            reason.append("Cash selling but F&O long → possible short-cover bounce")

    return {
        "regime": regime,
        "confidence": round(confidence, 2),
        "reason": " | ".join(reason) if reason else "Mixed / low signal",
        "streak": streak_info,
        "absorption": absorption,
        "fo": fo,
    }


def full_fii_nifty_analysis(
    fii_df: pd.DataFrame,
    nifty_df: pd.DataFrame,
) -> Dict[str, Any]:
    merged = _prepare_merged(fii_df, nifty_df)
    if merged.empty:
        return {
            "ok": False,
            "message": "Insufficient overlapping FII + Nifty history",
            "merged_rows": 0,
        }

    corr = correlation_analysis(merged)
    cum = cumulative_impact(merged, window=20)
    abs_info = dii_absorption(merged, lookback=15)
    streak = fii_streak(merged)
    fo = fo_positioning(merged)
    regime = regime_from_flows(merged, streak, abs_info, fo)

    last = merged.iloc[-1]
    latest = {
        "date": str(last["date"].date()) if pd.notna(last["date"]) else None,
        "fii_net_cr": round(float(last["fii_net"]), 1),
        "dii_net_cr": round(float(last["dii_net"]), 1) if pd.notna(last["dii_net"]) else None,
        "nifty_close": round(float(last["nifty_close"]), 2),
        "nifty_ret_pct": round(float(last["nifty_ret"]) * 100, 2) if pd.notna(last["nifty_ret"]) else None,
    }

    return {
        "ok": True,
        "merged_rows": len(merged),
        "latest": latest,
        "correlation": corr,
        "cumulative_20d": {k: v for k, v in cum.items() if k != "series"},
        "cumulative_series": cum.get("series"),
        "dii_absorption": abs_info,
        "streak": streak,
        "fo": fo,
        "regime": regime,
        "merged_tail": merged.tail(30),
    }
