#!/usr/bin/env python3
"""
Standalone alert checker – run via cron after market close or every hour.
Example cron (IST):  30 17 * * 1-5  cd /path/to/live_data_platform && python scripts/run_alerts.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.alerts.notifier import AlertManager
from src.analytics.fii_nifty import full_fii_nifty_analysis
from src.ingestion.real_sources import RealDataProvider


def main():
    provider = RealDataProvider()
    provider.refresh_history_if_needed(force=True)
    analysis = full_fii_nifty_analysis(provider.fii_history, provider.nifty_history)
    if not analysis.get("ok"):
        print("Analysis failed:", analysis.get("message"))
        return 1
    alerter = AlertManager(
        trigger_regimes={"FII_SELL_PRESSURE", "DII_SUPPORTED"},
        cooldown_minutes=180,
        channels=["telegram", "email"],
    )
    results = alerter.check_and_alert(analysis)
    print("Regime:", analysis["regime"]["regime"])
    print("Alert results:", results or "none (no trigger or cooldown)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
