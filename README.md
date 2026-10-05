# Live Data Platform

A modular, production-style architecture for real-time financial data ingestion, storage, analytics, and live dashboards.

```
LIVE DATA SOURCES
       │
┌──────┼──────┐
↓      ↓      ↓
NSE/Broker  FX/Macro  Other feeds
       │
       ↓
DATA INGESTION ENGINE
(every tick / 1s / 1m / EOD)
       │
       ↓
    DATABASE
       │
┌──────┴──────┐
↓             ↓
REAL-TIME   HISTORICAL
ENGINE         DB
       │
       ↓
ANALYTICS ENGINE
┌──────┼──────┐
↓      ↓      ↓
Technical  Statistical  Macro/Regime
       │
       ↓
 LIVE DASHBOARD
```

## Features

- **Simulated live feeds**: Equity ticks (NSE-style), FX pairs, macro indicators
- **Ingestion engine**: Configurable cadence (tick / 1s / 1min / EOD)
- **Dual storage**: In-memory real-time ring buffer + SQLite/Timescale-ready historical DB
- **Analytics**:
  - Technical: SMA, EMA, RSI, MACD, Bollinger Bands, ATR
  - Statistical: rolling z-scores, volatility regimes
  - Macro/Regime: simple risk-on / risk-off detector
- **Live Dashboard**: Streamlit + Plotly with auto-refresh

## Quick Start

```bash
cd live_data_platform
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the full pipeline + dashboard
streamlit run src/dashboard/app.py
```

The dashboard will open at http://localhost:8501 and start streaming simulated live data.

## Project Structure

```
live_data_platform/
├── config/
│   └── settings.yaml
├── src/
│   ├── ingestion/          # Data source adapters + ingestion engine
│   ├── database/           # Real-time + historical stores
│   ├── engines/            # Real-time processing engine
│   ├── analytics/          # Technical / Statistical / Regime models
│   └── dashboard/          # Streamlit live UI
├── data/                   # SQLite DB & sample files
├── scripts/                # Utility scripts
├── tests/
├── requirements.txt
└── README.md
```

## Extending to Real Feeds

1. Replace simulated sources in `src/ingestion/sources.py` with:
   - NSE: `nsepython` or official NSE APIs
   - Broker: Zerodha Kite, Upstox, Angel One, etc.
   - FX/Macro: Polygon, Twelve Data, FRED, Yahoo, etc.
2. Add API keys to `.env`
3. Update `config/settings.yaml`

## Architecture Notes

- **Ingestion** is async-friendly and can be scaled with Kafka / Redis Streams later.
- **Real-time engine** keeps a fixed-size rolling window for low-latency indicators.
- **Historical DB** uses SQLAlchemy + SQLite (swap to TimescaleDB / ClickHouse for production).
- **Analytics** are pure functions → easy to unit test and swap models.

## License

MIT

---

## FII × Nifty 50 Focus (current)

The platform now prioritises **FII outflows / inflows vs Nifty 50 influence**.

### Real data sources
| Data | Source | Cadence |
|------|--------|---------|
| Nifty 50 live | yfinance (`^NSEI`) + optional nsepython | Near real-time |
| FII / DII cash | Free API: `fii-diidata.mrchartist.com` (NSE provisional + NSDL) | Daily EOD (~5–7 PM IST) |
| F&O participant OI | Same API | Daily EOD |

### Analytics included
- Same-day & lagged correlation (FII net → Nifty returns)
- 20-day cumulative FII/DII vs Nifty return
- DII absorption ratio on FII sell days
- FII buy/sell streak + regime (FII_SELL_PRESSURE / DII_SUPPORTED / FII_LED_RALLY …)

### Run the focused dashboard
```bash
streamlit run src/dashboard/app.py
```

### Important limitation
**True intraday FII cash flows are not public.**  
What you get in real time is Nifty price + the latest published daily FII/DII context and historical influence metrics. For tick-level institutional data you need a broker or paid data vendor.
