# FII × Nifty 50 Live Platform — Quick Start (Laptop)

## 1. Unzip and open terminal in this folder

```bash
cd live_data_platform
```

## 2. Create virtual environment & install

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Mac / Linux:
source .venv/bin/activate

pip install -r requirements.txt
# Optional (Zerodha):
# pip install kiteconnect
```

## 3. (Optional) Real-time brokers & alerts

```bash
cp .env.example .env
# Edit .env with Upstox / Zerodha / Telegram / Email keys
```

## 4. Run the dashboard

```bash
streamlit run src/dashboard/app.py
```

Browser opens at http://localhost:8501

## 5. Alerts only (cron / after market)

```bash
python scripts/run_alerts.py
```

---

**What you get**
- Real FII/DII + F&O data (NSE/NSDL via free API)
- Nifty 50 live (yfinance; Upstox/Zerodha if configured)
- Correlation, DII absorption, FII streaks, F&O LS ratio regime
- Telegram + Email alerts on sell-pressure regimes

See README.md for full architecture.
