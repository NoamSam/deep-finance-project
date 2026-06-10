# Quant Portfolio Cockpit

![Python](https://img.shields.io/badge/python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit)

Multi-asset quantitative research cockpit: LSTM price forecasting, walk-forward backtesting, and mean-variance portfolio optimization, wrapped in an interactive Streamlit dashboard.

Universe: French equities (CAC-listed tickers) and US assets, with market history pulled from Yahoo Finance and cached locally.

## What it does

**1. Forecasting (TensorFlow / Keras)**
- LSTM models trained per asset (Huber loss, gradient clipping, dropout)
- A multi-feature variant enriched with US market factors (SPY, QQQ, VIX returns) alongside OHLC data
- Training cache keyed on data and hyperparameters to avoid redundant runs

**2. Walk-forward backtesting**
- Periodic rebalancing on model forecasts, compared against a naive baseline and a benchmark
- Performance metrics: returns, drawdown, turnover, NAV curve
- Backtest cache for instant re-display of previously computed configurations

**3. Portfolio optimization**
- Mean-variance optimization (SciPy) with long-only constraints and per-asset weight caps
- Risk aversion derived from investor profile and horizon
- Correlation-based universe filtering and covariance estimation

**4. Streamlit cockpit**
- Four tabs: predictions, backtest, summary, allocation
- Altair charts, progress feedback during training, cache management

## Project layout

```text
app/
├── streamlit_app.py        # dashboard (4 tabs)
├── models.py               # LSTM architectures (Keras)
├── train_runner.py         # training subprocess runner
├── training_cache.py       # training result cache
├── backtest_engine.py      # walk-forward engine + performance metrics
├── backtest_cache.py       # backtest result cache
├── portfolio_optimizer.py  # mean-variance optimization
├── us_market_features.py   # US factor features (SPY, QQQ, VIX)
├── update_curves.py        # Yahoo Finance data ingestion
└── base_tickers.txt        # asset universe
```

## Requirements

- **Python 3.12** (tested with 3.12.12). Avoid 3.14: incompatible with the TensorFlow stack used here.

## Installation

### Quick (macOS / Linux)

```bash
bash setup.sh
source venv/bin/activate
```

### Manual (macOS / Linux)

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Manual (Windows PowerShell)

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
source venv/bin/activate
streamlit run app/streamlit_app.py
```

Open http://localhost:8501. On first run, the app downloads price history from Yahoo Finance into `app/data/`; subsequent runs reuse the local cache.

## Disclaimer

Educational research project. Nothing here is investment advice.
