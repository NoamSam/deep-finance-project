# Deep Finance Project

Multi-asset price prediction platform with local model training, a backtesting engine, and an interactive Streamlit dashboard.

## Features

- Multi-asset market data ingestion (Yahoo Finance), with local caching in `app/data/`
- Deep learning model training (TensorFlow) with a training cache to avoid redundant runs
- Backtesting engine with portfolio optimization
- Interactive Streamlit dashboard to explore predictions, curves, and portfolio results

## Requirements

- **Python 3.12** (tested with 3.12.12)
- Avoid Python 3.14: incompatible with the TensorFlow stack used here

## Installation

### Quick option (macOS / Linux)

```bash
bash setup.sh
source venv/bin/activate
```

### Manual option (macOS / Linux)

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Manual option (Windows PowerShell)

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the app

```bash
source venv/bin/activate
streamlit run app/streamlit_app.py
```

Then open http://localhost:8501

## First run

- If `app/data/` is empty, the app downloads price history from Yahoo Finance
- Data files and local caches are then rebuilt automatically in `app/data/`
- The first run is therefore slower than subsequent ones

## Disclaimer

Educational project. Nothing here is investment advice.
