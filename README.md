# Deep Finance Project

Projet de prediction multi-actifs avec:
- dashboard Streamlit
- entrainement local du modele
- scripts de sanity checks pour les previsions et l'allocation de portefeuille

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 1. Lancer l'app

```bash
venv/bin/streamlit run app/streamlit_app.py
```

Ouvrir ensuite `http://localhost:8501`.

## 2. Lancer un entrainement simple

```bash
venv/bin/python app/train_runner.py \
  --csv app/data/assets/AAPL.csv \
  --model lstm \
  --epochs 1 \
  --window-size 120 \
  --horizon 5 \
  --batch-size 64 \
  --learning-rate 0.0005
```

## 3. Lancer le backtest marche avec export

```bash
venv/bin/python scripts/market_sanity_check.py \
  --assets AAPL MSFT NVDA \
  --start 2023-01-01 \
  --end 2026-02-28 \
  --horizon 5 \
  --epochs 1 \
  --num-cutoffs 6 \
  --spacing-steps 5 \
  --history-lookback 252 \
  --output-dir results/market_check
```

Fichiers generes:
- `results/market_check/asset_details.csv`
- `results/market_check/asset_summary.csv`
- `results/market_check/portfolio_summary.csv`
- `results/market_check/summary.json`

## Structure

- `app/` : runtime de l'application Streamlit
- `scripts/` : scripts de benchmark, sanity checks et reporting
- `app/data/` : donnees locales et caches (non versionnes)
