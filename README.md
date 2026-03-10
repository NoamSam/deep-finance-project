# Deep Finance Project

Projet de prediction multi-actifs avec:
- dashboard Streamlit
- entrainement local du modele

## Installation

### Option rapide (macOS / Linux)

```bash
bash setup.sh
source venv/bin/activate
```

### Option manuelle (macOS / Linux)

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Option manuelle (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 1. Lancer l'app

```bash
cd deep-finance-project
source venv/bin/activate
streamlit run app/streamlit_app.py
```

Ouvrir ensuite `http://localhost:8501`.

## Premier lancement

- l'application peut telecharger les historiques depuis Yahoo Finance si `app/data/` est vide
- les fichiers de donnees et les caches locaux sont ensuite recrees automatiquement dans `app/data/`
- le premier lancement peut donc etre plus lent que les suivants

## 2. Lancer un entrainement simple

```bash
cd deep-finance-project
source venv/bin/activate
python app/train_runner.py \
  --csv app/data/assets/AAPL.csv \
  --model lstm \
  --epochs 1 \
  --window-size 120 \
  --horizon 5 \
  --batch-size 64 \
  --learning-rate 0.0005
```
