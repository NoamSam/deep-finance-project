# Deep Finance Project

Projet de prediction multi-actifs avec:
- dashboard Streamlit
- entrainement local du modele

## Important

- Python 3.12 requis
- version testee: `Python 3.12.12`
- eviter Python 3.14, non compatible avec la stack TensorFlow utilisee ici

## Installation

### Option rapide (macOS / Linux)

```bash
bash setup.sh
source venv/bin/activate
```

### Option manuelle (macOS / Linux)

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Option manuelle (Windows PowerShell)

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Lancer l'app

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
