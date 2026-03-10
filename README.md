# Deep Finance Project

Projet de prediction multi-actifs avec:
- dashboard Streamlit
- entrainement local du modele

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
