#!/bin/bash
set -e

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Python introuvable. Installe Python 3 puis relance ce script."
  exit 1
fi

echo "Creation de l'environnement virtuel..."
"$PYTHON_BIN" -m venv venv
. venv/bin/activate

echo "Installation des dependances..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Environnement pret."
echo "Activation macOS/Linux : source venv/bin/activate"
echo "Sous Windows, utilise plutot les commandes du README."
