#!/bin/bash

# Nom du script : install_dependencies.sh
# Usage : ./install_dependencies.sh
# Description : Crée un venv, active-le, installe les dépendances requises

echo "📦 Création d'un environnement virtuel..."
python3 -m venv annot_venv
source annot_venv/bin/activate

echo "⬇️ Installation des dépendances..."
pip install --upgrade pip

# Installe les dépendances principales
pip install fastapi uvicorn python-multipart pillow transformers streamlit requests lottie

# Optionnel : requirements.txt si présent
if [ -f "requirements.txt" ]; then
    echo "📂 Fichier requirements.txt trouvé. Installation..."
    pip install -r requirements.txt
fi

echo "✅ Installation terminée. Environnement prêt !"
