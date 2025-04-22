#!/bin/bash

# Nom du script : launch_servers.sh
# Usage : ./launch_servers.sh
# Description : Lance l'API FastAPI et l'interface Streamlit

# Activer l'environnement
source annot_venv/bin/activate

# Ports
API_PORT=8000
STREAMLIT_PORT=8501

# Fichiers
API_FILE="app/main.py"
API_MODULE="app.main"
STREAMLIT_FILE="app/afro_vision_ui.py"

# Lancer le serveur FastAPI en arrière-plan
echo "🚀 Lancement de FastAPI sur http://localhost:$API_PORT"
uvicorn $API_MODULE:app --host 0.0.0.0 --port $API_PORT --reload &

# Lancer l'interface Streamlit
echo "🖼️  Lancement de l'interface Streamlit sur http://localhost:$STREAMLIT_PORT"
streamlit run $STREAMLIT_FILE --server.address=0.0.0.0 --server.port=$STREAMLIT_PORT
