#!/bin/zsh

# 1. Vérifier si Python 3 est installé
if ! command -v python &>/dev/null; then
  echo "Python n'est pas installé. Veuillez l'installer avant de continuer."
  exit 1
fi

# 2. Créer un environnement virtuel
ENV_DIR="venv"
if [ ! -d "$ENV_DIR" ]; then
  echo "Création de l'environnement virtuel..."
  python -m venv "$ENV_DIR"
fi

# 3. Activer l'environnement virtuel
source "$ENV_DIR/bin/activate"

# 4. Mettre à jour pip
echo "Mise à jour de pip..."
python -m pip install --upgrade pip

# 5. Installer les dépendances depuis le fichier requirements.txt
echo "Installation des dépendances nécessaires..."
python -m pip install -r requirements.txt
# python -m pip install torch opencv-python easyocr

# 6. Lancer le script Python
echo "Lancement du script Python..."
# python ../test_yolo_easyOCR_2.py
python $1

# 7. Désactiver l'environnement virtuel
deactivate