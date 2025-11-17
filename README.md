# Projet de Recolorisation d'Images Historiques

## Description
Application de vision artificielle pour la colorisation automatique d'images historiques en noir et blanc. Le projet utilise des réseaux de neurones convolutionnels (CNN) pour apprendre à restituer des couleurs plausibles et réalistes.

## Objectifs
- Convertir des images couleur en noir et blanc pour l'entraînement
- Entraîner un modèle CNN pour la recolorisation
- Évaluer la qualité avec des métriques (PSNR, SSIM)
- Appliquer le modèle sur des images historiques réelles

## Structure du Projet
```
Mini-Projet-Recolorisation-Images/
├── data/
│   ├── train/           # Images d'entraînement
│   ├── test/            # Images de test
│   └── historical/      # Images historiques à coloriser
├── src/
│   ├── model.py         # Architecture du modèle CNN
│   ├── train.py         # Entraînement
│   ├── utils.py         # Fonctions utilitaires (métriques, preprocessing)
│   └── colorize.py      # Script de colorisation
├── results/
│   └── predictions/     # Images colorisées
├── notebooks/
│   └── demo.ipynb       # Démonstration et tests
├── requirements.txt
├── README.md
└── main.py             # Point d'entrée simple
```

## Installation
```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation
```bash
# Entraînement du modèle
python main.py --mode train

# Colorisation d'images
python main.py --mode colorize --input data/historical

# Évaluation
python main.py --mode evaluate
```

## Technologies
- Python 3.8+
- PyTorch
- OpenCV
- scikit-image
