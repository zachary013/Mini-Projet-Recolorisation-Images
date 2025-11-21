# Projet de Recolorisation d'Images Historiques
![recolorisation](https://github.com/user-attachments/assets/b82123a4-f373-41f3-acd7-26edceb80988)


## Description
Application de vision artificielle pour la colorisation automatique d'images historiques en noir et blanc. Le projet utilise des réseaux de neurones convolutionnels (CNN) avec architecture encoder-decoder et skip connections pour apprendre à restituer des couleurs plausibles et réalistes.

## Objectifs
- ✅ Convertir des images couleur en noir et blanc pour l'entraînement
- ✅ Entraîner un modèle CNN pour la recolorisation
- ✅ Évaluer la qualité avec des métriques quantitatives (PSNR, SSIM)
- ✅ Évaluation subjective avec grilles de comparaison
- ✅ Appliquer le modèle sur des images historiques réelles

## Architecture Technique
- **Modèle**: CNN Encoder-Decoder avec Skip Connections
- **Espace colorimétrique**: LAB (L pour luminance, AB pour couleur)
- **Loss function**: Combinaison MSE (70%) + L1 (30%)
- **Optimiseur**: Adam avec weight decay
- **Validation**: Split train/validation avec monitoring

## Structure du Projet
```
Mini-Projet-Recolorisation-Images/
├── data/
│   ├── train/           # 4363 images d'entraînement
│   ├── test/            # 362 images de test
│   └── historical/      # 34 images historiques à coloriser
├── src/
│   ├── model.py         # Architecture CNN avec skip connections
│   ├── train.py         # Entraînement avec validation
│   ├── utils.py         # Métriques PSNR/SSIM, dataset
│   ├── colorize.py      # Script de colorisation
│   └── subjective_evaluation.py  # Évaluation qualitative
├── models/
│   └── *.pth           # Modèles entraînés
├── results/
│   ├── predictions/     # Images colorisées
│   ├── *.png           # Graphiques et grilles
│   └── *.json          # Métriques et rapports
├── requirements.txt
├── README.md
└── main.py             # Point d'entrée principal
```

## Installation
```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # macOS/Linux
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### Entraînement
```bash
# Entraînement standard (50 époques)
python main.py --mode train

# Entraînement personnalisé
python main.py --mode train --epochs 30 --batch-size 32 --learning-rate 0.0005
```

### Colorisation
```bash
# Coloriser les images historiques
python main.py --mode colorize --input data/historical

# Coloriser un dossier personnalisé
python main.py --mode colorize --input /chemin/vers/images
```

### Évaluation
```bash
# Évaluation quantitative (PSNR, SSIM)
python main.py --mode evaluate

# Démonstration sur une image
python main.py --mode demo --input data/test

# Évaluation complète (colorisation + métriques + analyse)
python main.py --mode complete
```

## Résultats Générés

### Fichiers de sortie
- `results/predictions/` - Images colorisées
- `results/evaluation_metrics.json` - Métriques PSNR/SSIM
- `results/training_loss.png` - Courbes d'entraînement
- `results/subjective_evaluation_grid.png` - Grille de comparaison
- `results/qualitative_assessment.json` - Rapport qualitatif

### Métriques de Performance
- **PSNR**: Peak Signal-to-Noise Ratio (> 20 dB = bonne qualité)
- **SSIM**: Structural Similarity Index (> 0.7 = bonne similarité)

## Technologies
- **Python 3.8+**
- **PyTorch** - Deep Learning
- **OpenCV** - Traitement d'images
- **scikit-image** - Métriques de qualité
- **Matplotlib** - Visualisation
- **NumPy** - Calculs numériques

## Dataset
- **Entraînement**: 4363 images (split 80/20 train/validation)
- **Test**: 362 images pour l'évaluation
- **Historiques**: 34 images en noir et blanc à coloriser

## Auteur
Zakariae AZARKAN
