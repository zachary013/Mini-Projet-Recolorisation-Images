# Projet de Recolorisation d'Images Historiques
![recolorisation](https://github.com/user-attachments/assets/a9e5f8fb-27e8-4a4a-88e0-af576f294521)

## Résumé

Ce projet implémente un système de vision artificielle pour la colorisation automatique d'images historiques en noir et blanc. L'approche utilise un réseau de neurones convolutionnel (CNN) avec architecture encoder-decoder et skip connections pour apprendre à restituer des couleurs plausibles et réalistes à partir d'images en niveaux de gris.

## Objectifs du Projet

Le projet vise à développer un modèle capable de :
- Convertir automatiquement des images couleur en noir et blanc pour constituer un dataset d'entraînement
- Entraîner un modèle CNN pour prédire les informations chromatiques manquantes
- Évaluer quantitativement la qualité des résultats avec des métriques PSNR et SSIM
- Effectuer une évaluation qualitative subjective des colorisations produites
- Appliquer le modèle entraîné sur de véritables images historiques

## Méthodologie

### Architecture du Modèle

Le modèle utilise une architecture CNN encoder-decoder inspirée des U-Net avec les caractéristiques suivantes :

- **Espace colorimétrique LAB** : Séparation de la luminance (L) et de la chrominance (a,b)
- **Encoder** : 4 couches convolutionnelles avec BatchNorm, ReLU et MaxPooling
- **Decoder** : 3 couches de déconvolution avec skip connections
- **Skip connections** : Préservation des détails fins par concaténation des features maps
- **Fonction d'activation finale** : Sigmoid pour normalisation [0,1]

### Preprocessing des Données

- **Redimensionnement** : Images normalisées à 256×256 pixels
- **Augmentation de données** : Flip horizontal (50%) et rotation aléatoire (±15°)
- **Normalisation LAB** : L ∈ [0,1], a,b ∈ [0,1] pour compatibilité avec Sigmoid
- **L-channel grafting** : Préservation de la luminance originale haute résolution

### Entraînement

- **Fonction de perte** : L1 Loss pour éviter les colorisations grises
- **Optimiseur** : Adam avec learning rate 0.001
- **Split train/validation** : 80/20 avec monitoring des pertes
- **Sauvegarde** : Checkpoints automatiques tous les 10 époques

## Dataset

Le dataset utilisé comprend :
- **Images d'entraînement** : 3,050 images couleur (split 80/20 = 2,440 train / 610 validation)
- **Images de test** : 608 images pour l'évaluation quantitative
- **Images historiques** : 1 image en noir et blanc pour démonstration

## Structure du Projet

```
Mini-Projet-Recolorisation-Images/
├── data/
│   ├── train/           # 3,050 images d'entraînement
│   ├── test/            # 608 images de test
│   └── historical/      # 1 image historique à coloriser
├── src/
│   ├── model.py         # Architecture CNN avec skip connections
│   ├── train.py         # Script d'entraînement avec validation
│   ├── utils.py         # Métriques PSNR/SSIM, dataset, conversions
│   ├── colorize.py      # Script de colorisation avec L-channel grafting
│   └── subjective_evaluation.py  # Évaluation qualitative
├── models/
│   └── *.pth           # Modèles entraînés (checkpoints)
├── results/
│   ├── predictions/     # Images colorisées
│   ├── *.png           # Graphiques et grilles d'évaluation
│   └── *.json          # Métriques et rapports
├── requirements.txt     # Dépendances Python
├── README.md           # Documentation
└── main.py             # Point d'entrée principal
```

## Installation et Configuration

### Prérequis
- Python 3.8+
- CUDA (optionnel, pour accélération GPU)

### Installation
```bash
# Cloner le projet
git clone <repository-url>
cd Mini-Projet-Recolorisation-Images

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# ou
.venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### Entraînement du Modèle
```bash
# Entraînement standard (50 époques)
python main.py --mode train

# Entraînement personnalisé
python main.py --mode train --epochs 30 --batch-size 32 --learning-rate 0.0005
```

### Colorisation d'Images
```bash
# Coloriser les images historiques
python main.py --mode colorize --input data/historical

# Utiliser un checkpoint spécifique
python main.py --mode colorize --input data/historical --model models/model_epoch_10.pth

# Coloriser un dossier personnalisé
python main.py --mode colorize --input /chemin/vers/images
```

### Évaluation des Résultats
```bash
# Évaluation quantitative (PSNR, SSIM)
python main.py --mode evaluate

# Démonstration sur une image
python main.py --mode demo --input data/test

# Évaluation complète (colorisation + métriques + analyse)
python main.py --mode complete
```

## Métriques d'Évaluation

### Métriques Quantitatives
- **PSNR (Peak Signal-to-Noise Ratio)** : Mesure la qualité de reconstruction
  - Valeurs > 20 dB considérées comme acceptables
  - Valeurs > 30 dB considérées comme bonnes
- **SSIM (Structural Similarity Index)** : Mesure la similarité structurelle
  - Valeurs > 0.7 considérées comme bonnes
  - Valeurs > 0.9 considérées comme excellentes

### Évaluation Qualitative
- Grilles de comparaison visuelle (Original vs Gris vs Colorisé)
- Analyse de la distribution des couleurs
- Rapport d'évaluation subjective avec critères définis

## Résultats Générés

Le système produit automatiquement :
- `results/predictions/` : Images colorisées en haute résolution
- `results/evaluation_metrics.json` : Métriques PSNR/SSIM détaillées
- `results/training_loss.png` : Courbes d'entraînement et validation
- `results/subjective_evaluation_grid.png` : Grille de comparaison visuelle
- `results/qualitative_assessment.json` : Rapport d'évaluation qualitative
- `results/colorization_demo.png` : Démonstration de colorisation

## Technologies Utilisées

- **PyTorch 2.0+** : Framework de deep learning
- **OpenCV 4.8+** : Traitement d'images et conversions colorimétriques
- **scikit-image** : Métriques de qualité d'image (PSNR, SSIM)
- **Matplotlib** : Visualisation et génération de graphiques
- **NumPy** : Calculs numériques et manipulation d'arrays
- **Pillow** : Manipulation d'images complémentaire

## Limitations et Améliorations Futures

### Limitations Actuelles
- Dataset relativement petit (3,050 images d'entraînement)
- Colorisation parfois trop saturée sur certains objets
- Difficultés avec les objets rares non vus pendant l'entraînement
- Transitions de couleurs parfois abruptes

### Améliorations Proposées
- Augmentation de la taille du dataset avec plus de diversité
- Implémentation d'une loss function combinée (MSE + L1 + Perceptual)
- Ajout de techniques d'attention pour améliorer la cohérence spatiale
- Fine-tuning sur des images historiques spécifiques
- Intégration de modèles pré-entraînés (transfer learning)

## Contributions et Développement

Le projet est structuré de manière modulaire pour faciliter les extensions :
- `src/model.py` : Modifications de l'architecture
- `src/train.py` : Ajustements des hyperparamètres d'entraînement
- `src/utils.py` : Nouvelles métriques ou preprocessing
- `src/colorize.py` : Améliorations du pipeline de colorisation

## Equipe

| Avatar                                                                                                  | Name | GitHub                                                 |
|---------------------------------------------------------------------------------------------------------|------|--------------------------------------------------------|
| <img src="https://github.com/zachary013.png" width="50" height="50" style="border-radius: 50%"/>        | Zakariae Azarkan | [@zachary013](https://github.com/zachary013)           |
| <img src="https://github.com/badrbenabdellah.png" width="50" height="50" style="border-radius: 50%"/>          | Badr Benabdellah | [@badrbenabdellah](https://github.com/badrbenabdellah) |
| <img src="https://github.com/bouba-34.png" width="50" height="50" style="border-radius: 50%"/>          | Sangare Boubacar | [@bouba-34](https://github.com/bouba-34)               |

---

*Ce projet constitue une implémentation complète d'un système de colorisation automatique, de l'entraînement à l'évaluation, avec une attention particulière portée à la qualité des résultats et à la reproductibilité des expériences.*
