# 🎨 Projet de Recolorisation Automatique d'Images

**Redonnez vie à vos images en noir et blanc grâce à l'intelligence artificielle !**

Ce projet utilise un modèle de Deep Learning (inspiré de U-Net) pour coloriser automatiquement des photos en niveaux de gris. L'objectif n'est pas seulement de deviner les couleurs, mais de produire un résultat plausible, réaliste et sémantiquement cohérent.

![Image de démonstration](https://github.com/user-attachments/assets/a9e5f8fb-27e8-4a4a-88e0-af576f294521)

---

## ✨ Galerie de Résultats

Voici quelques exemples de ce que notre modèle peut faire.

|                   Avant (N&B)                    |            Après (Colorisé par l'IA)             |
|:------------------------------------------------:|:------------------------------------------------:|
| <img src="/assets/image1-avant.jpg" width="300"> | <img src="/assets/image1-apres.png" width="300"> |
| <img src="/assets/image2-avant.jpg" width="300"> | <img src="/assets/image2-apres.png" width="300"> |

---

## 🚀 Essayez-le vous-même ! (Application Web)

La manière la plus simple de tester notre modèle est de lancer l'application web interactive.

**1. Installez les dépendances :**
```bash
# Assurez-vous d'avoir installé les prérequis
pip3 install -r requirements.txt
```

**2. Lancez l'application :**
```bash
streamlit run app.py
```

**3. Ouvrez votre navigateur :**
Rendez-vous sur l'adresse **http://localhost:8501** et déposez simplement votre image !

---

## 🛠️ Comment ça marche ? Le côté technique

Ce projet n'est pas magique ! Il repose sur des concepts solides de vision par ordinateur :

*   **Espace Couleur LAB** : On ne prédit que les informations de couleur (canaux `a` et `b`) à partir de la luminosité (canal `L`), ce qui simplifie la tâche.
*   **Architecture U-Net** : Un réseau de neurones de type Encoder-Decoder avec des "skip connections" qui permettent de conserver les détails fins de l'image originale, évitant ainsi un résultat flou.
*   **Fonction de Perte L1** : Encourage le modèle à produire des couleurs plus vives et moins "moyennes" ou grisâtres.
*   **L-Channel Grafting** : Une technique de post-traitement qui réinjecte la luminosité de l'image originale en haute résolution pour un résultat final net et détaillé.

---

## 📂 Structure du Projet

Le projet est organisé de manière modulaire pour faciliter la compréhension et les contributions.

```
Mini-Projet-Recolorisation-Images/
├── app.py             # L'application web Streamlit ✨
├── data/              # Dossiers pour les images d'entraînement et de test
├── models/            # Modèles PyTorch pré-entraînés (.pth)
├── results/           # Images colorisées, graphiques et métriques
├── src/               # Le code source du modèle, de l'entraînement, etc.
├── requirements.txt   # Les dépendances Python à installer
└── README.md          # Ce fichier !
```

---

## 👨‍💻 Pour les Développeurs : Entraînement et Utilisation en Ligne de Commande

Si vous souhaitez aller plus loin, vous pouvez entraîner le modèle ou l'utiliser directement depuis le terminal.

### 1. Prérequis

Assurez-vous d'avoir Python 3.8+ et d'avoir installé les dépendances :
```bash
pip3 install -r requirements.txt
```

### 2. Entraîner le Modèle

Pour lancer un nouvel entraînement sur les données du dossier `data/train/` :
```bash
python3 main.py --mode train
```
*Le script sauvegardera des checkpoints du modèle dans le dossier `models/`.*

### 3. Coloriser une Image

Pour coloriser une image ou un dossier d'images :
```bash
# Coloriser le dossier d'images historiques par défaut
python3 main.py --mode colorize

# Spécifier un dossier d'entrée et un modèle
python3 main.py --mode colorize --input /chemin/vers/vos/images --model models/model_epoch_50.pth
```
*Les résultats seront sauvegardés dans `results/predictions/`.*

---

## 👥 L'Équipe

Ce projet a été réalisé par :

| Avatar | Nom | GitHub |
|---|---|---|
| <img src="https://github.com/zachary013.png" width="50" height="50" style="border-radius: 50%"/> | Zakariae Azarkan | [@zachary013](https://github.com/zachary013) |
| <img src="https://github.com/badrbenabdellah.png" width="50" height="50" style="border-radius: 50%"/> | Badr Benabdellah | [@badrbenabdellah](https://github.com/badrbenabdellah) |
| <img src="https://github.com/bouba-34.png" width="50" height="50" style="border-radius: 50%"/> | Sangare Boubacar | [@bouba-34](https://github.com/bouba-34) |

*Sous la supervision du **Pr. M'hamed AIT KBIR**.*
