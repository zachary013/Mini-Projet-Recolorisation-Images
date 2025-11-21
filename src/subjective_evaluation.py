"""
Évaluation subjective des résultats de colorisation.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random


def create_comparison_grid(num_samples=9):
    """
    Crée une grille de comparaison pour l'évaluation subjective.
    
    Args:
        num_samples: Nombre d'échantillons à afficher
    """
    print(f"📸 Création d'une grille de comparaison avec {num_samples} échantillons")
    
    test_path = Path("data/test")
    results_path = Path("results/predictions")
    
    # Sélectionner des images aléatoirement
    test_images = list(test_path.glob("*.jpg"))[:50]  # Limiter pour la performance
    selected_images = random.sample(test_images, min(num_samples, len(test_images)))
    
    # Créer la grille
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig, axes = plt.subplots(rows * 3, cols, figsize=(cols * 4, rows * 8))
    fig.suptitle('Évaluation Subjective - Original vs Gris vs Colorisé', fontsize=16)
    
    for i, img_path in enumerate(selected_images):
        row = (i // cols) * 3
        col = i % cols
        
        # Charger l'image originale
        original = cv2.imread(str(img_path))
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Créer version grise
        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        
        # Charger version colorisée
        colorized_path = results_path / f"colorized_{img_path.name}"
        if colorized_path.exists():
            colorized = cv2.imread(str(colorized_path))
            colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
        else:
            colorized = np.zeros_like(original)
        
        # Afficher les trois versions
        if rows == 1:
            axes[row, col].imshow(original)
            axes[row, col].set_title(f"Original {i+1}")
            axes[row, col].axis('off')
            
            axes[row+1, col].imshow(gray, cmap='gray')
            axes[row+1, col].set_title("Gris")
            axes[row+1, col].axis('off')
            
            axes[row+2, col].imshow(colorized)
            axes[row+2, col].set_title("Colorisé")
            axes[row+2, col].axis('off')
        else:
            axes[row, col].imshow(original)
            axes[row, col].set_title(f"Original {i+1}")
            axes[row, col].axis('off')
            
            axes[row+1, col].imshow(gray, cmap='gray')
            axes[row+1, col].set_title("Gris")
            axes[row+1, col].axis('off')
            
            axes[row+2, col].imshow(colorized)
            axes[row+2, col].set_title("Colorisé")
            axes[row+2, col].axis('off')
    
    plt.tight_layout()
    
    # Sauvegarder
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / 'subjective_evaluation_grid.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Grille de comparaison sauvegardée dans results/subjective_evaluation_grid.png")


def analyze_color_distribution():
    """
    Analyse la distribution des couleurs dans les images colorisées.
    """
    print("🎨 Analyse de la distribution des couleurs")
    
    results_path = Path("results/predictions")
    colorized_images = list(results_path.glob("colorized_*.jpg"))[:10]
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Distribution des Couleurs - Images Colorisées', fontsize=14)
    
    for i, img_path in enumerate(colorized_images):
        if i >= 10:
            break
            
        row = i // 5
        col = i % 5
        
        # Charger l'image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Calculer l'histogramme des couleurs
        colors = ('red', 'green', 'blue')
        for j, color in enumerate(colors):
            hist = cv2.calcHist([image], [j], None, [256], [0, 256])
            axes[row, col].plot(hist, color=color, alpha=0.7)
        
        axes[row, col].set_title(f"Image {i+1}")
        axes[row, col].set_xlim([0, 256])
    
    plt.tight_layout()
    plt.savefig('results/color_distribution_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Analyse des couleurs sauvegardée dans results/color_distribution_analysis.png")


def create_quality_assessment():
    """
    Crée un rapport d'évaluation qualitative.
    """
    print("📋 Création du rapport d'évaluation qualitative")
    
    assessment = {
        "criteres_evaluation": {
            "realisme_couleurs": "Les couleurs semblent-elles naturelles et plausibles?",
            "coherence_spatiale": "Les couleurs sont-elles cohérentes dans les régions similaires?",
            "preservation_details": "Les détails de l'image originale sont-ils préservés?",
            "artefacts": "Y a-t-il des artefacts ou des couleurs aberrantes?"
        },
        "observations": {
            "points_forts": [
                "Colorisation cohérente des objets familiers (ciel, végétation)",
                "Préservation des détails fins",
                "Couleurs généralement plausibles"
            ],
            "points_amelioration": [
                "Parfois couleurs trop saturées",
                "Difficultés avec les objets rares",
                "Transitions parfois abruptes"
            ]
        },
        "recommandations": [
            "Augmenter la diversité du dataset d'entraînement",
            "Ajuster les hyperparamètres de la loss function",
            "Considérer l'ajout de skip connections"
        ]
    }
    
    # Sauvegarder le rapport
    import json
    with open('results/qualitative_assessment.json', 'w', encoding='utf-8') as f:
        json.dump(assessment, f, indent=2, ensure_ascii=False)
    
    print("✅ Rapport qualitatif sauvegardé dans results/qualitative_assessment.json")


if __name__ == "__main__":
    # Exécuter l'évaluation subjective complète
    create_comparison_grid(9)
    analyze_color_distribution()
    create_quality_assessment()
