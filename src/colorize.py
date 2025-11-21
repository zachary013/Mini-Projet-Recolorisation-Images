"""
Script de colorisation d'images avec le modèle entraîné.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from .model import ColorizationCNN
from .utils import rgb_to_lab, lab_to_rgb, tensor_to_image, combine_lab_channels


def colorize_images(input_path, model_path='models/colorization_model_final.pth'):
    """
    Colorise les images d'un dossier.
    
    Args:
        input_path: Chemin vers les images à coloriser
        model_path: Chemin vers le modèle entraîné
    """
    print(f"🎨 Colorisation des images de {input_path}")
    
    # Chargement du modèle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)
    
    # Dossier de sortie
    output_dir = Path('results/predictions')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Traitement des images
    input_dir = Path(input_path)
    image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
    
    if not image_files:
        print("❌ Aucune image trouvée dans le dossier")
        return
    
    for img_path in image_files:
        print(f"Traitement de {img_path.name}...")
        
        # Colorisation
        colorized = colorize_single_image(str(img_path), model, device)
        
        # Sauvegarde en PNG pour meilleure qualité
        output_path = output_dir / f"colorized_{img_path.stem}.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR))
        
        print(f"✅ Sauvegardé: {output_path}")
    
    print(f"🎉 Colorisation terminée ! {len(image_files)} images traitées")


def colorize_single_image(image_path, model, device):
    """
    Colorise une seule image.
    
    Args:
        image_path: Chemin vers l'image
        model: Modèle de colorisation
        device: Device de calcul
        
    Returns:
        Image colorisée (RGB)
    """
    # Chargement et préparation de l'image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    
    # Redimensionnement pour le modèle
    image_resized = cv2.resize(image, (256, 256))
    
    # Conversion en LAB et extraction du canal L
    lab_image = rgb_to_lab(image_resized)
    L_channel = lab_image[:, :, 0:1]
    
    # Préparation du tensor d'entrée
    L_tensor = torch.FloatTensor(L_channel).permute(2, 0, 1).unsqueeze(0)
    L_tensor = L_tensor.to(device)
    
    # Prédiction
    model.eval()
    with torch.no_grad():
        predicted_ab = model(L_tensor)
    
    # Reconstruction de l'image LAB
    L_np = tensor_to_image(L_tensor.squeeze(0))
    ab_np = tensor_to_image(predicted_ab.squeeze(0))
    
    lab_reconstructed = np.concatenate([L_np, ab_np], axis=2)
    
    # Conversion vers RGB
    rgb_colorized = lab_to_rgb(lab_reconstructed)
    
    # Redimensionnement à la taille originale
    rgb_colorized = cv2.resize(rgb_colorized, (original_size[1], original_size[0]))
    
    return rgb_colorized


def load_model(model_path, device):
    """Charge le modèle entraîné."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
    
    model = ColorizationCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"✅ Modèle chargé: {model_path}")
    return model


def demo_colorization(image_path, model_path='models/colorization_model_final.pth'):
    """
    Démonstration de colorisation avec affichage des résultats.
    
    Args:
        image_path: Chemin vers l'image de test
        model_path: Chemin vers le modèle
    """
    print(f"🎨 Démonstration de colorisation sur {image_path}")
    
    # Chargement du modèle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)
    
    # Chargement de l'image originale
    original = cv2.imread(image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Création de la version en niveaux de gris
    grayscale = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    
    # Colorisation
    colorized = colorize_single_image(image_path, model, device)
    
    # Affichage des résultats
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original)
    axes[0].set_title("Image originale")
    axes[0].axis('off')
    
    axes[1].imshow(grayscale, cmap='gray')
    axes[1].set_title("Niveaux de gris")
    axes[1].axis('off')
    
    axes[2].imshow(colorized)
    axes[2].set_title("Image colorisée")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Sauvegarde de la comparaison
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / 'colorization_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Démonstration terminée !")


if __name__ == "__main__":
    # Test de colorisation
    colorize_images('data/test')
