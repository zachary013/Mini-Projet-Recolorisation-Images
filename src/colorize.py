"""
Script de colorisation d'images historiques.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from model import get_model
from utils import preprocess_image, lab_to_rgb, combine_lab_channels, visualize_results


def colorize_images(input_path, model_path="results/model_final.pth"):
    """
    Colorise les images d'un dossier.
    
    Args:
        input_path: Chemin vers le dossier d'images à coloriser
        model_path: Chemin vers le modèle entraîné
    """
    
    # Vérifier que le modèle existe
    if not Path(model_path).exists():
        print(f"❌ Modèle non trouvé: {model_path}")
        print("Entraînez d'abord le modèle avec: python main.py --mode train")
        return
    
    # Charger le modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model("simple").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"✅ Modèle chargé depuis {model_path}")
    
    # Créer le dossier de sortie
    output_path = Path("results/predictions")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Lister les images à coloriser
    input_dir = Path(input_path)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(input_dir.glob(f"*{ext}"))
        image_paths.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not image_paths:
        print(f"❌ Aucune image trouvée dans {input_path}")
        return
    
    print(f"🎨 Colorisation de {len(image_paths)} images...")
    
    for image_path in image_paths:
        try:
            colorized_image = colorize_single_image(model, image_path, device)
            
            # Sauvegarder l'image colorisée
            output_file = output_path / f"colorized_{image_path.name}"
            cv2.imwrite(str(output_file), cv2.cvtColor(colorized_image, cv2.COLOR_RGB2BGR))
            
            print(f"✅ {image_path.name} -> {output_file}")
            
        except Exception as e:
            print(f"❌ Erreur avec {image_path.name}: {e}")
    
    print(f"🎉 Colorisation terminée ! Résultats dans {output_path}")


def colorize_single_image(model, image_path, device):
    """
    Colorise une seule image.
    
    Args:
        model: Modèle PyTorch
        image_path: Chemin vers l'image
        device: Device (CPU/GPU)
        
    Returns:
        Image colorisée (numpy array RGB)
    """
    
    # Charger et préprocesser l'image
    original_image = cv2.imread(str(image_path))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_size = original_image.shape[:2]
    
    # TODO: Implémenter la colorisation
    # 1. Redimensionner à 256x256
    resized = cv2.resize(original_image, (256, 256))
    
    # 2. Convertir en niveaux de gris et normaliser
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    gray_normalized = gray.astype(np.float32) / 255.0
    
    # 3. Convertir en tensor PyTorch
    gray_tensor = torch.from_numpy(gray_normalized).unsqueeze(0).unsqueeze(0).to(device)
    
    # 4. Prédiction avec le modèle
    with torch.no_grad():
        ab_pred = model(gray_tensor)
    
    # 5. Reconstruire l'image LAB
    L_channel = gray_tensor.squeeze().cpu().numpy()
    ab_channels = ab_pred.squeeze().cpu().numpy()
    
    # 6. Combiner les canaux L et ab
    lab_image = np.zeros((256, 256, 3))
    lab_image[:, :, 0] = L_channel
    lab_image[:, :, 1] = ab_channels[0]
    lab_image[:, :, 2] = ab_channels[1]
    
    # 7. Convertir LAB vers RGB
    colorized = lab_to_rgb(lab_image)
    
    # 8. Redimensionner à la taille originale
    colorized = cv2.resize(colorized, (original_size[1], original_size[0]))
    
    return colorized


def colorize_with_visualization(image_path, model_path="results/model_final.pth"):
    """
    Colorise une image et affiche le résultat.
    
    Args:
        image_path: Chemin vers l'image
        model_path: Chemin vers le modèle
    """
    
    # Charger le modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model("simple").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Charger l'image originale
    original = cv2.imread(str(image_path))
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Créer la version en niveaux de gris
    grayscale = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    
    # Coloriser
    colorized = colorize_single_image(model, image_path, device)
    
    # Visualiser les résultats
    visualize_results(original, grayscale, colorized)


def batch_colorize_and_compare(input_dir, model_path="results/model_final.pth"):
    """
    Colorise un batch d'images et crée des comparaisons visuelles.
    
    Args:
        input_dir: Dossier d'images d'entrée
        model_path: Chemin vers le modèle
    """
    
    input_path = Path(input_dir)
    image_paths = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    
    # Charger le modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model("simple").to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Créer une grille de comparaisons
    n_images = min(4, len(image_paths))  # Maximum 4 images
    fig, axes = plt.subplots(n_images, 3, figsize=(15, 5*n_images))
    
    if n_images == 1:
        axes = axes.reshape(1, -1)
    
    for i, image_path in enumerate(image_paths[:n_images]):
        # Charger l'image
        original = cv2.imread(str(image_path))
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        grayscale = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        
        # Coloriser
        colorized = colorize_single_image(model, image_path, device)
        
        # Afficher
        axes[i, 0].imshow(original)
        axes[i, 0].set_title(f"Original - {image_path.name}")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(grayscale, cmap='gray')
        axes[i, 1].set_title("Niveaux de gris")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(colorized)
        axes[i, 2].set_title("Colorisé")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig("results/batch_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Test de colorisation
    colorize_images("data/historical")
