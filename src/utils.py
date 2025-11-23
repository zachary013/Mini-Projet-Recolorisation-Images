"""
Fonctions utilitaires pour le projet de recolorisation.
Contient les fonctions de preprocessing, métriques et dataset.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class ImageColorizationDataset(Dataset):
    """Dataset pour la recolorisation d'images."""
    
    def __init__(self, data_path, transform=None):
        """
        Initialise le dataset.
        
        Args:
            data_path: Chemin vers le dossier d'images
            transform: Transformations à appliquer
        """
        self.data_path = Path(data_path)
        self.image_paths = list(self.data_path.glob("*.jpg")) + list(self.data_path.glob("*.png"))
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Récupère un échantillon du dataset.
        
        Returns:
            Tuple (image_gray, color_channels) en format LAB
        """
        image_path = self.image_paths[idx]
        
        # Charger l'image couleur
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Redimensionner (modifiable selon les ressources)
        target_size = (512, 512)  # Haute résolution
        image = cv2.resize(image, target_size)
        
        # Data Augmentation : Flip horizontal (50% de chance)
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Convertir en LAB
        lab_image = rgb_to_lab(image)
        
        # Séparer les canaux
        L_channel = lab_image[:, :, 0:1]  # Canal L (luminance)
        ab_channels = lab_image[:, :, 1:3]  # Canaux a et b (couleur)
        
        # Convertir en tenseurs PyTorch
        L_tensor = torch.from_numpy(L_channel.transpose(2, 0, 1)).float()
        ab_tensor = torch.from_numpy(ab_channels.transpose(2, 0, 1)).float()
        
        return L_tensor, ab_tensor


def rgb_to_lab(image):
    """
    Convertit une image RGB en espace colorimétrique LAB normalisé pour le modèle.
    Args:
        image: Image RGB (numpy array) uint8 (0-255) ou float
    Returns:
        Image LAB normalisée [0, 1]
    """
    # 1. Normalisation en float32 [0, 1] AVANT la conversion
    image_float = image.astype(np.float32) / 255.0
    
    # 2. Conversion RGB -> LAB
    # OpenCV sortira : L [0, 100], a [-127, 127], b [-127, 127]
    lab = cv2.cvtColor(image_float, cv2.COLOR_RGB2LAB)
    
    # 3. Normalisation pour le modèle (Sigmoid attend [0, 1])
    lab_norm = np.zeros_like(lab)
    lab_norm[:, :, 0] = lab[:, :, 0] / 100.0           # L: 0-100 -> [0, 1]
    lab_norm[:, :, 1] = (lab[:, :, 1] + 128.0) / 255.0 # a: -128..127 -> [0, 1]
    lab_norm[:, :, 2] = (lab[:, :, 2] + 128.0) / 255.0 # b: -128..127 -> [0, 1]
    
    return lab_norm


def lab_to_rgb(lab_image):
    """
    Convertit une image LAB normalisée [0, 1] vers RGB uint8.
    """
    lab = np.zeros_like(lab_image)
    
    # 1. Dénormalisation [0, 1] -> Valeurs LAB standards
    lab[:, :, 0] = lab_image[:, :, 0] * 100.0              # L -> 0-100
    lab[:, :, 1] = (lab_image[:, :, 1] * 255.0) - 128.0    # a -> -128..127
    lab[:, :, 2] = (lab_image[:, :, 2] * 255.0) - 128.0    # b -> -128..127
    
    # 2. Conversion LAB -> RGB
    # OpenCV gère la conversion float -> float RGB [0, 1]
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # 3. Retour vers uint8 [0, 255]
    rgb_uint8 = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    
    return rgb_uint8


def tensor_to_image(tensor):
    """Convertit un tensor PyTorch en image numpy."""
    if tensor.dim() == 4:  # Batch dimension
        tensor = tensor.squeeze(0)
    
    image = tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    
    return image


def combine_lab_channels(L_channel, ab_channels):
    """
    Combine les canaux L et ab pour former une image LAB complète.
    
    Args:
        L_channel: Canal de luminance
        ab_channels: Canaux de couleur a et b
        
    Returns:
        Image LAB complète
    """
    # TODO: Combiner les canaux
    L = tensor_to_image(L_channel)
    ab = tensor_to_image(ab_channels)
    
    lab_image = np.concatenate([L, ab], axis=2)
    return lab_image


def calculate_psnr(img1, img2):
    """
    Calcule le PSNR entre deux images.
    
    Args:
        img1, img2: Images à comparer
        
    Returns:
        Valeur PSNR
    """
    return peak_signal_noise_ratio(img1, img2)


def calculate_ssim(img1, img2):
    """
    Calcule le SSIM entre deux images.
    
    Args:
        img1, img2: Images à comparer
        
    Returns:
        Valeur SSIM
    """
    return structural_similarity(img1, img2, multichannel=True, channel_axis=2)


def evaluate_results():
    """
    Évalue les résultats de colorisation avec les métriques PSNR et SSIM.
    """
    print("📊 Évaluation des résultats...")
    
    test_path = Path("data/test")
    results_path = Path("results/predictions")
    
    if not results_path.exists():
        print("❌ Aucun résultat trouvé dans results/predictions/")
        return
    
    psnr_scores = []
    ssim_scores = []
    
    # Parcourir les images colorisées
    for colorized_file in results_path.glob("colorized_*.jpg"):
        original_name = colorized_file.name.replace("colorized_", "")
        original_path = test_path / original_name
        
        if original_path.exists():
            # Charger les images
            original = cv2.imread(str(original_path))
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            
            colorized = cv2.imread(str(colorized_file))
            colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
            
            # Redimensionner si nécessaire
            if original.shape != colorized.shape:
                colorized = cv2.resize(colorized, (original.shape[1], original.shape[0]))
            
            # Calculer les métriques
            psnr = calculate_psnr(original, colorized)
            ssim = calculate_ssim(original, colorized)
            
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)
    
    if psnr_scores:
        print(f"PSNR moyen: {np.mean(psnr_scores):.2f} dB (±{np.std(psnr_scores):.2f})")
        print(f"SSIM moyen: {np.mean(ssim_scores):.3f} (±{np.std(ssim_scores):.3f})")
        print(f"Nombre d'images évaluées: {len(psnr_scores)}")
        
        # Sauvegarder les résultats
        results = {
            'psnr_mean': np.mean(psnr_scores),
            'psnr_std': np.std(psnr_scores),
            'ssim_mean': np.mean(ssim_scores),
            'ssim_std': np.std(ssim_scores),
            'num_images': len(psnr_scores)
        }
        
        import json
        with open('results/evaluation_metrics.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("✅ Métriques sauvegardées dans results/evaluation_metrics.json")
    else:
        print("❌ Aucune image correspondante trouvée pour l'évaluation")


def visualize_results(original, grayscale, colorized, save_path=None):
    """
    Visualise les résultats de colorisation.
    
    Args:
        original: Image originale
        grayscale: Image en niveaux de gris
        colorized: Image colorisée
        save_path: Chemin de sauvegarde (optionnel)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    axes[1].imshow(grayscale, cmap='gray')
    axes[1].set_title("Niveaux de gris")
    axes[1].axis('off')
    
    axes[2].imshow(colorized)
    axes[2].set_title("Colorisé")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def preprocess_image(image_path, target_size=(512, 512)):
    """
    Préprocesse une image pour l'inférence (CORRIGÉ).
    """
    # 1. Charger l'image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    
    # 2. Utiliser rgb_to_lab pour avoir la MÊME normalisation que l'entraînement (0 à 1)
    lab_image = rgb_to_lab(image)
    
    # 3. Extraire le canal L
    L_channel = lab_image[:, :, 0:1]  # Shape (512, 512, 1)
    
    # 4. Convertir en Tensor PyTorch
    # Transpose pour passer de (H, W, C) -> (C, H, W)
    tensor = torch.from_numpy(L_channel.transpose(2, 0, 1)).float()
    
    # Ajouter la dimension de batch: (1, 1, 512, 512)
    tensor = tensor.unsqueeze(0)
    
    return tensor


if __name__ == "__main__":
    # Tests des fonctions utilitaires
    print("Test des fonctions utilitaires...")
    
    # Test de conversion RGB <-> LAB
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    lab = rgb_to_lab(test_image)
    rgb_back = lab_to_rgb(lab)
    
    print("✅ Tests terminés !")
