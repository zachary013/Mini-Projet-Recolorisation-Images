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
        # TODO: Implémenter le chargement des images
        image_path = self.image_paths[idx]
        
        # Charger l'image couleur
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Redimensionner (modifiable selon les ressources)
        target_size = (256, 256)  # Changez en (512, 512) si GPU puissant
        image = cv2.resize(image, target_size)
        
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
    Convertit une image RGB en espace colorimétrique LAB.
    
    Args:
        image: Image RGB (numpy array)
        
    Returns:
        Image LAB normalisée
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Normalisation cohérente [-1, 1] pour correspondre à tanh
    lab = lab.astype(np.float32)
    lab[:, :, 0] = (lab[:, :, 0] / 50.0) - 1.0  # L: 0-100 -> [-1, 1]
    lab[:, :, 1] = lab[:, :, 1] / 128.0  # a: -128-127 -> [-1, 1]
    lab[:, :, 2] = lab[:, :, 2] / 128.0  # b: -128-127 -> [-1, 1]
    
    return lab


def lab_to_rgb(lab_image):
    """
    Convertit une image LAB en RGB.
    
    Args:
        lab_image: Image LAB normalisée [-1, 1]
        
    Returns:
        Image RGB
    """
    lab = lab_image.copy()
    
    # Dénormalisation cohérente avec la normalisation
    lab[:, :, 0] = (lab[:, :, 0] + 1.0) * 50.0  # [-1, 1] -> [0, 100]
    lab[:, :, 1] = lab[:, :, 1] * 128.0  # [-1, 1] -> [-128, 127]
    lab[:, :, 2] = lab[:, :, 2] * 128.0  # [-1, 1] -> [-128, 127]
    
    # Clipping pour éviter les valeurs hors limites
    lab = np.clip(lab, [0, -128, -128], [100, 127, 127])
    lab = lab.astype(np.uint8)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return rgb


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


def preprocess_image(image_path, target_size=(256, 256)):
    """
    Préprocesse une image pour l'inférence.
    
    Args:
        image_path: Chemin vers l'image
        target_size: Taille cible
        
    Returns:
        Image préprocessée en tensor
    """
    # TODO: Charger et préprocesser l'image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    
    # Convertir en niveaux de gris pour l'entrée du modèle
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = gray.reshape(1, 1, *target_size)  # Ajouter dimensions batch et canal
    
    return torch.from_numpy(gray).float()


if __name__ == "__main__":
    # Tests des fonctions utilitaires
    print("Test des fonctions utilitaires...")
    
    # Test de conversion RGB <-> LAB
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    lab = rgb_to_lab(test_image)
    rgb_back = lab_to_rgb(lab)
    
    print("✅ Tests terminés !")
