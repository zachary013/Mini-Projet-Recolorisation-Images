"""
Script d'entraînement du modèle de recolorisation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from model import get_model
from utils import ImageColorizationDataset, rgb_to_lab, lab_to_rgb


def train_model(epochs=50, batch_size=16, learning_rate=0.001):
    """
    Entraîne le modèle de recolorisation.
    
    Args:
        epochs: Nombre d'époques d'entraînement
        batch_size: Taille des batches
        learning_rate: Taux d'apprentissage
    """
    
    # Configuration du device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du device: {device}")
    
    # TODO: Créer les datasets et dataloaders
    train_dataset = ImageColorizationDataset("data/train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialisation du modèle
    model = get_model("simple").to(device)
    
    # Loss function et optimizer
    criterion = nn.MSELoss()  # TODO: Tester d'autres loss functions
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Listes pour sauvegarder les métriques
    train_losses = []
    
    print(f"Début de l'entraînement pour {epochs} époques...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # Barre de progression
        pbar = tqdm(train_loader, desc=f"Époque {epoch+1}/{epochs}")
        
        for batch_idx, (gray_images, color_targets) in enumerate(pbar):
            # TODO: Implémenter la boucle d'entraînement
            gray_images = gray_images.to(device)
            color_targets = color_targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(gray_images)
            loss = criterion(predictions, color_targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        # Calcul de la loss moyenne pour l'époque
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        print(f"Époque {epoch+1}: Loss moyenne = {avg_loss:.4f}")
        
        # Sauvegarde du modèle tous les 10 époques
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss)
    
    # Sauvegarde finale
    save_model(model, "model_final.pth")
    plot_training_curves(train_losses)
    
    print("✅ Entraînement terminé !")


def save_checkpoint(model, optimizer, epoch, loss):
    """Sauvegarde un checkpoint du modèle."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    Path("results").mkdir(exist_ok=True)
    torch.save(checkpoint, f"results/checkpoint_epoch_{epoch+1}.pth")


def save_model(model, filename):
    """Sauvegarde le modèle final."""
    Path("results").mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"results/{filename}")


def load_model(model_path):
    """Charge un modèle sauvegardé."""
    model = get_model("simple")
    model.load_state_dict(torch.load(model_path))
    return model


def plot_training_curves(train_losses):
    """Affiche les courbes d'entraînement."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Époque')
    plt.ylabel('Loss')
    plt.title('Courbe d\'entraînement')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/training_curve.png')
    plt.show()


def validate_model(model, val_loader, device):
    """
    Évalue le modèle sur l'ensemble de validation.
    
    Args:
        model: Modèle à évaluer
        val_loader: DataLoader de validation
        device: Device (CPU/GPU)
        
    Returns:
        Loss moyenne de validation
    """
    model.eval()
    val_loss = 0.0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for gray_images, color_targets in val_loader:
            gray_images = gray_images.to(device)
            color_targets = color_targets.to(device)
            
            predictions = model(gray_images)
            loss = criterion(predictions, color_targets)
            val_loss += loss.item()
    
    return val_loss / len(val_loader)


if __name__ == "__main__":
    # Lancement de l'entraînement
    train_model(epochs=50)
