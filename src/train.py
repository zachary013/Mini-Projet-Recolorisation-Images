"""
Script d'entraînement pour le modèle de recolorisation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from pathlib import Path
import matplotlib.pyplot as plt

from .model import ColorizationCNN
from .utils import ImageColorizationDataset


def train_model(epochs=50, batch_size=16, learning_rate=0.001):
    """
    Entraîne le modèle de recolorisation.
    
    Args:
        epochs: Nombre d'époques d'entraînement
        batch_size: Taille des batches
        learning_rate: Taux d'apprentissage
    """
    print(f"🚀 Début de l'entraînement - {epochs} époques")
    
    # Configuration du device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device utilisé: {device}")
    
    # Création des datasets
    train_dataset = ImageColorizationDataset('data/train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialisation du modèle avec loss améliorée
    model = ColorizationCNN().to(device)
    
    # Loss combinée : MSE + L1 pour éviter les couleurs délavées
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    
    def combined_loss(pred, target):
        return 0.7 * mse_loss(pred, target) + 0.3 * l1_loss(pred, target)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Listes pour stocker les pertes
    train_losses = []
    
    # Boucle d'entraînement
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (L_channel, ab_channels) in enumerate(train_loader):
            L_channel = L_channel.to(device)
            ab_channels = ab_channels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predicted_ab = model(L_channel)
            
            # Calcul de la perte améliorée
            loss = combined_loss(predicted_ab, ab_channels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        print(f'Epoch {epoch+1}/{epochs} - Perte moyenne: {avg_loss:.4f}')
        
        # Sauvegarde du modèle tous les 10 époques
        if (epoch + 1) % 10 == 0:
            save_model(model, f'model_epoch_{epoch+1}.pth')
    
    # Sauvegarde finale
    save_model(model, 'colorization_model_final.pth')
    
    # Graphique des pertes
    plot_training_loss(train_losses)
    
    print("✅ Entraînement terminé !")


def save_model(model, filename):
    """Sauvegarde le modèle."""
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    torch.save(model.state_dict(), models_dir / filename)
    print(f"💾 Modèle sauvegardé: {filename}")


def load_model(filename, device='cpu'):
    """Charge un modèle sauvegardé."""
    model = ColorizationCNN()
    model.load_state_dict(torch.load(f'models/{filename}', map_location=device))
    model.to(device)
    return model


def plot_training_loss(losses):
    """Affiche le graphique des pertes d'entraînement."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Évolution de la perte pendant l\'entraînement')
    plt.xlabel('Époque')
    plt.ylabel('Perte MSE')
    plt.grid(True)
    
    # Sauvegarde du graphique
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / 'training_loss.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    train_model(epochs=20)  # Test avec 20 époques
