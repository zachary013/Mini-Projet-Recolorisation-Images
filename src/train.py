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
    
    # Création des datasets avec split train/validation
    full_dataset = ImageColorizationDataset('data/train')
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset: {train_size} train, {val_size} validation")
    
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
    val_losses = []
    
    # Boucle d'entraînement
    for epoch in range(epochs):
        # Phase d'entraînement
        model.train()
        epoch_train_loss = 0.0
        
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
            
            epoch_train_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Phase de validation
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for L_channel, ab_channels in val_loader:
                L_channel = L_channel.to(device)
                ab_channels = ab_channels.to(device)
                
                predicted_ab = model(L_channel)
                loss = combined_loss(predicted_ab, ab_channels)
                epoch_val_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Sauvegarde du modèle tous les 10 époques
        if (epoch + 1) % 10 == 0:
            save_model(model, f'model_epoch_{epoch+1}.pth')
    
    # Sauvegarde finale
    save_model(model, 'colorization_model_final.pth')
    
    # Graphique des pertes
    plot_training_loss(train_losses, val_losses)
    
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


def plot_training_loss(train_losses, val_losses=None):
    """Affiche le graphique des pertes d'entraînement."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    
    if val_losses:
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.legend()
    
    plt.title('Évolution de la perte pendant l\'entraînement')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.grid(True)
    
    # Sauvegarde du graphique
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / 'training_loss.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    train_model(epochs=20)  # Test avec 20 époques
