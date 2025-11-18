"""
Architecture du modèle CNN pour la recolorisation d'images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorizationCNN(nn.Module):
    """
    Modèle CNN pour la recolorisation d'images.
    Prend en entrée une image en niveaux de gris (1 canal) 
    et prédit les canaux de couleur (2 canaux ab dans l'espace LAB).
    """
    
    def __init__(self):
        super(ColorizationCNN, self).__init__()
        
        # Encoder (downsampling)
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        
        # Decoder (upsampling)
        self.upconv1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(64, 2, 2, stride=2)
        
    def forward(self, x):
        # Encoder
        x1 = F.relu(self.conv1(x))
        x2 = F.max_pool2d(x1, 2)
        x3 = F.relu(self.conv2(x2))
        x4 = F.max_pool2d(x3, 2)
        x5 = F.relu(self.conv3(x4))
        x6 = F.max_pool2d(x5, 2)
        
        # Decoder
        x7 = F.relu(self.upconv1(x6))
        x8 = F.relu(self.upconv2(x7))
        x9 = torch.tanh(self.upconv3(x8))
        
        return x9


def count_parameters(model):
    """Compte le nombre de paramètres du modèle."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test du modèle
    model = ColorizationCNN()
    print(f"Nombre de paramètres: {count_parameters(model):,}")
    
    # Test avec une image factice
    x = torch.randn(1, 1, 256, 256)  # Batch de 1, 1 canal, 256x256
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
