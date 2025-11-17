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
        
        # TODO: Définir l'architecture du réseau
        # Encoder (extraction de features)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Decoder (reconstruction couleur)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 2, kernel_size=3, padding=1)  # 2 canaux ab
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
    
    def forward(self, x):
        """
        Forward pass du modèle.
        
        Args:
            x: Image en niveaux de gris (batch_size, 1, H, W)
            
        Returns:
            Prédiction des canaux couleur (batch_size, 2, H, W)
        """
        # TODO: Implémenter le forward pass
        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Decoder
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.tanh(self.conv4(x))  # Tanh pour valeurs entre -1 et 1
        
        return x


class UNetColorization(nn.Module):
    """
    Architecture U-Net simplifiée pour la recolorisation.
    """
    
    def __init__(self):
        super(UNetColorization, self).__init__()
        
        # TODO: Implémenter une architecture U-Net
        # Encoder
        self.enc1 = self._conv_block(1, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        
        # Decoder
        self.dec3 = self._conv_block(256, 128)
        self.dec2 = self._conv_block(128, 64)
        self.dec1 = nn.Conv2d(64, 2, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def _conv_block(self, in_channels, out_channels):
        """Bloc de convolution avec BatchNorm et ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """Forward pass U-Net."""
        # TODO: Implémenter le forward avec skip connections
        pass


def get_model(model_type="simple"):
    """
    Factory function pour créer le modèle.
    
    Args:
        model_type: Type de modèle ("simple" ou "unet")
        
    Returns:
        Modèle PyTorch
    """
    if model_type == "simple":
        return ColorizationCNN()
    elif model_type == "unet":
        return UNetColorization()
    else:
        raise ValueError(f"Type de modèle non supporté: {model_type}")


def count_parameters(model):
    """Compte le nombre de paramètres du modèle."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test du modèle
    model = get_model("simple")
    print(f"Nombre de paramètres: {count_parameters(model):,}")
    
    # Test avec une image factice
    x = torch.randn(1, 1, 256, 256)  # Batch de 1, 1 canal, 256x256
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
