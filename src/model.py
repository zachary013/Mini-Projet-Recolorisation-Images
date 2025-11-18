"""
Architecture du modèle CNN pour la recolorisation d'images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorizationCNN(nn.Module):
    """
    Modèle CNN amélioré avec skip connections pour la recolorisation.
    """
    
    def __init__(self):
        super(ColorizationCNN, self).__init__()
        
        # Encoder avec BatchNorm
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Decoder avec skip connections (dimensions corrigées)
        self.upconv1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)  # Pas de skip ici
        self.bn5 = nn.BatchNorm2d(256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)  # Pas de skip ici
        self.bn6 = nn.BatchNorm2d(128)
        
        self.upconv3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv7 = nn.Conv2d(64, 64, 3, padding=1)   # Pas de skip ici
        self.bn7 = nn.BatchNorm2d(64)
        
        self.final_conv = nn.Conv2d(64, 2, 3, padding=1)
        
    def forward(self, x):
        # Encoder avec skip connections
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.max_pool2d(x1, 2)
        
        x3 = F.relu(self.bn2(self.conv2(x2)))
        x4 = F.max_pool2d(x3, 2)
        
        x5 = F.relu(self.bn3(self.conv3(x4)))
        x6 = F.max_pool2d(x5, 2)
        
        x7 = F.relu(self.bn4(self.conv4(x6)))
        
        # Decoder - pas de pooling sur la dernière couche encoder
        x9 = F.relu(self.upconv1(x7))  # Utiliser x7 au lieu de x8
        x9 = F.relu(self.bn5(self.conv5(x9)))
        
        x10 = F.relu(self.upconv2(x9))
        x10 = F.relu(self.bn6(self.conv6(x10)))
        
        x11 = F.relu(self.upconv3(x10))
        x11 = F.relu(self.bn7(self.conv7(x11)))
        
        # Sortie finale [-1, 1] pour correspondre à la normalisation
        output = torch.tanh(self.final_conv(x11))
        
        return output


def count_parameters(model):
    """Compte le nombre de paramètres du modèle."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ColorizationCNN()
    print(f"Paramètres: {count_parameters(model):,}")
    
    x = torch.randn(1, 1, 256, 256)
    output = model(x)
    print(f"Input: {x.shape} → Output: {output.shape}")
