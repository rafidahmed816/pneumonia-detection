# pneumonia_detection/resnet.py
import torch
import torch.nn as nn
from torchvision import models

class ResNet18Binary(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        # Handle torchvision API across versions
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
        except Exception:
            backbone = models.resnet18(pretrained=pretrained)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )
        self.backbone = backbone

    def forward(self, x):
        # sigmoid to match current BCELoss trainer
        return torch.sigmoid(self.backbone(x))
