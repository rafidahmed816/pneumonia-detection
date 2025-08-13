import torch.nn as nn
from torchvision import models

class PneumoniaModel(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_feats = m.fc.in_features
        m.fc = nn.Linear(in_feats, 2)
        self.backbone = m

    def forward(self, x):
        return self.backbone(x)
