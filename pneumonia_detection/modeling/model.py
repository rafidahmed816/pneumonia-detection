import torch
import torch.nn as nn
from torchvision import models

class PneumoniaModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Use pretrained ResNet18
        self.model = models.resnet18(pretrained=True)
        
        # Replace last layer for binary classification
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2) 
        )
    
    def forward(self, x):
        return self.model(x)