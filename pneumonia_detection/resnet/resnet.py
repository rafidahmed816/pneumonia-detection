import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Binary(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet18Binary, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Linear(
            self.resnet.fc.in_features, 1
        ) 

    def forward(self, x):
        return torch.sigmoid(self.resnet(x))