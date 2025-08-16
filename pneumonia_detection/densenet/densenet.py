import torch
import torch.nn as nn
import torchvision.models as models

class DenseNet121Binary(nn.Module):
    def __init__(self, pretrained=False):
        super(DenseNet121Binary, self).__init__()
        # Load pre-trained DenseNet121 model
        self.densenet = models.densenet121(pretrained=pretrained)
        
        # Modify the classifier (final fully connected layer) for binary classification
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, 1)  # Binary classification (PNEUMONIA or NORMAL)
    
    def forward(self, x):
        return torch.sigmoid(self.densenet(x))  # Sigmoid for binary classification
