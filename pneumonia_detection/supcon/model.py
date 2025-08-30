import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        
    def forward(self, features, labels=None, mask=None):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class SupConResNet(nn.Module):
    """Backbone + projection head for supervised contrastive learning"""
    
    def __init__(self, name='resnet18', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        model_fun = getattr(models, name)
        self.encoder = model_fun(pretrained=True)
        
        # Remove the final classification layer
        if hasattr(self.encoder, 'fc'):
            dim_in = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        elif hasattr(self.encoder, 'classifier'):
            dim_in = self.encoder.classifier.in_features
            self.encoder.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown architecture: {name}")
        
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(f'head not supported: {head}')
    
    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat


class SupConClassifier(nn.Module):
    """Linear classifier for supervised contrastive learning"""
    
    def __init__(self, name='resnet18', num_classes=2, feat_dim=128):
        super(SupConClassifier, self).__init__()
        model_fun = getattr(models, name)
        self.encoder = model_fun(pretrained=True)
        
        # Remove the final classification layer and get feature dimension
        if hasattr(self.encoder, 'fc'):
            dim_in = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        elif hasattr(self.encoder, 'classifier'):
            dim_in = self.encoder.classifier.in_features
            self.encoder.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown architecture: {name}")
        
        # Projection head (same as encoder)
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim)
        )
        
        # Classification head
        self.classifier = nn.Linear(feat_dim, num_classes)
    
    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        logits = self.classifier(feat)
        return torch.sigmoid(logits)


class PneumoniaSupConModel(nn.Module):
    def __init__(self, backbone='resnet18', feat_dim=128, num_classes=1):
        super(PneumoniaSupConModel, self).__init__()
        self.backbone_name = backbone
        self.feat_dim = feat_dim
        
        # Load pre-trained backbone
        if backbone == 'resnet18':
            backbone_model = models.resnet18(pretrained=True)
            # Modify first conv layer to handle grayscale converted to RGB
            backbone_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            dim_in = backbone_model.fc.in_features
            backbone_model.fc = nn.Identity()
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented")
        
        self.encoder = backbone_model
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim)
        )
        
        # Classification head
        self.classification_head = nn.Linear(feat_dim, num_classes)
        
    def forward(self, x, mode='both'):
        """
        Forward pass with different modes:
        - 'contrastive': Return normalized features for contrastive loss
        - 'classify': Return classification logits
        - 'both': Return both features and logits
        """
        # Extract features from backbone
        features = self.encoder(x)
        
        # Project features
        projected_features = self.projection_head(features)
        
        if mode == 'contrastive':
            # Normalize features for contrastive learning
            return F.normalize(projected_features, dim=1)
        elif mode == 'classify':
            # Classification logits
            logits = self.classification_head(projected_features)
            return torch.sigmoid(logits)
        elif mode == 'both':
            # Both normalized features and classification logits
            normalized_features = F.normalize(projected_features, dim=1)
            logits = self.classification_head(projected_features)
            return normalized_features, torch.sigmoid(logits)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def get_features(self, x):
        """Extract normalized features for evaluation"""
        features = self.encoder(x)
        projected_features = self.projection_head(features)
        return F.normalize(projected_features, dim=1)
    
    def classify(self, x):
        """Direct classification without contrastive features"""
        features = self.encoder(x)
        projected_features = self.projection_head(features)
        logits = self.classification_head(projected_features)
        return torch.sigmoid(logits)