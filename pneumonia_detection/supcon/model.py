import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional


class SupConLoss(nn.Module):
    """Improved SupCon loss with better numerical stability"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...]")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # Compute similarity matrix with improved numerical stability
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # More stable numerical computation
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Create masks
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # Compute probabilities with numerical stability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Handle case where mask.sum(1) might be 0
        mask_sum = mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=1e-8)  # Avoid division by zero

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # Loss with gradient clipping
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        # Clip extreme values
        loss = torch.clamp(loss, max=10.0)

        return loss


class SupConModel(nn.Module):
    """Improved SupCon model with better architecture and training stability"""

    def __init__(
        self, backbone="resnet18", feat_dim=128, num_classes=1, dropout_rate=0.3
    ):
        super(SupConModel, self).__init__()
        self.backbone_name = backbone
        self.feat_dim = feat_dim
        self.dropout_rate = dropout_rate

        # Load pre-trained backbone with better initialization
        if backbone == "resnet18":
            backbone_model = models.resnet18(weights="IMAGENET1K_V1")
            dim_in = backbone_model.fc.in_features
            backbone_model.fc = nn.Identity()
        elif backbone == "resnet50":
            backbone_model = models.resnet50(weights="IMAGENET1K_V1")
            dim_in = backbone_model.fc.in_features
            backbone_model.fc = nn.Identity()
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented")

        self.encoder = backbone_model

        # Simpler, more stable projection head
        self.projection_head = nn.Sequential(
            nn.Linear(dim_in, dim_in // 2),
            nn.BatchNorm1d(dim_in // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(dim_in // 2, feat_dim),
        )

        # Simpler classification head
        self.classification_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(feat_dim // 2, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Better weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mode="both"):
        features = self.encoder(x)
        projected_features = self.projection_head(features)

        if mode == "contrastive":
            return F.normalize(projected_features, dim=1)
        elif mode == "classify":
            logits = self.classification_head(projected_features)
            return logits
        elif mode == "both":
            normalized_features = F.normalize(projected_features, dim=1)
            logits = self.classification_head(projected_features)
            return normalized_features, torch.sigmoid(logits)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def get_features(self, x):
        features = self.encoder(x)
        projected_features = self.projection_head(features)
        return F.normalize(projected_features, dim=1)

    def classify(self, x):
        features = self.encoder(x)
        projected_features = self.projection_head(features)
        logits = self.classification_head(projected_features)
        return logits
