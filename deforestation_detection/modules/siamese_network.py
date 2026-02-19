"""
Siamese Network Module
======================

Siamese CNN architecture for bi-temporal change detection.
Uses shared ResNet-50 backbone with feature fusion for binary
change/no-change classification.

Reference:
    Daudt, R.C., et al. (2018). Fully Convolutional Siamese Networks
    for Change Detection. ICIP 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset
import numpy as np


class SiameseResNet(nn.Module):
    """
    Siamese network with shared ResNet-50 backbone for change detection.

    Architecture:
        - Shared ResNet-50 backbone (modified for multi-channel input)
        - Feature extraction from both time periods
        - Fusion: [f1; f2; |f1-f2|] (concatenation + absolute difference)
        - Classification head: FC -> BN -> ReLU -> Dropout -> FC -> 2

    Input: Two patches (C, H, W) from time 1 and time 2
    Output: Binary prediction (change / no-change)
    """

    def __init__(self, in_channels=23, num_classes=2, pretrained=True,
                 backbone='resnet50', dropout=0.5):
        """
        Args:
            in_channels: Number of input channels per time period
            num_classes: Number of output classes (2: change/no-change)
            pretrained: Use ImageNet pretrained weights
            backbone: ResNet variant ('resnet50' or 'resnet34')
            dropout: Dropout rate in classification head
        """
        super(SiameseResNet, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Load backbone
        if backbone == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_model = models.resnet50(weights=weights)
            feature_dim = 2048
        elif backbone == 'resnet34':
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            base_model = models.resnet34(weights=weights)
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Modify first conv layer for multi-channel input
        original_conv = base_model.conv1
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize: replicate pretrained weights across channels
        if pretrained:
            with torch.no_grad():
                # Average pretrained RGB weights, then tile to match in_channels
                avg_weight = original_conv.weight.mean(dim=1, keepdim=True)
                self.conv1.weight.copy_(avg_weight.repeat(1, in_channels, 1, 1))

        # Shared backbone (everything except fc layer)
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        # Feature dimension after fusion: f1 + f2 + |f1-f2| = 3 * feature_dim
        fusion_dim = 3 * feature_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def extract_features(self, x):
        """Extract features from a single branch."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x1, x2):
        """
        Forward pass with two input patches.

        Args:
            x1: (B, C, H, W) patch at time 1
            x2: (B, C, H, W) patch at time 2

        Returns:
            (B, num_classes) logits
        """
        # Shared feature extraction
        f1 = self.extract_features(x1)
        f2 = self.extract_features(x2)

        # Feature fusion: concatenation + absolute difference
        diff = torch.abs(f1 - f2)
        fused = torch.cat([f1, f2, diff], dim=1)

        # Classification
        out = self.classifier(fused)
        return out


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in change detection.

    Reduces loss for well-classified examples, focusing training on
    hard negatives. Particularly useful when no-change >> change.

    Reference:
        Lin, T.Y., et al. (2017). Focal Loss for Dense Object Detection.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Class weights (tensor or None for uniform)
            gamma: Focusing parameter (0 = CE, 2 = default focal)
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class SiameseDataset(Dataset):
    """
    PyTorch Dataset for bi-temporal patch pairs.

    Provides synchronized augmentation for both time periods
    to maintain spatial correspondence.
    """

    def __init__(self, patches_t1, patches_t2, labels, augment=False):
        """
        Args:
            patches_t1: (N, C, H, W) patches at time 1
            patches_t2: (N, C, H, W) patches at time 2
            labels: (N,) binary labels
            augment: Apply synchronized augmentation
        """
        self.patches_t1 = torch.FloatTensor(patches_t1)
        self.patches_t2 = torch.FloatTensor(patches_t2)
        self.labels = torch.LongTensor(labels)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        p1 = self.patches_t1[idx]
        p2 = self.patches_t2[idx]
        label = self.labels[idx]

        if self.augment:
            p1, p2 = self._synchronized_augment(p1, p2)

        return p1, p2, label

    def _synchronized_augment(self, p1, p2):
        """Apply same random augmentation to both patches."""
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            p1 = torch.flip(p1, [2])
            p2 = torch.flip(p2, [2])

        # Random vertical flip
        if torch.rand(1).item() > 0.5:
            p1 = torch.flip(p1, [1])
            p2 = torch.flip(p2, [1])

        # Random 90-degree rotation
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            p1 = torch.rot90(p1, k, [1, 2])
            p2 = torch.rot90(p2, k, [1, 2])

        return p1, p2


def get_siamese_model(in_channels=23, backbone='resnet50', pretrained=True,
                      dropout=0.5, device='cuda'):
    """
    Create and return a Siamese network model.

    Args:
        in_channels: Number of input channels
        backbone: ResNet variant
        pretrained: Use pretrained weights
        dropout: Dropout rate
        device: Device to place model on

    Returns:
        SiameseResNet model on specified device
    """
    model = SiameseResNet(
        in_channels=in_channels,
        num_classes=2,
        pretrained=pretrained,
        backbone=backbone,
        dropout=dropout
    )

    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'
        print("CUDA not available, using CPU")

    model = model.to(device)
    return model


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
