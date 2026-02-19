"""
Model Architectures for Scene Classification

Provides a unified interface to create pretrained models with modified
classification heads for any number of output classes.
"""

import torch
import torch.nn as nn
from torchvision import models


def create_model(model_name, num_classes, pretrained=True):
    """
    Create a classification model with pretrained weights.

    Args:
        model_name: One of the keys in config.MODELS
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights

    Returns:
        model: nn.Module with modified classifier head
    """
    weights = 'IMAGENET1K_V1' if pretrained else None

    if model_name == 'resnet50':
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'resnet101':
        model = models.resnet101(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'densenet121':
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes)

    elif model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(weights=weights)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes)

    elif model_name == 'vit_b_16':
        model = models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(
            model.heads.head.in_features, num_classes)

    elif model_name == 'swin_t':
        model = models.swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, num_classes)

    elif model_name == 'convnext_tiny':
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(
            model.classifier[2].in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_info(model_name, num_classes):
    """Get model metadata without creating full model."""
    model = create_model(model_name, num_classes, pretrained=False)
    total, trainable = count_parameters(model)
    del model
    return {
        'name': model_name,
        'total_params': total,
        'trainable_params': trainable,
        'params_m': total / 1e6,
    }
