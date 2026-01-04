#!/usr/bin/env python3
"""
Model Factory - Create Any Model with Proper Adaptations
==========================================================

Factory pattern for creating deep learning models with automatic:
- Input channel adaptation (for 23-channel multispectral input)
- Output class adaptation (for 6 land cover classes)
- Pretrained weight loading (ImageNet → fine-tune)
- Device placement (CPU/GPU)

Supports:
- ResNet (18, 34, 50, 101, 152)
- EfficientNet (B0, B1, B3)
- ConvNeXt (Tiny, Small)
- DenseNet (121, 169)
- Inception (V3)

Author: Claude Sonnet 4.5
Date: 2026-01-04
"""

import torch
import torch.nn as nn
from torchvision import models
import timm  # PyTorch Image Models - for ConvNeXt, EfficientNet

# Handle imports for both direct execution and module import
try:
    from modules.model_registry import get_model_info, MODEL_REGISTRY
except ModuleNotFoundError:
    from model_registry import get_model_info, MODEL_REGISTRY

# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(model_name, num_classes=6, input_channels=23,
                pretrained=True, device='cuda'):
    """
    Create any registered model with proper input/output adaptation.

    Args:
        model_name: Name from MODEL_REGISTRY (e.g., 'resnet50', 'efficientnet_b3')
        num_classes: Number of output classes (default: 6 for land cover)
        input_channels: Number of input channels (default: 23 for multispectral)
        pretrained: Use ImageNet pretrained weights (default: True)
        device: 'cuda' or 'cpu'

    Returns:
        model: PyTorch model ready for training
        info: Model metadata from registry
    """

    # Validate model exists
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not in registry. "
                        f"Available: {list(MODEL_REGISTRY.keys())}")

    info = get_model_info(model_name)
    family = info['family']

    print(f"\n🏗️  Creating {info['display_name']}...")
    print(f"   Family: {family}")
    print(f"   Pretrained: {'Yes (ImageNet)' if pretrained else 'No'}")
    print(f"   Input: {input_channels} channels → {num_classes} classes")

    # Create model based on family
    if family == 'resnet':
        model = _create_resnet(model_name, num_classes, input_channels, pretrained)
    elif family == 'efficientnet':
        model = _create_efficientnet(model_name, num_classes, input_channels, pretrained)
    elif family == 'convnext':
        model = _create_convnext(model_name, num_classes, input_channels, pretrained)
    elif family == 'densenet':
        model = _create_densenet(model_name, num_classes, input_channels, pretrained)
    elif family == 'inception':
        model = _create_inception(model_name, num_classes, input_channels, pretrained)
    else:
        raise NotImplementedError(f"Family '{family}' not implemented yet")

    # Move to device
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   ✓ Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"   ✓ Trainable: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"   ✓ Device: {device}")

    return model, info


# ============================================================================
# FAMILY-SPECIFIC CREATION FUNCTIONS
# ============================================================================

def _create_resnet(model_name, num_classes, input_channels, pretrained):
    """Create ResNet family models."""

    # Map model names to torchvision functions
    resnet_models = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
    }

    if model_name not in resnet_models:
        raise ValueError(f"Unknown ResNet variant: {model_name}")

    # Create base model
    if pretrained:
        model = resnet_models[model_name](weights='IMAGENET1K_V1')
    else:
        model = resnet_models[model_name](weights=None)

    # Adapt first conv layer for multispectral input
    if input_channels != 3:
        original_conv = model.conv1
        model.conv1 = nn.Conv2d(
            input_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize new channels randomly, copy RGB weights if pretrained
        if pretrained:
            with torch.no_grad():
                # Copy first 3 channels from pretrained weights
                model.conv1.weight[:, :3, :, :] = original_conv.weight
                # Initialize remaining channels randomly
                nn.init.kaiming_normal_(model.conv1.weight[:, 3:, :, :],
                                       mode='fan_out', nonlinearity='relu')

    # Adapt final FC layer for num_classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def _create_efficientnet(model_name, num_classes, input_channels, pretrained):
    """Create EfficientNet family models using timm library."""

    # Map model names to timm names
    efficientnet_timm_names = {
        'efficientnet_b0': 'efficientnet_b0',
        'efficientnet_b1': 'efficientnet_b1',
        'efficientnet_b3': 'efficientnet_b3',
    }

    if model_name not in efficientnet_timm_names:
        raise ValueError(f"Unknown EfficientNet variant: {model_name}")

    timm_name = efficientnet_timm_names[model_name]

    # Create model using timm
    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        in_chans=input_channels,  # timm handles multispectral input!
        num_classes=num_classes
    )

    return model


def _create_convnext(model_name, num_classes, input_channels, pretrained):
    """Create ConvNeXt family models using timm library."""

    # Map model names to timm names
    convnext_timm_names = {
        'convnext_tiny': 'convnext_tiny',
        'convnext_small': 'convnext_small',
    }

    if model_name not in convnext_timm_names:
        raise ValueError(f"Unknown ConvNeXt variant: {model_name}")

    timm_name = convnext_timm_names[model_name]

    # Create model using timm
    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        in_chans=input_channels,  # timm handles multispectral input!
        num_classes=num_classes
    )

    return model


def _create_densenet(model_name, num_classes, input_channels, pretrained):
    """Create DenseNet family models."""

    # Map model names to torchvision functions
    densenet_models = {
        'densenet121': models.densenet121,
        'densenet169': models.densenet169,
    }

    if model_name not in densenet_models:
        raise ValueError(f"Unknown DenseNet variant: {model_name}")

    # Create base model
    if pretrained:
        model = densenet_models[model_name](weights='IMAGENET1K_V1')
    else:
        model = densenet_models[model_name](weights=None)

    # Adapt first conv layer for multispectral input
    if input_channels != 3:
        original_conv = model.features.conv0
        model.features.conv0 = nn.Conv2d(
            input_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize new channels
        if pretrained:
            with torch.no_grad():
                # Copy first 3 channels from pretrained weights
                model.features.conv0.weight[:, :3, :, :] = original_conv.weight
                # Initialize remaining channels randomly
                nn.init.kaiming_normal_(model.features.conv0.weight[:, 3:, :, :],
                                       mode='fan_out', nonlinearity='relu')

    # Adapt final FC layer for num_classes
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    return model


def _create_inception(model_name, num_classes, input_channels, pretrained):
    """Create Inception family models."""

    if model_name != 'inception_v3':
        raise ValueError(f"Unknown Inception variant: {model_name}")

    # Create base model
    if pretrained:
        model = models.inception_v3(weights='IMAGENET1K_V1', aux_logits=True)
    else:
        model = models.inception_v3(weights=None, aux_logits=True)

    # Adapt first conv layer for multispectral input
    if input_channels != 3:
        original_conv = model.Conv2d_1a_3x3.conv
        model.Conv2d_1a_3x3.conv = nn.Conv2d(
            input_channels, 32,
            kernel_size=3, stride=2, bias=False
        )

        # Initialize new channels
        if pretrained:
            with torch.no_grad():
                # Copy first 3 channels from pretrained weights
                model.Conv2d_1a_3x3.conv.weight[:, :3, :, :] = original_conv.weight
                # Initialize remaining channels randomly
                nn.init.kaiming_normal_(model.Conv2d_1a_3x3.conv.weight[:, 3:, :, :],
                                       mode='fan_out', nonlinearity='relu')

    # Adapt final FC layer for num_classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # Adapt auxiliary classifier
    if model.aux_logits:
        num_ftrs_aux = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs_aux, num_classes)

    return model


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_model_summary(model, input_size=(23, 32, 32)):
    """Get summary of model architecture."""
    from torchinfo import summary

    return summary(
        model,
        input_size=(1, *input_size),  # Batch size 1
        col_names=["output_size", "num_params", "trainable"],
        depth=3,
        verbose=0
    )


def test_model_creation():
    """Test creating all registered models."""
    print("\n" + "="*80)
    print("TESTING MODEL CREATION FOR ALL REGISTERED MODELS")
    print("="*80)

    test_models = [
        'resnet50',
        'efficientnet_b3',
        'convnext_tiny',
        'densenet121',
        'inception_v3'
    ]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for model_name in test_models:
        try:
            model, info = create_model(
                model_name,
                num_classes=6,
                input_channels=23,
                pretrained=False,  # Don't download weights for testing
                device=device
            )

            # Test forward pass
            # Inception-V3 requires min 75x75 input, others can use 32x32
            if model_name == 'inception_v3':
                dummy_input = torch.randn(2, 23, 299, 299).to(device)
                model.eval()  # Disable aux_logits in eval mode
            else:
                dummy_input = torch.randn(2, 23, 32, 32).to(device)

            with torch.no_grad():
                output = model(dummy_input)

            print(f"   ✓ Forward pass: {dummy_input.shape} → {output.shape}")
            print(f"   ✓ Model creation SUCCESSFUL!\n")

        except Exception as e:
            print(f"   ✗ ERROR: {str(e)}\n")

    print("="*80)


if __name__ == '__main__':
    """Demo: Test model creation."""
    test_model_creation()
