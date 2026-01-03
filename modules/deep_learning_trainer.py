#!/usr/bin/env python3
"""
Deep Learning Trainer Module - ResNet, ViT, U-Net Support
==========================================================

This module provides training infrastructure for deep learning models:
    - ResNet (transfer learning from ImageNet)
    - Vision Transformer (ViT) - future
    - U-Net (semantic segmentation) - future
    - Other CNN architectures

Design Philosophy:
    - Modular and extensible - easy to add new models
    - Follows same pattern as model_trainer.py for consistency
    - Framework-agnostic where possible (PyTorch primary)

Functions:
    - get_resnet_model: Create ResNet50 with pretrained weights
    - train_model: Train any deep learning model
    - evaluate_model: Evaluate on test set
    - get_predictions: Get predictions for visualization

Author: Claude Sonnet 4.5
Date: 2026-01-01
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import time
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# MODEL CREATION
# ============================================================================

def get_resnet_model(num_classes=6, pretrained=True, freeze_base=True,
                     model_type='resnet50', verbose=False):
    """
    Create ResNet model with transfer learning from ImageNet.

    Parameters
    ----------
    num_classes : int, default=6
        Number of output classes
    pretrained : bool, default=True
        Use pretrained ImageNet weights
    freeze_base : bool, default=True
        Freeze convolutional base (only train final layers)
    model_type : str, default='resnet50'
        ResNet variant: 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
    verbose : bool, default=False
        Print model information

    Returns
    -------
    model : torch.nn.Module
        ResNet model ready for training

    Notes
    -----
    - Pretrained weights from ImageNet (1000 classes)
    - Final FC layer modified for num_classes
    - If freeze_base=True, only final layer is trainable
    - First conv layer expects RGB (3 channels), will be modified for multispectral

    Examples
    --------
    >>> model = get_resnet_model(num_classes=6, pretrained=True)
    >>> print(model)
    """

    if verbose:
        print(f"\n🏗️  Creating {model_type} model:")
        print(f"   Pretrained: {pretrained}")
        print(f"   Freeze base: {freeze_base}")
        print(f"   Output classes: {num_classes}")

    # Get model architecture
    if model_type == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif model_type == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
    elif model_type == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    elif model_type == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
    elif model_type == 'resnet152':
        model = models.resnet152(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Freeze base layers if requested
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    # Modify final FC layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    if verbose:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Frozen parameters: {total_params - trainable_params:,}")

    return model


def modify_first_conv_for_multispectral(model, in_channels=23, model_type='resnet50'):
    """
    Modify first convolutional layer to accept multispectral input.

    Parameters
    ----------
    model : torch.nn.Module
        ResNet model with 3-channel input
    in_channels : int, default=23
        Number of input channels (bands + indices)
    model_type : str, default='resnet50'
        ResNet variant for layer naming

    Returns
    -------
    model : torch.nn.Module
        Modified model with multispectral input support

    Notes
    -----
    - Original ResNet expects RGB (3 channels)
    - We have 23 channels (10 bands + 13 indices)
    - Strategy: Average pretrained weights across channels or use random init

    Examples
    --------
    >>> model = get_resnet_model(num_classes=6)
    >>> model = modify_first_conv_for_multispectral(model, in_channels=23)
    """

    # Get original first conv layer
    original_conv = model.conv1

    # Create new conv layer with correct input channels
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False
    )

    # Initialize new layer
    # Strategy: Replicate pretrained weights across channels
    with torch.no_grad():
        # Get pretrained weights (out_channels, 3, kernel, kernel)
        pretrained_weight = original_conv.weight

        # Repeat weights across input channels
        # (out_channels, in_channels, kernel, kernel)
        new_weight = pretrained_weight.repeat(1, in_channels // 3 + 1, 1, 1)[:, :in_channels, :, :]

        # Normalize to maintain similar activation scale
        new_weight = new_weight * (3 / in_channels)

        # Set new weights
        new_conv.weight = nn.Parameter(new_weight)

    # Replace first conv layer
    model.conv1 = new_conv

    return model


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_loader, val_loader, num_epochs=20,
               learning_rate=0.001, device='cuda', class_weights=None,
               verbose=False):
    """
    Train deep learning model with validation.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    num_epochs : int, default=20
        Number of training epochs
    learning_rate : float, default=0.001
        Learning rate for optimizer
    device : str, default='cuda'
        Device to use ('cuda' or 'cpu')
    class_weights : torch.Tensor, optional
        Class weights for imbalanced data
    verbose : bool, default=False
        Print training progress

    Returns
    -------
    history : dict
        Training history with loss and accuracy per epoch
    best_model_state : dict
        State dict of best model (by validation accuracy)

    Examples
    --------
    >>> history, best_state = train_model(
    ...     model, train_loader, val_loader,
    ...     num_epochs=20,
    ...     learning_rate=0.001,
    ...     device='cuda'
    ... )
    """

    # Move model to device
    model = model.to(device)

    # Loss function
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_time': []
    }

    best_val_acc = 0.0
    best_model_state = None

    if verbose:
        print(f"\n🚀 Starting training:")
        print(f"   Epochs: {num_epochs}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Device: {device}")
        print(f"   Batch size: {train_loader.batch_size}")

    # Training loop
    for epoch in range(num_epochs):
        epoch_start = time.time()

        # ====================================================================
        # TRAINING PHASE
        # ====================================================================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # ====================================================================
        # VALIDATION PHASE
        # ====================================================================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Statistics
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        # Update learning rate
        scheduler.step(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        # Record history
        epoch_time = time.time() - epoch_start
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)

        if verbose:
            print(f"\n   Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s):")
            print(f"     Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"     Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            if val_acc == best_val_acc:
                print(f"     ⭐ Best model saved!")

    if verbose:
        print(f"\n✅ Training complete!")
        print(f"   Best validation accuracy: {best_val_acc:.4f}")

    return history, best_model_state


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, test_loader, device='cuda', class_names=None,
                  verbose=False):
    """
    Evaluate model on test set.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model
    test_loader : DataLoader
        Test data loader
    device : str, default='cuda'
        Device to use ('cuda' or 'cpu')
    class_names : list, optional
        List of class names for reporting
    verbose : bool, default=False
        Print evaluation results

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'accuracy': Overall accuracy
        - 'f1_macro': Macro-averaged F1 score
        - 'f1_weighted': Weighted F1 score
        - 'y_true': True labels
        - 'y_pred': Predicted labels
        - 'report': Classification report string

    Examples
    --------
    >>> results = evaluate_model(model, test_loader, device='cuda')
    >>> print(f"Test accuracy: {results['accuracy']:.4f}")
    """

    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    if verbose:
        print(f"\n📊 Evaluating model on test set...")

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Classification report
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(y_true)))]

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0
    )

    results = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'y_true': y_true,
        'y_pred': y_pred,
        'report': report
    }

    if verbose:
        print(f"\n   Accuracy: {accuracy:.4f}")
        print(f"   F1 (macro): {f1_macro:.4f}")
        print(f"   F1 (weighted): {f1_weighted:.4f}")
        print(f"\n{report}")

    return results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_model(model, filepath, metadata=None):
    """
    Save model checkpoint with metadata.

    Parameters
    ----------
    model : torch.nn.Module or dict
        Model or model state dict
    filepath : str
        Path to save checkpoint
    metadata : dict, optional
        Additional metadata to save

    Examples
    --------
    >>> save_model(model, 'models/resnet50_best.pth', metadata={'accuracy': 0.89})
    """

    checkpoint = {
        'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else model,
        'metadata': metadata or {}
    }

    torch.save(checkpoint, filepath)


def load_model(model, filepath, device='cuda'):
    """
    Load model checkpoint.

    Parameters
    ----------
    model : torch.nn.Module
        Model architecture
    filepath : str
        Path to checkpoint
    device : str, default='cuda'
        Device to load model to

    Returns
    -------
    model : torch.nn.Module
        Loaded model
    metadata : dict
        Saved metadata

    Examples
    --------
    >>> model, metadata = load_model(model, 'models/resnet50_best.pth')
    """

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    return model, checkpoint.get('metadata', {})
