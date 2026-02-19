"""
Deep Learning Trainer Module
=============================

Training loops for ResNet (PCC) and Siamese CNN change detection.
Handles training, validation, checkpointing, and evaluation.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    precision_score, recall_score, confusion_matrix,
    classification_report
)

from .siamese_network import SiameseResNet, FocalLoss, SiameseDataset


def train_resnet_classifier(model, train_loader, val_loader, num_epochs=50,
                             lr=1e-4, weight_decay=1e-5, patience=10,
                             save_dir='results/models/pcc_resnet101',
                             device='cuda', verbose=True):
    """
    Train ResNet for per-year land cover classification (PCC approach).

    Args:
        model: ResNet model (from torchvision, modified for multi-channel)
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        num_epochs: Maximum training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        patience: Early stopping patience
        save_dir: Directory to save model and history
        device: Training device
        verbose: Print progress

    Returns:
        dict with training history and best metrics
    """
    os.makedirs(save_dir, exist_ok=True)

    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'
        if verbose:
            print("CUDA not available, using CPU")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=verbose
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
    }

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    if verbose:
        print("\n" + "=" * 60)
        print("RESNET CLASSIFIER TRAINING (PCC)")
        print("=" * 60)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        else:
            epochs_no_improve += 1

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"\n  Early stopping at epoch {epoch + 1} "
                      f"(best epoch: {best_epoch + 1})")
            break

    # Save training history
    np.savez(os.path.join(save_dir, 'training_history.npz'), **history)

    if verbose:
        print(f"\n  Best validation loss: {best_val_loss:.4f} (epoch {best_epoch + 1})")
        print(f"  Model saved to: {save_dir}")

    return history


def train_siamese_model(model, train_loader, val_loader, num_epochs=50,
                         lr=1e-4, weight_decay=1e-5, patience=10,
                         focal_gamma=2.0, save_dir='results/models/siamese_resnet50',
                         device='cuda', verbose=True):
    """
    Train Siamese network for change detection.

    Args:
        model: SiameseResNet model
        train_loader: Training DataLoader (yields t1, t2, label)
        val_loader: Validation DataLoader
        num_epochs: Maximum training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        patience: Early stopping patience
        focal_gamma: Focal loss gamma parameter
        save_dir: Directory to save model and history
        device: Training device
        verbose: Print progress

    Returns:
        dict with training history and best metrics
    """
    os.makedirs(save_dir, exist_ok=True)

    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'
        if verbose:
            print("CUDA not available, using CPU")

    model = model.to(device)

    # Focal loss for class imbalance
    criterion = FocalLoss(gamma=focal_gamma)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=verbose
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [],
    }

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    if verbose:
        print("\n" + "=" * 60)
        print("SIAMESE NETWORK TRAINING")
        print("=" * 60)
        print(f"  Device: {device}")
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {params:,}")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_t1, batch_t2, batch_y in train_loader:
            batch_t1 = batch_t1.to(device)
            batch_t2 = batch_t2.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_t1, batch_t2)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_t1.size(0)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

        train_loss /= len(all_labels)
        train_acc = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch_t1, batch_t2, batch_y in val_loader:
                batch_t1 = batch_t1.to(device)
                batch_t2 = batch_t2.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_t1, batch_t2)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_t1.size(0)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())

        val_loss /= len(val_labels)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
        else:
            epochs_no_improve += 1

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1:3d}/{num_epochs} | "
                  f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"Acc: {train_acc:.4f}/{val_acc:.4f} | "
                  f"F1: {train_f1:.4f}/{val_f1:.4f}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"\n  Early stopping at epoch {epoch + 1} "
                      f"(best: {best_epoch + 1})")
            break

    np.savez(os.path.join(save_dir, 'training_history.npz'), **history)

    if verbose:
        print(f"\n  Best val loss: {best_val_loss:.4f} (epoch {best_epoch + 1})")

    return history


def evaluate_model(model, test_loader, device='cuda', is_siamese=False, verbose=True):
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained PyTorch model
        test_loader: Test DataLoader
        device: Device
        is_siamese: Whether model is Siamese (expects paired inputs)
        verbose: Print results

    Returns:
        dict with predictions, targets, and metrics
    """
    if not torch.cuda.is_available() and device == 'cuda':
        device = 'cpu'

    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            if is_siamese:
                batch_t1, batch_t2, batch_y = batch
                batch_t1 = batch_t1.to(device)
                batch_t2 = batch_t2.to(device)
                outputs = model(batch_t1, batch_t2)
            else:
                batch_X, batch_y = batch
                batch_X = batch_X.to(device)
                outputs = model(batch_X)

            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Compute metrics
    accuracy = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)

    # Binary metrics for change detection
    precision = precision_score(all_targets, all_preds, average='binary', zero_division=0)
    recall_val = recall_score(all_targets, all_preds, average='binary', zero_division=0)
    f1_change = f1_score(all_targets, all_preds, average='binary', zero_division=0)

    target_names = ['No Change', 'Deforestation']
    report = classification_report(
        all_targets, all_preds, target_names=target_names, zero_division=0
    )

    if verbose:
        print("\n" + "=" * 50)
        print("TEST SET EVALUATION")
        print("=" * 50)
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")
        print(f"  F1 (change): {f1_change:.4f}")
        print(f"  Kappa: {kappa:.4f}")
        print(f"  Precision (change): {precision:.4f}")
        print(f"  Recall (change): {recall_val:.4f}")
        print(f"\n{report}")

    return {
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_change': f1_change,
        'kappa': kappa,
        'precision_change': precision,
        'recall_change': recall_val,
        'confusion_matrix': cm,
        'report': report,
    }


def save_test_results(results, save_dir, verbose=True):
    """
    Save test results to npz file.

    Args:
        results: Dict from evaluate_model
        save_dir: Directory to save results
        verbose: Print save path
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'test_results.npz')

    np.savez(
        save_path,
        predictions=results['predictions'],
        targets=results['targets'],
        probabilities=results['probabilities'],
        accuracy=results['accuracy'],
        f1_macro=results['f1_macro'],
        f1_weighted=results['f1_weighted'],
        f1_change=results['f1_change'],
        kappa=results['kappa'],
        confusion_matrix=results['confusion_matrix'],
    )

    if verbose:
        print(f"  Results saved to: {save_path}")


def create_resnet_for_classification(num_classes, in_channels=23,
                                      variant='resnet101', pretrained=True):
    """
    Create ResNet model for land cover classification (PCC approach).

    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels
        variant: ResNet variant ('resnet18', 'resnet34', 'resnet101', 'resnet152')
        pretrained: Use pretrained weights

    Returns:
        Modified ResNet model
    """
    from torchvision import models

    model_constructors = {
        'resnet18': (models.resnet18, models.ResNet18_Weights.DEFAULT),
        'resnet34': (models.resnet34, models.ResNet34_Weights.DEFAULT),
        'resnet50': (models.resnet50, models.ResNet50_Weights.DEFAULT),
        'resnet101': (models.resnet101, models.ResNet101_Weights.DEFAULT),
        'resnet152': (models.resnet152, models.ResNet152_Weights.DEFAULT),
    }

    if variant not in model_constructors:
        raise ValueError(f"Unsupported variant: {variant}")

    constructor, weights = model_constructors[variant]
    model = constructor(weights=weights if pretrained else None)

    # Modify first conv layer for multi-channel input
    original_conv = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    if pretrained:
        with torch.no_grad():
            avg_weight = original_conv.weight.mean(dim=1, keepdim=True)
            model.conv1.weight.copy_(avg_weight.repeat(1, in_channels, 1, 1))

    # Modify final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
