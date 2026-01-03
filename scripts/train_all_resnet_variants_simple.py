#!/usr/bin/env python3
"""
Train All ResNet Variants - Simple Version (IMPROVED)
======================================================

Trains ResNet18, ResNet34, ResNet101, ResNet152.
Uses same approach as ResNet50 but for different architectures.

IMPROVEMENTS:
- ✅ Data augmentation (flip, rotation)
- ✅ Learning rate scheduler (ReduceLROnPlateau)
- ✅ Early stopping with patience
- ✅ Better logging
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
import random
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

from modules.data_loader import load_klhk_data, load_sentinel2_tiles
from modules.feature_engineering import calculate_spectral_indices, combine_bands_and_indices
from modules.preprocessor import rasterize_klhk
from modules.data_preparation import extract_patches

# ============================================================================
# AUGMENTED DATASET CLASS
# ============================================================================

class AugmentedPatchDataset(Dataset):
    """Dataset with data augmentation for training."""

    def __init__(self, X, y, augment=False, flip_prob=0.5, rot_prob=0.5):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
        self.flip_prob = flip_prob
        self.rot_prob = rot_prob

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        patch = self.X[idx]
        label = self.y[idx]

        if self.augment:
            # Random horizontal flip
            if random.random() < self.flip_prob:
                patch = torch.flip(patch, [2])
            # Random vertical flip
            if random.random() < self.flip_prob:
                patch = torch.flip(patch, [1])
            # Random rotation (90/180/270)
            if random.random() < self.rot_prob:
                k = random.randint(1, 3)
                patch = torch.rot90(patch, k, [1, 2])

        return patch, label

print("="*80)
print("TRAINING ALL RESNET VARIANTS (IMPROVED)")
print("="*80)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Config
VARIANTS = ['resnet18', 'resnet34', 'resnet101', 'resnet152']
CONFIG = {
    'patch_size': 32,
    'max_patches': 100000,
    'batch_size': 16,
    'epochs': 50,  # Increased (early stopping will prevent full run)
    'lr': 0.001,  # Higher initial LR (scheduler will reduce it)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    # Scheduler params
    'scheduler_patience': 3,
    'scheduler_factor': 0.5,
    'scheduler_min_lr': 1e-6,

    # Early stopping
    'early_stop_patience': 7,

    # Augmentation
    'augmentation': True,
    'aug_flip_prob': 0.5,
    'aug_rot_prob': 0.5
}

print(f"Device: {CONFIG['device']}")
print(f"Augmentation: {'ON' if CONFIG['augmentation'] else 'OFF'}")
print(f"LR Scheduler: ReduceLROnPlateau (patience={CONFIG['scheduler_patience']})")
print(f"Early Stopping: patience={CONFIG['early_stop_patience']}")

# Load data (once)
print("\n" + "-"*80)
print("LOADING DATA")
print("-"*80)

klhk_gdf = load_klhk_data('data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson', verbose=False)
print(f"✓ KLHK: {len(klhk_gdf):,} polygons")

sentinel2_bands, s2_profile = load_sentinel2_tiles([
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000010496.tif'
], verbose=False)
print(f"✓ Sentinel-2: {sentinel2_bands.shape}")

indices = calculate_spectral_indices(sentinel2_bands, verbose=False)
features = combine_bands_and_indices(sentinel2_bands, indices)
print(f"✓ Features: {features.shape[0]} bands")

# Clean NaN/Inf values (CRITICAL FIX #4!)
print("\n🧹 Cleaning NaN/Inf values...")
has_nan = np.isnan(features).any()
has_inf = np.isinf(features).any()
if has_nan or has_inf:
    print(f"   Found NaN: {has_nan}, Inf: {has_inf}")
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    print("   ✓ Replaced with 0")
else:
    print("   ✓ No NaN/Inf found")

klhk_raster = rasterize_klhk(klhk_gdf, s2_profile, verbose=False)
print(f"✓ Labels: {klhk_raster.shape}")

# Extract patches
print("\nExtracting patches...")
X_patches, y_patches = extract_patches(
    features, klhk_raster,
    patch_size=CONFIG['patch_size'],
    max_patches=CONFIG['max_patches'],
    random_state=42,
    verbose=True
)
print(f"✓ Patches: X={X_patches.shape}, y={y_patches.shape}")

# Normalize EACH CHANNEL INDEPENDENTLY (CRITICAL FIX!)
print("\nNormalizing each channel independently...")
n_samples, n_channels, height, width = X_patches.shape
X_flat = X_patches.reshape(n_samples, n_channels, -1)

for c in range(n_channels):
    channel_data = X_flat[:, c, :].flatten()
    mean = np.mean(channel_data)
    std = np.std(channel_data)

    # Avoid division by zero
    if std < 1e-10:
        std = 1.0

    # Normalize this channel
    X_patches[:, c, :, :] = (X_patches[:, c, :, :] - mean) / std

    if c % 5 == 0:
        print(f"   Channel {c:2d}: mean={mean:8.4f}, std={std:8.4f}")

X_normalized = X_patches
print(f"✓ Normalized")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y_patches, test_size=0.2, random_state=42, stratify=y_patches
)
print(f"✓ Train: {X_train.shape[0]:,}, Test: {X_test.shape[0]:,}")

# Training function
def train_variant(variant_name):
    print("\n" + "="*80)
    print(f"TRAINING {variant_name.upper()}")
    print("="*80)

    start_time = time.time()
    os.makedirs(f'results/{variant_name}', exist_ok=True)

    # Datasets with augmentation
    train_dataset = AugmentedPatchDataset(
        X_train, y_train,
        augment=CONFIG['augmentation'],
        flip_prob=CONFIG['aug_flip_prob'],
        rot_prob=CONFIG['aug_rot_prob']
    )
    test_dataset = AugmentedPatchDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])

    # Model
    from torchvision import models
    if variant_name == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif variant_name == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif variant_name == 'resnet101':
        model = models.resnet101(pretrained=True)
    elif variant_name == 'resnet152':
        model = models.resnet152(pretrained=True)

    # FIX: Properly adapt first conv layer for 23 channels
    # This approach PROVEN to work with ResNet50 (79.80% accuracy)
    # CRITICAL: Replace conv1 FIRST, then modify weights directly on model
    original_conv = model.conv1
    model.conv1 = nn.Conv2d(23, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Initialize new conv layer weights (average pretrained weights and repeat)
    with torch.no_grad():
        weight = original_conv.weight.data  # (64, 3, 7, 7)
        # Average across RGB channels, then repeat for all 23 channels
        model.conv1.weight.data = weight.mean(dim=1, keepdim=True).repeat(1, 23, 1, 1)
    model.fc = nn.Linear(model.fc.in_features, 6)
    model = model.to(CONFIG['device'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=CONFIG['scheduler_factor'],
        patience=CONFIG['scheduler_patience'],
        min_lr=CONFIG['scheduler_min_lr'],
        verbose=False
    )

    # Training history
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(CONFIG['epochs']):
        # Train
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(CONFIG['device']), labels.to(CONFIG['device'])
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = correct / total

        # Val
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(CONFIG['device']), labels.to(CONFIG['device'])
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(test_loader)
        val_acc = correct / total

        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        # Check for improvement
        improved = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'models/{variant_name}_best.pth')
            patience_counter = 0
            improved = True
        else:
            patience_counter += 1

        print(f"[{epoch+1:2d}/{CONFIG['epochs']}] "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} "
              f"Acc: {train_acc*100:.2f}%/{val_acc*100:.2f}% "
              f"LR: {current_lr:.6f} "
              f"{'✓' if improved else ''} "
              f"[P:{patience_counter}]")

        # Early stopping
        if patience_counter >= CONFIG['early_stop_patience']:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            print(f"   No improvement for {CONFIG['early_stop_patience']} epochs")
            print(f"   Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
            break

    # Final eval
    model.load_state_dict(torch.load(f'models/{variant_name}_best.pth'))
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(CONFIG['device'])
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())

    acc = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

    print(f"\n✓ Test Accuracy: {acc*100:.2f}%")
    print(f"✓ F1 Macro: {f1_macro:.4f}")
    print(f"✓ F1 Weighted: {f1_weighted:.4f}")
    print(f"✓ Best epoch: {best_epoch}/{epoch+1}")
    print(f"✓ Time: {(time.time()-start_time)/60:.1f} min")

    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(all_targets, all_preds,
                                target_names=['Water', 'Trees', 'Crops', 'Shrub', 'Built', 'Bare'],
                                zero_division=0))

    # Save results
    np.savez(f'results/{variant_name}/training_history.npz', **history)
    np.savez(f'results/{variant_name}/test_results.npz',
             predictions=all_preds, targets=all_targets,
             accuracy=acc, f1_macro=f1_macro, f1_weighted=f1_weighted,
             best_epoch=best_epoch, total_epochs=epoch+1)

    return {
        'variant': variant_name,
        'acc': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'best_epoch': best_epoch,
        'total_epochs': epoch+1,
        'time_min': (time.time()-start_time)/60
    }

# Train all
summaries = []
for i, variant in enumerate(VARIANTS, 1):
    print(f"\n{'#'*80}")
    print(f"# {i}/4: {variant.upper()}")
    print(f"{'#'*80}")
    summary = train_variant(variant)
    summaries.append(summary)

# Summary
print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
for s in summaries:
    print(f"{s['variant']}: {s['acc']*100:.2f}% "
          f"(F1: {s['f1_macro']:.4f}, F1w: {s['f1_weighted']:.4f}) "
          f"[{s['best_epoch']}/{s['total_epochs']} epochs, {s['time_min']:.1f}min]")

import json
with open('results/all_variants_summary.json', 'w') as f:
    json.dump(summaries, f, indent=2)

print(f"\n✓ Saved to results/all_variants_summary.json")
print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
