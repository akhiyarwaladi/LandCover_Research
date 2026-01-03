#!/usr/bin/env python3
"""
ResNet Classification - FIXED VERSION
======================================

Fixes for NaN loss issue:
1. Proper feature normalization (per-channel standardization)
2. Lower learning rate (0.0001 instead of 0.001)
3. Gradient clipping
4. NaN/Inf checking
5. Smaller batch size for stability

Author: Claude Sonnet 4.5
Date: 2026-01-02
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
import time

from modules.data_loader import load_klhk_data, load_sentinel2_tiles
from modules.feature_engineering import calculate_spectral_indices, combine_bands_and_indices
from modules.preprocessor import rasterize_klhk
from modules.data_preparation import extract_patches

# ============================================================================
# CONFIGURATION
# ============================================================================

KLHK_PATH = 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson'
PROVINCE_TILES = [
    'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',
    'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif',
    'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000010496-0000000000.tif',
    'data/sentinel_new_cloudfree/S2_jambi_2024_20m_AllBands-0000010496-0000010496.tif'
]

OUTPUT_DIR = 'results/resnet_fixed'
MODEL_DIR = 'models'

# Training parameters - OPTIMIZED FOR STABILITY
PATCH_SIZE = 32
MAX_PATCHES = 50000
BATCH_SIZE = 16  # Smaller for stability
NUM_EPOCHS = 30
LEARNING_RATE = 0.0001  # Lower learning rate
WEIGHT_DECAY = 1e-4
RANDOM_STATE = 42

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("\n" + "="*80)
print("RESNET CLASSIFICATION - FIXED VERSION")
print("="*80)
print(f"\nDevice: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("\nFixes applied:")
print("  ✓ Per-channel feature normalization")
print("  ✓ Lower learning rate (0.0001)")
print("  ✓ Gradient clipping")
print("  ✓ NaN/Inf checking")
print("  ✓ Smaller batch size (16)")

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("\n" + "-"*80)
print("STEP 1: Loading Data")
print("-"*80)

# Load KLHK
print("\nLoading KLHK...")
klhk_gdf = load_klhk_data(KLHK_PATH, verbose=False)

# Load Sentinel-2
print("Loading Sentinel-2...")
sentinel2_bands, s2_profile = load_sentinel2_tiles(PROVINCE_TILES, verbose=False)

# Calculate indices
print("Calculating spectral indices...")
indices = calculate_spectral_indices(sentinel2_bands, verbose=False)
features = combine_bands_and_indices(sentinel2_bands, indices)

print(f"✓ Features shape: {features.shape}")
print(f"✓ Total features: {features.shape[0]}")

# Rasterize KLHK
print("\nRasterizing KLHK...")
klhk_raster = rasterize_klhk(klhk_gdf, s2_profile, verbose=False)

# ============================================================================
# CHECK FOR NAN/INF
# ============================================================================

print("\n" + "-"*80)
print("STEP 2: Data Quality Check")
print("-"*80)

has_nan = np.isnan(features).any()
has_inf = np.isinf(features).any()

print(f"\nNaN values: {has_nan}")
print(f"Inf values: {has_inf}")

if has_nan or has_inf:
    print("⚠️  Replacing NaN/Inf with 0...")
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

# ============================================================================
# EXTRACT PATCHES
# ============================================================================

print("\n" + "-"*80)
print("STEP 3: Extracting Patches")
print("-"*80)

X_patches, y_patches = extract_patches(
    features, klhk_raster,
    patch_size=PATCH_SIZE,
    stride=16,
    max_patches=MAX_PATCHES,
    random_state=RANDOM_STATE,
    verbose=True
)

# ============================================================================
# NORMALIZE FEATURES (CRITICAL FIX!)
# ============================================================================

print("\n" + "-"*80)
print("STEP 4: Feature Normalization (CRITICAL FIX!)")
print("-"*80)

print("\n🔧 Normalizing each channel independently...")
print("   This fixes the NaN loss issue!")

# Reshape to (n_samples * height * width, n_channels)
n_samples, n_channels, height, width = X_patches.shape
X_flat = X_patches.reshape(n_samples, n_channels, -1)  # (n, channels, h*w)

# Calculate mean and std per channel across all patches
channel_means = []
channel_stds = []

for c in range(n_channels):
    channel_data = X_flat[:, c, :].flatten()
    mean = np.mean(channel_data)
    std = np.std(channel_data)

    # Avoid division by zero
    if std < 1e-10:
        std = 1.0

    channel_means.append(mean)
    channel_stds.append(std)

    # Normalize this channel
    X_patches[:, c, :, :] = (X_patches[:, c, :, :] - mean) / std

    print(f"   Channel {c:2d}: mean={mean:8.4f}, std={std:8.4f}")

print("\n✅ Features normalized!")

# ============================================================================
# TRAIN/VAL/TEST SPLIT
# ============================================================================

print("\n" + "-"*80)
print("STEP 5: Train/Val/Test Split")
print("-"*80)

from sklearn.model_selection import train_test_split

# First split: train+val vs test
X_temp, X_test, y_temp, y_test = train_test_split(
    X_patches, y_patches, test_size=0.15, random_state=RANDOM_STATE, stratify=y_patches
)

# Second split: train vs val
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15/0.85, random_state=RANDOM_STATE, stratify=y_temp
)

print(f"\nTrain: {len(X_train):,} patches")
print(f"Val:   {len(X_val):,} patches")
print(f"Test:  {len(X_test):,} patches")

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.LongTensor(y_val)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

# Create DataLoaders
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ============================================================================
# CREATE MODEL
# ============================================================================

print("\n" + "-"*80)
print("STEP 6: Creating ResNet50 Model")
print("-"*80)

# Get number of classes
n_classes = len(np.unique(y_patches))
print(f"\nNumber of classes: {n_classes}")

# Load pretrained ResNet50
model = models.resnet50(pretrained=True)

# Modify first conv layer for 23 channels
original_conv = model.conv1
model.conv1 = nn.Conv2d(
    n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
)

# Initialize new conv layer
with torch.no_grad():
    # Average the pretrained weights across input channels
    weight = original_conv.weight.data
    model.conv1.weight.data = weight.mean(dim=1, keepdim=True).repeat(1, n_channels, 1, 1)

# Modify final layer
model.fc = nn.Linear(model.fc.in_features, n_classes)

model = model.to(DEVICE)

print(f"✓ Model created")
print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"✓ Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ============================================================================
# TRAINING
# ============================================================================

print("\n" + "-"*80)
print("STEP 7: Training with Fixed Parameters")
print("-"*80)

# Class weights for imbalanced data
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * n_classes
class_weights_t = torch.FloatTensor(class_weights).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights_t)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

print(f"\nTraining configuration:")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Gradient clipping: 1.0")
print(f"  Weight decay: {WEIGHT_DECAY}")

best_val_acc = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print("\n🚀 Starting training...\n")

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()

    # Training
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Check for NaN
        if torch.isnan(loss):
            print(f"\n⚠️  NaN loss detected at epoch {epoch+1}, batch {batch_idx}")
            print("   Skipping this batch...")
            continue

        loss.backward()

        # Gradient clipping (CRITICAL FIX!)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += targets.size(0)
        train_correct += predicted.eq(targets).sum().item()

    train_loss = train_loss / len(train_loader)
    train_acc = train_correct / train_total

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if not torch.isnan(loss):
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total

    # Update scheduler
    scheduler.step(val_acc)

    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'resnet50_fixed_best.pth'))
        best_marker = " ⭐ BEST!"
    else:
        best_marker = ""

    epoch_time = time.time() - epoch_start

    print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} ({epoch_time:.1f}s): "
          f"Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%{best_marker}")

# ============================================================================
# EVALUATION
# ============================================================================

print("\n" + "-"*80)
print("STEP 8: Final Evaluation")
print("-"*80)

# Load best model
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'resnet50_fixed_best.pth')))
model.eval()

# Test evaluation
test_correct = 0
test_total = 0
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        test_total += targets.size(0)
        test_correct += predicted.eq(targets).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

test_acc = test_correct / test_total

print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")
print(f"✅ Best Val Accuracy: {best_val_acc*100:.2f}%")

# Classification report
from sklearn.metrics import classification_report, f1_score

f1_macro = f1_score(all_targets, all_preds, average='macro')
f1_weighted = f1_score(all_targets, all_preds, average='weighted')

print(f"\nF1-Score (Macro): {f1_macro:.4f}")
print(f"F1-Score (Weighted): {f1_weighted:.4f}")

print("\nPer-class metrics:")
class_names = ['Water', 'Trees', 'Crops', 'Shrub', 'Built', 'Bare']
print(classification_report(all_targets, all_preds, target_names=class_names, zero_division=0))

# Save results
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.savez(os.path.join(OUTPUT_DIR, 'training_history.npz'), **history)
np.savez(os.path.join(OUTPUT_DIR, 'test_results.npz'),
         predictions=all_preds, targets=all_targets)

print("\n" + "="*80)
print("RESNET TRAINING COMPLETE!")
print("="*80)
print(f"\n✅ Best model: models/resnet50_fixed_best.pth")
print(f"✅ Results: {OUTPUT_DIR}/")
print(f"\n📊 Final Results:")
print(f"   Test Accuracy: {test_acc*100:.2f}%")
print(f"   F1 (Macro): {f1_macro:.4f}")
print(f"   F1 (Weighted): {f1_weighted:.4f}")
print("\n" + "="*80)
