#!/usr/bin/env python3
"""
Train Multiple Model Architectures - Comprehensive Comparison
==============================================================

Trains diverse model architectures for land cover classification comparison:
- ResNet50 (baseline)
- EfficientNet-B3 (efficient compound scaling)
- ConvNeXt-Tiny (modern CNN)
- DenseNet-121 (dense connections)
- Inception-V3 (multi-scale features)

Each model will:
1. Train on same dataset (80k train, 20k test)
2. Save best model to models/
3. Save training history to results/models/{model_name}/
4. Save test results to results/models/{model_name}/

Uses model factory for easy addition of new architectures.

Author: Claude Sonnet 4.5
Date: 2026-01-04
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

# Import modules
from modules.data_loader import load_klhk_data, load_sentinel2_tiles
from modules.feature_engineering import calculate_spectral_indices, combine_bands_and_indices
from modules.preprocessor import rasterize_klhk, prepare_training_data
from modules.data_preparation import LandCoverPatchDataset, get_augmentation_transforms
from modules.dl_predictor import predict_spatial
from modules.model_factory import create_model
from modules.model_registry import get_model_info, RECOMMENDED_MODELS

print("\n" + "="*80)
print("TRAINING MULTIPLE MODEL ARCHITECTURES")
print("="*80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================================================
# Configuration
# ============================================================================

# Models to train - use recommended models for research comparison
# Comment/uncomment models as needed
MODELS_TO_TRAIN = [
    'resnet50',         # Baseline - already trained, will skip if exists
    'efficientnet_b3',  # Efficient compound scaling
    'convnext_tiny',    # Modern CNN (likely winner)
    'densenet121',      # Lightweight dense connections
    'inception_v3',     # Multi-scale feature extraction
]

# Get architecture specs from model registry
ARCH_SPECS = {model: get_model_info(model) for model in MODELS_TO_TRAIN}

# Training config (same as ResNet50 for fair comparison)
CONFIG = {
    'patch_size': 32,
    'stride': 16,
    'batch_size': 16,
    'num_epochs': 30,
    'learning_rate': 0.0001,
    'num_classes': 6,
    'input_channels': 23,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 0,  # Windows compatibility
    'sample_size': 100000,
    'test_size': 0.2,
    'random_state': 42
}

print(f"\n🖥️  Device: {CONFIG['device']}")
print(f"📊 Training samples: {CONFIG['sample_size']}")
print(f"🔢 Models to train: {len(MODELS_TO_TRAIN)}")
for model_name in MODELS_TO_TRAIN:
    specs = ARCH_SPECS[model_name]
    print(f"   - {specs['display_name']}: {specs['params']/1e6:.1f}M params, "
          f"{specs['family'].upper()} family")

# ============================================================================
# Load Data (once, reuse for all variants)
# ============================================================================

print("\n" + "-"*80)
print("STEP 1: LOADING DATA (once, reused for all variants)")
print("-"*80)

# KLHK data
print("\n1/4 Loading KLHK ground truth...")
KLHK_PATH = 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson'
klhk_gdf = load_klhk_data(KLHK_PATH, verbose=False)
print(f"✓ Loaded {len(klhk_gdf):,} polygons")

# Sentinel-2 tiles
print("\n2/4 Loading Sentinel-2 imagery...")
SENTINEL2_TILES = [
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000010496.tif'
]
sentinel2_bands, s2_profile = load_sentinel2_tiles(SENTINEL2_TILES, verbose=False)
print(f"✓ Loaded mosaic: {sentinel2_bands.shape}")

# Calculate features
print("\n3/4 Calculating spectral indices...")
indices = calculate_spectral_indices(sentinel2_bands, verbose=False)
features = combine_bands_and_indices(sentinel2_bands, indices)
print(f"✓ Total features: {features.shape[0]} bands")

# Rasterize KLHK
print("\n4/4 Rasterizing KLHK labels...")
klhk_raster = rasterize_klhk(klhk_gdf, s2_profile, verbose=False)
print(f"✓ Rasterized labels: {klhk_raster.shape}")

# Prepare training data (extract patches)
print("\n📦 Preparing training dataset...")
from modules.data_preparation import extract_patches

X_patches, y_patches = extract_patches(
    features, klhk_raster,
    patch_size=CONFIG['patch_size'],
    stride=CONFIG['stride'],
    max_patches=CONFIG['sample_size'],
    random_state=CONFIG['random_state'],
    verbose=True
)
print(f"✓ Extracted patches: X={X_patches.shape}, y={y_patches.shape}")

# Calculate normalization params (same for all models)
print("\n📊 Calculating normalization parameters...")
channel_means = X_patches.mean(axis=(0, 2, 3))
channel_stds = X_patches.std(axis=(0, 2, 3))
print(f"✓ Means shape: {channel_means.shape}")
print(f"✓ Stds shape: {channel_stds.shape}")

# Split train/test (same for all models)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_patches, y_patches,
    test_size=CONFIG['test_size'],
    random_state=CONFIG['random_state'],
    stratify=y_patches
)
print(f"\n✓ Train set: {X_train.shape[0]:,} samples")
print(f"✓ Test set: {X_test.shape[0]:,} samples")

# ============================================================================
# Training Function
# ============================================================================

def train_model(model_name, X_train, y_train, X_test, y_test,
                channel_means, channel_stds, config):
    """Train a single model (works with any architecture)."""

    print("\n" + "="*80)
    model_info = get_model_info(model_name)
    print(f"TRAINING {model_info['display_name']} ({model_info['family'].upper()})")
    print("="*80)

    start_time = time.time()

    # Create output directories (centralized structure)
    model_dir = 'models'
    results_dir = f'results/models/{model_name}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n📁 Output:")
    print(f"   Model: {model_dir}/{model_name}_best.pth")
    print(f"   Results: {results_dir}/")

    # Create datasets
    print(f"\n📦 Creating datasets...")

    # Normalize data using channel statistics
    X_train_norm = (X_train - channel_means[None, :, None, None]) / (channel_stds[None, :, None, None] + 1e-8)
    X_test_norm = (X_test - channel_means[None, :, None, None]) / (channel_stds[None, :, None, None] + 1e-8)

    # Get augmentation transforms
    train_transform = get_augmentation_transforms('train')
    test_transform = get_augmentation_transforms('test')

    train_dataset = LandCoverPatchDataset(
        X_train_norm, y_train, transform=train_transform, normalize=False
    )
    test_dataset = LandCoverPatchDataset(
        X_test_norm, y_test, transform=test_transform, normalize=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers']
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=config['num_workers']
    )

    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")

    # Create model using factory (handles all architectures automatically!)
    model, _ = create_model(
        model_name,
        num_classes=config['num_classes'],
        input_channels=config['input_channels'],
        pretrained=True,  # Use ImageNet pretrained weights
        device=config['device']
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Training loop
    print(f"\n🚀 Training for {config['num_epochs']} epochs...")

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(config['num_epochs']):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(config['device'])
            labels = labels.to(config['device'])

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(config['device'])
                labels = labels.to(config['device'])

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(test_loader)
        val_acc = val_correct / val_total

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'{model_dir}/{model_name}_best.pth')

        epoch_time = time.time() - epoch_start

        print(f"Epoch [{epoch+1:2d}/{config['num_epochs']}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | "
              f"Time: {epoch_time:.1f}s "
              f"{'✓ BEST' if epoch+1 == best_epoch else ''}")

    # Save training history
    np.savez(
        f'{results_dir}/training_history.npz',
        train_loss=history['train_loss'],
        train_acc=history['train_acc'],
        val_loss=history['val_loss'],
        val_acc=history['val_acc']
    )

    # Final evaluation on test set with best model
    print(f"\n📊 Final evaluation (best model from epoch {best_epoch})...")
    model.load_state_dict(torch.load(f'{model_dir}/{model_name}_best.pth'))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(config['device'])
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    test_acc = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

    print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")
    print(f"✅ F1-Score (Macro): {f1_macro:.4f}")
    print(f"✅ F1-Score (Weighted): {f1_weighted:.4f}")

    # Save test results
    np.savez(
        f'{results_dir}/test_results.npz',
        predictions=all_preds,
        targets=all_targets,
        accuracy=test_acc,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted
    )

    # Training summary
    total_time = time.time() - start_time

    print(f"\n⏱️  Training Time: {total_time/60:.1f} minutes")
    print(f"📊 Best Epoch: {best_epoch}")
    print(f"📊 Best Val Acc: {best_val_acc*100:.2f}%")

    # Save summary
    summary = {
        'variant': model_name,
        'parameters': total_params,
        'depth': specs['depth'],
        'best_epoch': best_epoch,
        'best_val_acc': float(best_val_acc),
        'test_acc': float(test_acc),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'training_time_minutes': total_time / 60,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    import json
    with open(f'{results_dir}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary

# ============================================================================
# Train All Variants
# ============================================================================

all_summaries = []

for i, variant in enumerate(MODELS_TO_TRAIN, 1):
    print(f"\n\n{'#'*80}")
    print(f"# MODEL {i}/{len(MODELS_TO_TRAIN)}: {variant.upper()}")
    print(f"{'#'*80}")

    summary = train_model(
        variant, X_train, y_train, X_test, y_test,
        channel_means, channel_stds, CONFIG
    )

    all_summaries.append(summary)

    print(f"\n✅ {variant.upper()} COMPLETE!")
    print(f"   Accuracy: {summary['test_acc']*100:.2f}%")
    print(f"   Time: {summary['training_time_minutes']:.1f} minutes")

# ============================================================================
# Generate Predictions for All Variants
# ============================================================================

print("\n\n" + "="*80)
print("GENERATING SPATIAL PREDICTIONS FOR ALL MODELS")
print("="*80)

for model_name in MODELS_TO_TRAIN:
    model_path = f'models/{model_name}_best.pth'
    if not os.path.exists(model_path):
        print(f"\n⏭️  Skipping {model_name} - model not trained")
        continue

    print(f"\n🗺️  Generating predictions for {ARCH_SPECS[model_name]['display_name']}...")

    try:
        predictions, results = predict_spatial(
            model=model_path,
            features=features,
            labels=klhk_raster,
            channel_means=channel_means,
            channel_stds=channel_stds,
            patch_size=CONFIG['patch_size'],
            stride=CONFIG['stride'],
            batch_size=64,
            device=CONFIG['device'],
            verbose=False
        )

        # Save predictions
        np.save(f'results/{variant}/predictions.npy', predictions)

        print(f"✓ Saved predictions: results/{variant}/predictions.npy")
        print(f"  Accuracy: {results['accuracy']*100:.2f}%")
        print(f"  Speed: {results['speed']:.0f} patches/sec")

    except Exception as e:
        print(f"❌ Error generating predictions for {variant}: {e}")

# ============================================================================
# Final Summary
# ============================================================================

print("\n\n" + "="*80)
print("ALL MODELS TRAINING COMPLETE!")
print("="*80)

print(f"\n📊 SUMMARY OF ALL MODELS:")
print("-"*80)
print(f"{'Model':<20} {'Params (M)':<12} {'Test Acc (%)':<14} {'F1 (Macro)':<12} {'Time (min)':<12}")
print("-"*80)

for summary in all_summaries:
    print(f"{summary['variant']:<20} "
          f"{summary['parameters']/1e6:<12.1f} "
          f"{summary['test_acc']*100:<14.2f} "
          f"{summary['f1_macro']:<12.4f} "
          f"{summary['training_time_minutes']:<12.1f}")

print("-"*80)

# Save combined summary
import json
with open('results/all_models_summary.json', 'w') as f:
    json.dump(all_summaries, f, indent=2)

print(f"\n✅ Combined summary saved to: results/all_models_summary.json")

total_time = sum(s['training_time_minutes'] for s in all_summaries)
print(f"\n⏱️  Total Training Time: {total_time:.1f} minutes ({total_time/60:.2f} hours)")

print(f"\n📁 Models saved to: models/")
for model_name in MODELS_TO_TRAIN:
    model_path = f'models/{model_name}_best.pth'
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024**2)
        print(f"   ✓ {model_name}_best.pth ({size_mb:.1f} MB)")

print(f"\n📁 Results saved to:")
for model_name in MODELS_TO_TRAIN:
    results_dir = f'results/models/{model_name}'
    if os.path.exists(results_dir):
        print(f"   ✓ {results_dir}/")

print("\n" + "="*80)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
print("\n🎉 ALL DONE! Ready for multi-architecture comparison!")
print("\n📊 Next steps:")
print("   1. Run scripts/generate_publication_comparison.py")
print("   2. Run scripts/generate_statistical_analysis.py")
print("   3. Compare results across all model families!")
print("\n" + "="*80)
