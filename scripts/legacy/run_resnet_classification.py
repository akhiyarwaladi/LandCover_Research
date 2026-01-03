#!/usr/bin/env python3
"""
ResNet Deep Learning Classification - Main Orchestrator
========================================================

This script performs land cover classification using ResNet transfer learning
with KLHK ground truth data. This approach is DIFFERENT from the previous
Random Forest work.

NOVEL CONTRIBUTIONS:
1. Demonstrates successful KLHK geometry access via KMZ format
2. Applies deep learning (ResNet transfer learning) to KLHK data
3. Compares deep learning vs traditional machine learning approaches
4. Establishes baseline for future advanced architectures (U-Net, ViT)

METHODOLOGY:
- Transfer learning with ResNet50 pretrained on ImageNet
- Patch-based classification (32x32 patches)
- Fine-tuning only final layers
- Data augmentation (flips, rotations)
- Validation-based early stopping

EXPECTED RESULTS:
- Accuracy: 85-90% (vs 74.95% with Random Forest)
- Training time: ~30-60 minutes (GPU required)
- Model size: ~100 MB (pretrained weights)

Usage:
    python scripts/run_resnet_classification.py

Configuration is done via constants at the top of the script.

NOTE: Requires PyTorch and CUDA-capable GPU for reasonable training time.
      Can run on CPU but will be significantly slower.
"""

import sys
import os

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.data_loader import load_klhk_data, load_sentinel2_tiles
from modules.feature_engineering import (
    calculate_spectral_indices,
    combine_bands_and_indices
)
from modules.preprocessor import rasterize_klhk
from modules.data_preparation import (
    extract_patches,
    get_data_loaders,
    get_class_weights
)
from modules.deep_learning_trainer import (
    get_resnet_model,
    modify_first_conv_for_multispectral,
    train_model,
    evaluate_model,
    save_model
)
import torch
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input data paths
KLHK_PATH = 'data/klhk/KLHK_PL2024_Jambi_Full_WithGeometry.geojson'
SENTINEL2_TILES = [
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000000000-0000010496.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000000000.tif',
    'data/sentinel/S2_jambi_2024_20m_AllBands-0000010496-0000010496.tif'
]

# Output directory
OUTPUT_DIR = 'results/resnet_classification'
MODEL_DIR = 'models'

# Patch extraction configuration
PATCH_SIZE = 32          # Size of patches (32x32 pixels)
STRIDE = 16             # Stride for sliding window (overlap)
MAX_PATCHES = 50000     # Maximum patches to extract (memory limit)

# Training configuration
BATCH_SIZE = 32         # Batch size for training
NUM_EPOCHS = 20         # Number of training epochs
LEARNING_RATE = 0.001   # Initial learning rate
VAL_SIZE = 0.15         # Validation split
TEST_SIZE = 0.15        # Test split

# Model configuration
MODEL_TYPE = 'resnet50'  # ResNet variant
PRETRAINED = True        # Use ImageNet pretrained weights
FREEZE_BASE = True       # Freeze convolutional base

# Hardware configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 4         # Data loader workers

# Random seed
RANDOM_STATE = 42

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Main execution workflow for ResNet classification."""

    print("=" * 70)
    print("DEEP LEARNING LAND COVER CLASSIFICATION - ResNet Transfer Learning")
    print("=" * 70)
    print("\n🎯 OBJECTIVE: Apply deep learning to KLHK ground truth data")
    print("   (Different method from previous Random Forest work)")
    print("\nData Sources:")
    print(f"  - KLHK Reference: {KLHK_PATH}")
    print(f"    └─ Access Method: KMZ format (28,100 polygons)")
    print(f"  - Sentinel-2 Tiles: {len(SENTINEL2_TILES)} tiles")
    print(f"\nDeep Learning Configuration:")
    print(f"  - Model: {MODEL_TYPE} (transfer learning)")
    print(f"  - Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - Device: {DEVICE}")

    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ------------------------------------------------------------------------
    # STEP 1: Load KLHK Reference Data
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 1: Loading KLHK Reference Data")
    print("-" * 70)

    klhk_gdf = load_klhk_data(KLHK_PATH, verbose=True)

    # ------------------------------------------------------------------------
    # STEP 2: Load Sentinel-2 Imagery
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 2: Loading Sentinel-2 Imagery")
    print("-" * 70)

    sentinel2_bands, s2_profile = load_sentinel2_tiles(SENTINEL2_TILES, verbose=True)

    # ------------------------------------------------------------------------
    # STEP 3: Calculate Spectral Indices
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 3: Calculating Spectral Indices")
    print("-" * 70)

    indices = calculate_spectral_indices(sentinel2_bands, verbose=True)
    features = combine_bands_and_indices(sentinel2_bands, indices)
    print(f"\n  Total features: {features.shape[0]} (10 bands + 13 indices)")

    # ------------------------------------------------------------------------
    # STEP 4: Rasterize KLHK Ground Truth
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 4: Rasterizing KLHK Ground Truth")
    print("-" * 70)

    klhk_raster = rasterize_klhk(klhk_gdf, s2_profile, verbose=True)

    # ------------------------------------------------------------------------
    # STEP 5: Extract Patches for Deep Learning
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 5: Extracting Image Patches")
    print("-" * 70)
    print("\n🔄 This is DIFFERENT from Random Forest:")
    print("   - Random Forest: Uses individual pixels")
    print("   - ResNet: Uses 32x32 patches (spatial context)")

    X_patches, y_patches = extract_patches(
        features,
        klhk_raster,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        max_patches=MAX_PATCHES,
        random_state=RANDOM_STATE,
        verbose=True
    )

    # ------------------------------------------------------------------------
    # STEP 6: Create Data Loaders
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 6: Creating Data Loaders")
    print("-" * 70)

    train_loader, val_loader, test_loader = get_data_loaders(
        X_patches, y_patches,
        batch_size=BATCH_SIZE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        num_workers=NUM_WORKERS,
        verbose=True
    )

    # Calculate class weights for imbalanced data
    class_weights = get_class_weights(y_patches, device=DEVICE)
    print(f"\n  Class weights calculated for imbalanced data")

    # ------------------------------------------------------------------------
    # STEP 7: Create ResNet Model
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 7: Creating ResNet Model")
    print("-" * 70)

    num_classes = len(np.unique(y_patches))

    model = get_resnet_model(
        num_classes=num_classes,
        pretrained=PRETRAINED,
        freeze_base=FREEZE_BASE,
        model_type=MODEL_TYPE,
        verbose=True
    )

    # Modify for multispectral input (23 channels)
    print(f"\n  Modifying first layer for {features.shape[0]} spectral channels...")
    model = modify_first_conv_for_multispectral(
        model,
        in_channels=features.shape[0],
        model_type=MODEL_TYPE
    )

    print(f"\n✅ Model ready for training!")

    # ------------------------------------------------------------------------
    # STEP 8: Train Model
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 8: Training ResNet Model")
    print("=" * 70)
    print("\n⏱️  This will take approximately 30-60 minutes on GPU...")
    print("   (Or 4-6 hours on CPU)")

    history, best_model_state = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        class_weights=class_weights,
        verbose=True
    )

    # Load best model
    model.load_state_dict(best_model_state)

    # Save best model
    model_path = f'{MODEL_DIR}/resnet50_best.pth'
    save_model(model, model_path, metadata={
        'accuracy': max(history['val_acc']),
        'num_classes': num_classes,
        'patch_size': PATCH_SIZE,
        'model_type': MODEL_TYPE
    })
    print(f"\n✅ Best model saved to: {model_path}")

    # ------------------------------------------------------------------------
    # STEP 9: Evaluate on Test Set
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 9: Evaluating on Test Set")
    print("=" * 70)

    # Class names
    class_names = [
        'Water', 'Trees/Forest', 'Crops/Agriculture',
        'Shrub/Scrub', 'Built Area', 'Bare Ground'
    ]

    results = evaluate_model(
        model,
        test_loader,
        device=DEVICE,
        class_names=class_names,
        verbose=True
    )

    # Save training history and predictions for figure generation
    print(f"\n📊 Saving training history and predictions...")

    # Save training history
    history_path = f'{OUTPUT_DIR}/training_history.npz'
    np.savez(history_path,
             train_loss=np.array(history['train_loss']),
             train_acc=np.array(history['train_acc']),
             val_loss=np.array(history['val_loss']),
             val_acc=np.array(history['val_acc']),
             epoch_time=np.array(history['epoch_time']))
    print(f"   ✓ Training history saved: {history_path}")

    # Save test predictions
    predictions_path = f'{OUTPUT_DIR}/test_predictions.npz'
    np.savez(predictions_path,
             y_true=results['y_true'],
             y_pred=results['y_pred'])
    print(f"   ✓ Test predictions saved: {predictions_path}")

    # ------------------------------------------------------------------------
    # STEP 10: Results Summary
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY - ResNet vs Random Forest")
    print("=" * 70)

    print(f"\n📊 ResNet50 Results:")
    print(f"   Accuracy: {results['accuracy']:.4f}")
    print(f"   F1 (macro): {results['f1_macro']:.4f}")
    print(f"   F1 (weighted): {results['f1_weighted']:.4f}")

    print(f"\n📊 Random Forest Baseline (Previous Work):")
    print(f"   Accuracy: 0.7495")
    print(f"   F1 (macro): 0.542")
    print(f"   F1 (weighted): 0.744")

    improvement = results['accuracy'] - 0.7495
    print(f"\n💡 Improvement: {improvement:+.2%}")

    if improvement > 0:
        print(f"   ✅ ResNet outperforms Random Forest by {improvement:.2%}")
        print(f"   ✅ Deep learning shows advantage for land cover classification")
    else:
        print(f"   ⚠️  ResNet did not improve over Random Forest")
        print(f"   ℹ️  This can happen with small datasets")

    # ------------------------------------------------------------------------
    # COMPLETION
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESNET CLASSIFICATION COMPLETE!")
    print("=" * 70)
    print(f"\n✅ Results saved to: {OUTPUT_DIR}/")
    print(f"✅ Model saved to: {model_path}")
    print(f"\n🔬 NOVEL CONTRIBUTIONS:")
    print(f"   1. First deep learning application to KLHK ground truth")
    print(f"   2. Transfer learning demonstrates {improvement:+.2%} improvement")
    print(f"   3. Establishes baseline for future U-Net/ViT research")
    print(f"\n📊 NEXT STEPS:")
    print(f"   - Visualize training curves: plot history")
    print(f"   - Generate confusion matrix and per-class metrics")
    print(f"   - Compare with Random Forest results in detail")
    print(f"   - Prepare results for manuscript")

    return {
        'model': model,
        'history': history,
        'results': results
    }


if __name__ == "__main__":
    # Check for PyTorch
    try:
        import torch
    except ImportError:
        print("\n❌ ERROR: PyTorch not installed!")
        print("\nPlease install PyTorch:")
        print("  conda install pytorch torchvision -c pytorch")
        print("  or")
        print("  pip install torch torchvision")
        sys.exit(1)

    # Check for CUDA
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available - training will use CPU")
        print("   Expected training time: 4-6 hours (vs 30-60 min on GPU)")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(0)

    results = main()
