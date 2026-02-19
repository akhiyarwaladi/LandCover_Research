"""
Train Post-Classification Comparison (PCC) with ResNet-101

Approach A: Classify each annual composite independently using ResNet-101,
then compare consecutive years to detect deforestation.

Usage:
    python scripts/train_pcc_resnet.py [--variant resnet101] [--epochs 50]

Output:
    results/models/pcc_resnet101/
        best_model.pth
        training_history.npz
        test_results.npz
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_loader import (
    load_sentinel2_tiles, find_sentinel2_tiles, get_study_years
)
from modules.feature_engineering import calculate_spectral_indices, combine_bands_and_indices
from modules.deep_learning_trainer import (
    create_resnet_for_classification, train_resnet_classifier,
    evaluate_model, save_test_results
)
from modules.change_detector import post_classification_comparison


# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENTINEL_DIR = os.path.join(BASE_DIR, 'data', 'sentinel')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Training parameters
NUM_CLASSES = 7  # Water, Forest, Grass, Crops, Shrub, Built, Bare
PATCH_SIZE = 32
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_klhk_labels():
    """Load KLHK reference data for land cover classification."""
    parent_dir = os.path.join(BASE_DIR, '..')
    sys.path.insert(0, os.path.join(parent_dir, 'modules'))

    klhk_path = os.path.join(parent_dir, 'data', 'klhk',
                              'KLHK_PL2024_Jambi_Full_WithGeometry.geojson')

    if not os.path.exists(klhk_path):
        print(f"WARNING: KLHK data not found at {klhk_path}")
        return None

    from modules.data_loader import load_klhk_data
    return load_klhk_data(klhk_path)


def create_classification_patches(features, labels, patch_size=32,
                                   sample_size=100000, random_state=42):
    """
    Extract patches for ResNet classification training.

    Args:
        features: (C, H, W) feature stack
        labels: (H, W) class labels
        patch_size: Patch size
        sample_size: Max number of patches
        random_state: Random seed

    Returns:
        patches (N, C, ps, ps), labels (N,)
    """
    C, H, W = features.shape
    half = patch_size // 2

    rng = np.random.RandomState(random_state)

    # Find valid pixels (with labels)
    valid_y, valid_x = np.where(labels >= 0)

    # Filter to valid patch positions
    valid_mask = (
        (valid_y >= half) & (valid_y < H - half) &
        (valid_x >= half) & (valid_x < W - half)
    )
    valid_y = valid_y[valid_mask]
    valid_x = valid_x[valid_mask]

    if len(valid_y) == 0:
        return np.array([]), np.array([])

    # Sample
    if sample_size and len(valid_y) > sample_size:
        idx = rng.choice(len(valid_y), sample_size, replace=False)
        valid_y = valid_y[idx]
        valid_x = valid_x[idx]

    patches = []
    patch_labels = []

    for y, x in zip(valid_y, valid_x):
        patch = features[:, y - half:y + half, x - half:x + half]
        if patch.shape[1] == patch_size and patch.shape[2] == patch_size:
            if not np.any(np.isnan(patch)):
                patches.append(patch)
                patch_labels.append(labels[y, x])

    return np.array(patches, dtype=np.float32), np.array(patch_labels, dtype=np.int64)


def classify_annual_map(model, features, patch_size=32, batch_size=256, device='cpu'):
    """
    Generate full classification map for one year.

    Args:
        model: Trained ResNet model
        features: (C, H, W) feature stack
        patch_size: Patch size used in training
        batch_size: Inference batch size
        device: Device

    Returns:
        (H, W) classified map
    """
    C, H, W = features.shape
    half = patch_size // 2
    result = np.full((H, W), -1, dtype=np.int16)

    model.eval()
    model = model.to(device)

    # Process in batches of rows
    with torch.no_grad():
        for i in range(half, H - half, 1):
            patches = []
            positions = []

            for j in range(half, W - half, patch_size // 2):
                patch = features[:, i - half:i + half, j - half:j + half]
                if patch.shape[1] == patch_size and patch.shape[2] == patch_size:
                    if not np.any(np.isnan(patch)):
                        patches.append(patch)
                        positions.append((i, j))

            if not patches:
                continue

            # Batch prediction
            for start in range(0, len(patches), batch_size):
                batch = np.array(patches[start:start + batch_size])
                batch_tensor = torch.FloatTensor(batch).to(device)
                outputs = model(batch_tensor)
                _, predicted = torch.max(outputs, 1)
                preds = predicted.cpu().numpy()

                for pred, (pi, pj) in zip(preds, positions[start:start + batch_size]):
                    result[pi, pj] = pred

    return result


def main():
    parser = argparse.ArgumentParser(description='Train PCC-ResNet for change detection')
    parser.add_argument('--variant', default='resnet101',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    print("=" * 60)
    print("POST-CLASSIFICATION COMPARISON (PCC) TRAINING")
    print("=" * 60)
    print(f"  Backbone: {args.variant}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Device: {DEVICE}")

    save_dir = os.path.join(RESULTS_DIR, 'models', f'pcc_{args.variant}')
    os.makedirs(save_dir, exist_ok=True)

    # Load KLHK reference data for training
    print("\nLoading reference data...")
    klhk_gdf = load_klhk_labels()

    if klhk_gdf is None:
        print("ERROR: Cannot train without KLHK reference data")
        return

    # Load one year of Sentinel-2 for training the classifier
    print("\nLoading Sentinel-2 training data...")
    train_year = 2024
    tiles = find_sentinel2_tiles(SENTINEL_DIR, train_year)

    if not tiles:
        # Try parent project data
        parent_sentinel = os.path.join(BASE_DIR, '..', 'data', 'sentinel_new_cloudfree')
        import glob
        tiles = sorted(glob.glob(os.path.join(parent_sentinel, '*.tif')))

    if not tiles:
        print("ERROR: No Sentinel-2 data found for training")
        return

    bands, profile = load_sentinel2_tiles(tiles)
    indices = calculate_spectral_indices(bands)
    features = combine_bands_and_indices(bands, indices)

    # Rasterize KLHK labels
    from modules.preprocessor import create_forest_mask
    parent_modules = os.path.join(BASE_DIR, '..', 'modules')
    sys.path.insert(0, parent_modules)

    from rasterio.features import rasterize
    shapes = [(geom, value) for geom, value in
              zip(klhk_gdf.geometry, klhk_gdf['class_simplified'])]
    labels = rasterize(
        shapes,
        out_shape=(profile['height'], profile['width']),
        transform=profile['transform'],
        fill=-1, dtype=np.int16
    )

    print(f"  Features shape: {features.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Valid labeled pixels: {np.sum(labels >= 0):,}")

    # Create patches
    print("\nExtracting training patches...")
    patches, patch_labels = create_classification_patches(
        features, labels, patch_size=PATCH_SIZE, sample_size=100000
    )

    if len(patches) == 0:
        print("ERROR: No valid patches extracted")
        return

    print(f"  Patches: {patches.shape}")
    print(f"  Labels: {patch_labels.shape}")

    # Remap labels to contiguous 0-based indices
    unique_classes = np.unique(patch_labels)
    class_mapping = {c: i for i, c in enumerate(unique_classes)}
    mapped_labels = np.array([class_mapping[l] for l in patch_labels])
    num_classes = len(unique_classes)

    print(f"  Number of classes: {num_classes}")
    print(f"  Class mapping: {class_mapping}")

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        patches, mapped_labels, test_size=0.2, random_state=42, stratify=mapped_labels
    )

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), torch.LongTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    # Create model
    model = create_resnet_for_classification(
        num_classes=num_classes, in_channels=features.shape[0],
        variant=args.variant, pretrained=True
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model: {args.variant} ({total_params:,} parameters)")

    # Train
    history = train_resnet_classifier(
        model, train_loader, test_loader,
        num_epochs=args.epochs, lr=args.lr, patience=PATIENCE,
        save_dir=save_dir, device=DEVICE
    )

    # Evaluate
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth'),
                                      weights_only=True))
    results = evaluate_model(model, test_loader, device=DEVICE, is_siamese=False)
    save_test_results(results, save_dir)

    # Save class mapping for inference
    np.savez(os.path.join(save_dir, 'class_mapping.npz'),
             unique_classes=unique_classes,
             class_mapping=np.array(list(class_mapping.items())))

    print("\n" + "=" * 60)
    print("PCC RESNET TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  F1 (macro): {results['f1_macro']:.4f}")
    print(f"  Model saved: {save_dir}")


if __name__ == '__main__':
    main()
