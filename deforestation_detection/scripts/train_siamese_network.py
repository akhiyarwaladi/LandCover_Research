"""
Train Siamese CNN for Change Detection

Approach B: Siamese ResNet-50 that learns to detect deforestation
directly from bi-temporal image pairs.

Usage:
    python scripts/train_siamese_network.py [--backbone resnet50] [--epochs 50]

Input:
    data/patches/patches_{year1}_{year2}.npz (from prepare_patches.py)

Output:
    results/models/siamese_resnet50/
        best_model.pth
        training_history.npz
        test_results.npz
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.siamese_network import (
    get_siamese_model, SiameseDataset, count_parameters
)
from modules.deep_learning_trainer import (
    train_siamese_model, evaluate_model, save_test_results
)
from modules.preprocessor import split_patches_train_test
from modules.data_loader import get_consecutive_year_pairs


# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCHES_DIR = os.path.join(BASE_DIR, 'data', 'patches')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 10
FOCAL_GAMMA = 2.0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_all_patches(verbose=True):
    """
    Load and combine patches from all year pairs.

    Returns:
        dict with 'patches_t1', 'patches_t2', 'labels'
    """
    year_pairs = get_consecutive_year_pairs()
    all_t1 = []
    all_t2 = []
    all_labels = []

    for year1, year2 in year_pairs:
        patch_file = os.path.join(PATCHES_DIR, f'patches_{year1}_{year2}.npz')

        if not os.path.exists(patch_file):
            if verbose:
                print(f"  Skipping {year1}-{year2}: file not found")
            continue

        data = np.load(patch_file)
        all_t1.append(data['patches_t1'])
        all_t2.append(data['patches_t2'])
        all_labels.append(data['labels'])

        if verbose:
            n = len(data['labels'])
            n_change = np.sum(data['labels'] == 1)
            print(f"  {year1}-{year2}: {n:,} patches ({n_change:,} change)")

    if not all_t1:
        return None

    combined = {
        'patches_t1': np.concatenate(all_t1, axis=0),
        'patches_t2': np.concatenate(all_t2, axis=0),
        'labels': np.concatenate(all_labels, axis=0),
    }

    if verbose:
        total = len(combined['labels'])
        total_change = np.sum(combined['labels'] == 1)
        print(f"\n  Total: {total:,} patches ({total_change:,} change, "
              f"{total - total_change:,} no-change)")

    return combined


def main():
    parser = argparse.ArgumentParser(description='Train Siamese CNN for change detection')
    parser.add_argument('--backbone', default='resnet50',
                        choices=['resnet34', 'resnet50'])
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--dropout', type=float, default=0.5)
    args = parser.parse_args()

    print("=" * 60)
    print("SIAMESE NETWORK TRAINING")
    print("=" * 60)
    print(f"  Backbone: {args.backbone}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {DEVICE}")

    save_dir = os.path.join(RESULTS_DIR, 'models', f'siamese_{args.backbone}')
    os.makedirs(save_dir, exist_ok=True)

    # Load patches
    print("\nLoading training patches...")
    patches = load_all_patches()

    if patches is None:
        print("\nERROR: No patch files found!")
        print(f"Expected in: {PATCHES_DIR}")
        print("Run 'python scripts/prepare_patches.py' first")
        return

    in_channels = patches['patches_t1'].shape[1]
    print(f"\n  Input channels: {in_channels}")
    print(f"  Patch size: {patches['patches_t1'].shape[2]}x{patches['patches_t1'].shape[3]}")

    # Split train/test
    print("\nSplitting data...")
    train_patches, test_patches = split_patches_train_test(
        patches, test_size=0.2, random_state=42
    )

    print(f"  Train: {len(train_patches['labels']):,} patches "
          f"({np.sum(train_patches['labels'] == 1):,} change)")
    print(f"  Test: {len(test_patches['labels']):,} patches "
          f"({np.sum(test_patches['labels'] == 1):,} change)")

    # Create datasets and loaders
    train_dataset = SiameseDataset(
        train_patches['patches_t1'], train_patches['patches_t2'],
        train_patches['labels'], augment=True
    )
    test_dataset = SiameseDataset(
        test_patches['patches_t1'], test_patches['patches_t2'],
        test_patches['labels'], augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )

    # Create model
    model = get_siamese_model(
        in_channels=in_channels, backbone=args.backbone,
        pretrained=True, dropout=args.dropout, device=DEVICE
    )

    n_params = count_parameters(model)
    print(f"\n  Model parameters: {n_params:,}")

    # Train
    history = train_siamese_model(
        model, train_loader, test_loader,
        num_epochs=args.epochs, lr=args.lr,
        patience=PATIENCE, focal_gamma=FOCAL_GAMMA,
        save_dir=save_dir, device=DEVICE
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth'),
                                      weights_only=True))
    results = evaluate_model(model, test_loader, device=DEVICE, is_siamese=True)
    save_test_results(results, save_dir)

    print("\n" + "=" * 60)
    print("SIAMESE NETWORK TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  F1 (macro): {results['f1_macro']:.4f}")
    print(f"  F1 (change): {results['f1_change']:.4f}")
    print(f"  Kappa: {results['kappa']:.4f}")
    print(f"  Model saved: {save_dir}")


if __name__ == '__main__':
    main()
