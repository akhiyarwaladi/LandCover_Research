"""
Preprocessor Module
===================

Handles creation of change labels from Hansen GFC, patch extraction
for Siamese networks, and data preparation for all approaches.
"""

import numpy as np
from sklearn.model_selection import train_test_split


# Default tree cover threshold (Hansen standard)
DEFAULT_TREECOVER_THRESHOLD = 30


def create_forest_mask(treecover2000, threshold=DEFAULT_TREECOVER_THRESHOLD, verbose=True):
    """
    Create binary forest mask from Hansen tree cover 2000.

    Args:
        treecover2000: (H, W) tree cover percentage array
        threshold: Minimum % to classify as forest
        verbose: Print info

    Returns:
        (H, W) boolean array (True = forest)
    """
    mask = treecover2000 >= threshold

    if verbose:
        total = mask.size
        forest = np.sum(mask)
        print(f"Forest mask: {forest:,} pixels ({100 * forest / total:.1f}%) "
              f"at {threshold}% threshold")

    return mask


def create_annual_change_labels(lossyear, year_range=(18, 24),
                                 treecover2000=None,
                                 treecover_threshold=DEFAULT_TREECOVER_THRESHOLD,
                                 verbose=True):
    """
    Extract per-year deforestation labels from Hansen lossyear.

    Hansen lossyear values 1-24 correspond to years 2001-2024.
    Values 18-24 correspond to our study period 2018-2024.

    Args:
        lossyear: (H, W) Hansen lossyear array
        year_range: Tuple (start_val, end_val) for lossyear values
        treecover2000: Optional tree cover for masking non-forest
        treecover_threshold: Minimum tree cover % to consider
        verbose: Print statistics

    Returns:
        dict: year -> (H, W) binary change label (1=deforestation, 0=no change)
    """
    if verbose:
        print("Creating annual change labels from Hansen GFC...")

    start_val, end_val = year_range
    annual_labels = {}

    # Optional forest mask
    forest_mask = None
    if treecover2000 is not None:
        forest_mask = create_forest_mask(treecover2000, treecover_threshold, verbose=False)

    for yr_val in range(start_val, end_val + 1):
        year = 2000 + yr_val
        label = (lossyear == yr_val).astype(np.uint8)

        if forest_mask is not None:
            # Only count deforestation where there was forest
            label = label & forest_mask

        change_pixels = np.sum(label)
        total_pixels = label.size

        annual_labels[year] = label

        if verbose:
            area_ha = change_pixels * 0.04  # 20m pixel = 0.04 ha
            print(f"  {year}: {change_pixels:,} deforestation pixels "
                  f"({area_ha:,.0f} ha, {100 * change_pixels / total_pixels:.3f}%)")

    return annual_labels


def create_cumulative_change_labels(lossyear, base_year_val=18, end_year_val=24,
                                     treecover2000=None,
                                     treecover_threshold=DEFAULT_TREECOVER_THRESHOLD,
                                     verbose=True):
    """
    Create cumulative deforestation labels from base year to each subsequent year.

    Args:
        lossyear: (H, W) Hansen lossyear array
        base_year_val: Starting lossyear value (18 = 2018)
        end_year_val: Ending lossyear value (24 = 2024)
        treecover2000: Optional tree cover for masking
        treecover_threshold: Minimum tree cover %
        verbose: Print statistics

    Returns:
        dict: year -> (H, W) cumulative change label
    """
    if verbose:
        print("Creating cumulative change labels...")

    forest_mask = None
    if treecover2000 is not None:
        forest_mask = create_forest_mask(treecover2000, treecover_threshold, verbose=False)

    cumulative_labels = {}
    for yr_val in range(base_year_val, end_year_val + 1):
        year = 2000 + yr_val
        # Cumulative loss: all losses from base_year through this year
        label = ((lossyear >= base_year_val) & (lossyear <= yr_val)).astype(np.uint8)

        if forest_mask is not None:
            label = label & forest_mask

        cumulative_labels[year] = label

        if verbose:
            change_pixels = np.sum(label)
            area_ha = change_pixels * 0.04
            print(f"  2018-{year}: {change_pixels:,} cumulative deforestation "
                  f"({area_ha:,.0f} ha)")

    return cumulative_labels


def create_bitemporal_change_label(lossyear, year1, year2,
                                    treecover2000=None,
                                    treecover_threshold=DEFAULT_TREECOVER_THRESHOLD):
    """
    Create binary change label for a specific year pair.

    Args:
        lossyear: (H, W) Hansen lossyear array
        year1: Start year (e.g., 2018)
        year2: End year (e.g., 2019)
        treecover2000: Optional tree cover mask
        treecover_threshold: Minimum tree cover %

    Returns:
        (H, W) binary label: 1 = deforestation between year1 and year2
    """
    yr1_val = year1 - 2000
    yr2_val = year2 - 2000

    # Loss that occurred after year1 and up to (including) year2
    label = ((lossyear > yr1_val) & (lossyear <= yr2_val)).astype(np.uint8)

    if treecover2000 is not None:
        forest_mask = treecover2000 >= treecover_threshold
        label = label & forest_mask

    return label


def extract_bitemporal_patches(t1_data, t2_data, labels, patch_size=32,
                                stride=None, max_patches=None,
                                balance_ratio=3.0, random_state=42,
                                verbose=True):
    """
    Extract paired patches for Siamese network training.

    Args:
        t1_data: (C, H, W) features at time 1
        t2_data: (C, H, W) features at time 2
        labels: (H, W) binary change labels
        patch_size: Size of square patches
        stride: Step between patches (default: patch_size // 2)
        max_patches: Maximum total patches to extract
        balance_ratio: No-change to change ratio (e.g., 3.0 = 3:1)
        random_state: Random seed
        verbose: Print progress

    Returns:
        dict with keys:
            'patches_t1': (N, C, patch_size, patch_size)
            'patches_t2': (N, C, patch_size, patch_size)
            'labels': (N,) binary labels
    """
    if verbose:
        print(f"Extracting bi-temporal patches (size={patch_size})...")

    if stride is None:
        stride = patch_size // 2

    C, H, W = t1_data.shape
    half = patch_size // 2

    # Collect patch center coordinates
    change_centers = []
    nochange_centers = []

    for i in range(half, H - half, stride):
        for j in range(half, W - half, stride):
            patch_labels = labels[i - half:i + half, j - half:j + half]
            t1_patch = t1_data[:, i - half:i + half, j - half:j + half]

            # Skip patches with NaN/Inf
            if np.any(np.isnan(t1_patch)) or np.any(np.isinf(t1_patch)):
                continue

            # Classify patch by majority label
            change_fraction = np.mean(patch_labels)

            if change_fraction >= 0.3:  # At least 30% change
                change_centers.append((i, j))
            elif change_fraction == 0:  # Pure no-change
                nochange_centers.append((i, j))

    if verbose:
        print(f"  Change patches found: {len(change_centers):,}")
        print(f"  No-change patches found: {len(nochange_centers):,}")

    # Balance classes
    rng = np.random.RandomState(random_state)
    n_change = len(change_centers)

    if n_change == 0:
        if verbose:
            print("  WARNING: No change patches found!")
        return {'patches_t1': np.array([]), 'patches_t2': np.array([]),
                'labels': np.array([])}

    n_nochange = min(int(n_change * balance_ratio), len(nochange_centers))
    nochange_idx = rng.choice(len(nochange_centers), n_nochange, replace=False)
    selected_nochange = [nochange_centers[i] for i in nochange_idx]

    all_centers = change_centers + selected_nochange
    all_labels = [1] * n_change + [0] * n_nochange

    if max_patches and len(all_centers) > max_patches:
        idx = rng.choice(len(all_centers), max_patches, replace=False)
        all_centers = [all_centers[i] for i in idx]
        all_labels = [all_labels[i] for i in idx]

    # Extract patches
    patches_t1 = []
    patches_t2 = []

    for (ci, cj) in all_centers:
        p1 = t1_data[:, ci - half:ci + half, cj - half:cj + half]
        p2 = t2_data[:, ci - half:ci + half, cj - half:cj + half]
        patches_t1.append(p1)
        patches_t2.append(p2)

    result = {
        'patches_t1': np.array(patches_t1, dtype=np.float32),
        'patches_t2': np.array(patches_t2, dtype=np.float32),
        'labels': np.array(all_labels, dtype=np.int64),
    }

    if verbose:
        print(f"  Total patches: {len(all_labels):,}")
        print(f"    Change: {sum(all_labels):,}")
        print(f"    No-change: {len(all_labels) - sum(all_labels):,}")
        print(f"  Patch shape: ({C}, {patch_size}, {patch_size})")

    return result


def prepare_pixel_training_data(t1_features, t2_features, labels,
                                 sample_size=None, balance_ratio=3.0,
                                 random_state=42, verbose=True):
    """
    Prepare pixel-level training data for RF change detection.

    Args:
        t1_features: (23, H, W) features at time 1
        t2_features: (23, H, W) features at time 2
        labels: (H, W) binary change labels
        sample_size: Max total samples
        balance_ratio: No-change to change ratio
        random_state: Random seed
        verbose: Print info

    Returns:
        tuple: (X, y) where X is (N, 56) and y is (N,)
    """
    from .feature_engineering import create_stacked_features

    if verbose:
        print("Preparing pixel-level training data...")

    stacked = create_stacked_features(t1_features, t2_features, verbose=False)
    n_features = stacked.shape[0]

    X = stacked.reshape(n_features, -1).T
    y = labels.flatten()

    # Remove invalid pixels
    valid_mask = ~np.any(np.isnan(X), axis=1) & ~np.any(np.isinf(X), axis=1)
    X = X[valid_mask]
    y = y[valid_mask]

    # Balance classes
    rng = np.random.RandomState(random_state)
    change_idx = np.where(y == 1)[0]
    nochange_idx = np.where(y == 0)[0]

    n_change = len(change_idx)
    n_nochange = min(int(n_change * balance_ratio), len(nochange_idx))

    if n_change == 0:
        if verbose:
            print("  WARNING: No change samples found!")
        return X[:0], y[:0]

    selected_nochange = rng.choice(nochange_idx, n_nochange, replace=False)
    selected_idx = np.concatenate([change_idx, selected_nochange])
    rng.shuffle(selected_idx)

    X = X[selected_idx]
    y = y[selected_idx]

    if sample_size and len(y) > sample_size:
        idx = rng.choice(len(y), sample_size, replace=False)
        X = X[idx]
        y = y[idx]

    if verbose:
        print(f"  Total samples: {len(y):,}")
        print(f"    Change: {np.sum(y == 1):,}")
        print(f"    No-change: {np.sum(y == 0):,}")

    return X, y.astype(int)


def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and test sets with stratification.

    Args:
        X: Feature array
        y: Label array
        test_size: Proportion for test set
        random_state: Random seed

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def split_patches_train_test(patches_dict, test_size=0.2, random_state=42):
    """
    Split patch data into training and test sets.

    Args:
        patches_dict: Dict from extract_bitemporal_patches
        test_size: Proportion for test set
        random_state: Random seed

    Returns:
        tuple: (train_dict, test_dict)
    """
    n = len(patches_dict['labels'])
    indices = np.arange(n)

    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state,
        stratify=patches_dict['labels']
    )

    train_dict = {
        'patches_t1': patches_dict['patches_t1'][train_idx],
        'patches_t2': patches_dict['patches_t2'][train_idx],
        'labels': patches_dict['labels'][train_idx],
    }
    test_dict = {
        'patches_t1': patches_dict['patches_t1'][test_idx],
        'patches_t2': patches_dict['patches_t2'][test_idx],
        'labels': patches_dict['labels'][test_idx],
    }

    return train_dict, test_dict
