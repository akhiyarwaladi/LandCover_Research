#!/usr/bin/env python3
"""
AOA (Area of Applicability) Calculator Module
==============================================

Implements the Area of Applicability methodology from Meyer & Pebesma (2021)
for assessing the reliability of spatial predictions.

Reference:
Meyer, H., & Pebesma, E. (2021). Predicting into unknown space?
Estimating the area of applicability of spatial prediction models.
Methods in Ecology and Evolution, 12, 1620-1633.
https://doi.org/10.1111/2041-210X.13650

The AOA identifies areas where model predictions are reliable based on
dissimilarity to training data in multidimensional feature space.
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold


def calculate_dissimilarity_index(
    X_train,
    X_predict,
    feature_weights=None,
    cv_folds=10,
    remove_outliers=True,
    percentile=0.95,
    verbose=True
):
    """
    Calculate Dissimilarity Index (DI) for prediction data.

    The DI represents the normalized and weighted minimum distance to the
    nearest training data point divided by the average distance within
    the training data.

    Parameters:
    -----------
    X_train : array-like, shape (n_train_samples, n_features)
        Training data features
    X_predict : array-like, shape (n_predict_samples, n_features)
        Prediction data features
    feature_weights : array-like, shape (n_features,), optional
        Feature importance weights. If None, equal weights are used.
    cv_folds : int, default=10
        Number of cross-validation folds for threshold calculation
    remove_outliers : bool, default=True
        Whether to remove outliers when calculating threshold
    percentile : float, default=0.95
        Percentile for outlier removal (if remove_outliers=True)
    verbose : bool, default=True
        Print progress messages

    Returns:
    --------
    DI_predict : array, shape (n_predict_samples,)
        Dissimilarity index for each prediction sample
    threshold : float
        AOA threshold derived from training data
    DI_train_cv : array, shape (n_train_samples,)
        Dissimilarity index for training data (cross-validation)
    """

    if verbose:
        print("\nCalculating Area of Applicability (AOA)...")
        print(f"  Training samples: {X_train.shape[0]:,}")
        print(f"  Prediction samples: {X_predict.shape[0]:,}")
        print(f"  Features: {X_train.shape[1]}")

    # Use equal weights if not provided
    if feature_weights is None:
        feature_weights = np.ones(X_train.shape[1])
    else:
        # Normalize weights to sum to number of features
        feature_weights = np.array(feature_weights)
        feature_weights = feature_weights / feature_weights.sum() * len(feature_weights)

    if verbose:
        print(f"  Feature weights range: [{feature_weights.min():.4f}, {feature_weights.max():.4f}]")

    # Apply weights to features
    X_train_weighted = X_train * feature_weights
    X_predict_weighted = X_predict * feature_weights

    # Calculate average distance within training data
    if verbose:
        print("\n  Calculating average distance within training data...")

    # Sample a subset for efficiency (use max 5000 samples)
    n_sample = min(5000, X_train.shape[0])
    np.random.seed(42)
    sample_idx = np.random.choice(X_train.shape[0], n_sample, replace=False)
    X_train_sample = X_train_weighted[sample_idx]

    # Calculate pairwise distances
    train_distances = cdist(X_train_sample, X_train_sample, metric='euclidean')
    # Get average distance (excluding diagonal)
    mask = ~np.eye(train_distances.shape[0], dtype=bool)
    avg_train_distance = train_distances[mask].mean()

    if verbose:
        print(f"    Average training distance: {avg_train_distance:.4f}")

    # Calculate DI for training data using cross-validation
    if verbose:
        print(f"\n  Calculating DI for training data (CV folds={cv_folds})...")

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    DI_train_cv = np.zeros(X_train.shape[0])

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        # For each validation sample, find minimum distance to training samples
        X_fold_train = X_train_weighted[train_idx]
        X_fold_val = X_train_weighted[val_idx]

        # Calculate distances from validation to training
        distances = cdist(X_fold_val, X_fold_train, metric='euclidean')

        # Minimum distance for each validation sample
        min_distances = distances.min(axis=1)

        # Normalize by average training distance
        DI_train_cv[val_idx] = min_distances / avg_train_distance

    # Calculate threshold (max DI with outlier removal)
    if remove_outliers:
        threshold = np.percentile(DI_train_cv, percentile * 100)
        if verbose:
            print(f"    DI threshold (outliers removed at {percentile*100}%): {threshold:.4f}")
            print(f"    Training DI range: [{DI_train_cv.min():.4f}, {DI_train_cv.max():.4f}]")
    else:
        threshold = DI_train_cv.max()
        if verbose:
            print(f"    DI threshold (max): {threshold:.4f}")

    # Calculate DI for prediction data IN CHUNKS to avoid memory issues
    if verbose:
        print(f"\n  Calculating DI for prediction data...")

    # Process in chunks to avoid memory issues (max 100k samples per chunk)
    chunk_size = 100000
    n_chunks = int(np.ceil(X_predict.shape[0] / chunk_size))
    DI_predict = np.zeros(X_predict.shape[0])

    if verbose and n_chunks > 1:
        print(f"    Processing in {n_chunks} chunks (chunk_size={chunk_size:,})...")

    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, X_predict.shape[0])

        # Calculate distances for this chunk
        X_chunk = X_predict_weighted[start_idx:end_idx]
        distances = cdist(X_chunk, X_train_weighted, metric='euclidean')

        # Minimum distance for each prediction sample in chunk
        min_distances = distances.min(axis=1)

        # Normalize by average training distance
        DI_predict[start_idx:end_idx] = min_distances / avg_train_distance

        if verbose and n_chunks > 1:
            print(f"      Chunk {i+1}/{n_chunks} complete ({end_idx:,}/{X_predict.shape[0]:,} samples)")

    if verbose:
        print(f"    Prediction DI range: [{DI_predict.min():.4f}, {DI_predict.max():.4f}]")
        within_aoa = (DI_predict <= threshold).sum()
        pct_within = (within_aoa / len(DI_predict)) * 100
        print(f"    Samples within AOA: {within_aoa:,} ({pct_within:.2f}%)")

    return DI_predict, threshold, DI_train_cv


def calculate_aoa_map(
    features,
    X_train,
    feature_weights=None,
    cv_folds=10,
    remove_outliers=True,
    percentile=0.95,
    verbose=True
):
    """
    Calculate AOA map for full spatial prediction.

    Parameters:
    -----------
    features : array-like, shape (n_features, height, width)
        Full feature stack for spatial prediction
    X_train : array-like, shape (n_train_samples, n_features)
        Training data features
    feature_weights : array-like, shape (n_features,), optional
        Feature importance weights
    cv_folds : int, default=10
        Number of cross-validation folds
    remove_outliers : bool, default=True
        Whether to remove outliers when calculating threshold
    percentile : float, default=0.95
        Percentile for outlier removal
    verbose : bool, default=True
        Print progress messages

    Returns:
    --------
    aoa_map : array, shape (height, width)
        Binary AOA map (1=inside AOA, 0=outside AOA)
    di_map : array, shape (height, width)
        Dissimilarity index map
    threshold : float
        AOA threshold value
    """

    n_features, height, width = features.shape

    if verbose:
        print(f"\nCalculating AOA map for {height}x{width} pixels...")

    # Reshape features to (n_pixels, n_features)
    features_2d = features.reshape(n_features, -1).T

    # Calculate DI for all pixels
    DI_predict, threshold, DI_train_cv = calculate_dissimilarity_index(
        X_train,
        features_2d,
        feature_weights=feature_weights,
        cv_folds=cv_folds,
        remove_outliers=remove_outliers,
        percentile=percentile,
        verbose=verbose
    )

    # Reshape DI back to spatial dimensions
    di_map = DI_predict.reshape(height, width)

    # Create binary AOA map
    aoa_map = (di_map <= threshold).astype(np.uint8)

    if verbose:
        print(f"\nAOA Map Statistics:")
        print(f"  Inside AOA: {aoa_map.sum():,} pixels ({(aoa_map.sum() / aoa_map.size)*100:.2f}%)")
        print(f"  Outside AOA: {(aoa_map == 0).sum():,} pixels ({((aoa_map == 0).sum() / aoa_map.size)*100:.2f}%)")

    return aoa_map, di_map, threshold


def get_feature_importance_weights(pipeline, feature_names=None):
    """
    Extract feature importance weights from trained model pipeline.

    Parameters:
    -----------
    pipeline : sklearn.Pipeline
        Trained scikit-learn pipeline
    feature_names : list, optional
        List of feature names

    Returns:
    --------
    importance : array
        Feature importance weights
    """

    # Try to extract classifier from pipeline
    if hasattr(pipeline, 'named_steps'):
        if 'classifier' in pipeline.named_steps:
            classifier = pipeline.named_steps['classifier']
        else:
            # Use last step
            classifier = pipeline.steps[-1][1]
    else:
        classifier = pipeline

    # Extract feature importance
    if hasattr(classifier, 'feature_importances_'):
        # Tree-based models
        importance = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        # Linear models - use absolute coefficients
        coef = classifier.coef_
        if len(coef.shape) > 1:
            # Multi-class: average across classes
            importance = np.abs(coef).mean(axis=0)
        else:
            # Binary classification
            importance = np.abs(coef)
    else:
        # Default to equal weights
        if feature_names is not None:
            importance = np.ones(len(feature_names))
        else:
            raise ValueError("Cannot extract feature importance and no feature_names provided")

    return importance
