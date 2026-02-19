"""
Model Trainer Module (ML)
=========================

Handles training and evaluation of machine learning classifiers
for pixel-based change detection (Random Forest baseline).
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, cohen_kappa_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


# Change detection class names
CHANGE_CLASS_NAMES = {0: 'No Change', 1: 'Deforestation'}


def get_change_classifiers():
    """
    Get dictionary of classifiers for binary change detection.

    Returns:
        dict: Classifier name -> Pipeline
    """
    classifiers = {}

    classifiers['Random Forest'] = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=300,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        ))
    ])

    classifiers['Extra Trees'] = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()),
        ('classifier', ExtraTreesClassifier(
            n_estimators=300,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        ))
    ])

    if HAS_LIGHTGBM:
        classifiers['LightGBM'] = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler()),
            ('classifier', lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=15,
                learning_rate=0.1,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42,
                verbose=-1
            ))
        ])

    return classifiers


def train_change_model(pipeline, X_train, y_train, X_test, y_test,
                        model_name='RF', verbose=True):
    """
    Train and evaluate a single change detection model.

    Args:
        pipeline: Scikit-learn pipeline
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_name: Name for logging
        verbose: Print progress

    Returns:
        dict: Results including metrics, predictions, pipeline
    """
    if verbose:
        print(f"\n--- Training {model_name} ---")

    start_time = time.time()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    training_time = time.time() - start_time

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1_change = f1_score(y_test, y_pred, average='binary', zero_division=0)

    if verbose:
        print(f"  Time: {training_time:.2f}s")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        print(f"  F1 (change class): {f1_change:.4f}")
        print(f"  Kappa: {kappa:.4f}")
        print(f"  Precision (change): {precision:.4f}")
        print(f"  Recall (change): {recall:.4f}")

    return {
        'pipeline': pipeline,
        'y_test': y_test,
        'y_pred': y_pred,
        'training_time': training_time,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_change': f1_change,
        'kappa': kappa,
        'precision_change': precision,
        'recall_change': recall,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'report': classification_report(
            y_test, y_pred, target_names=['No Change', 'Deforestation'],
            zero_division=0
        ),
    }


def train_all_change_models(X_train, y_train, X_test, y_test, verbose=True):
    """
    Train and evaluate all available change detection classifiers.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        verbose: Print progress

    Returns:
        dict: Results for each classifier
    """
    if verbose:
        print("\n" + "=" * 60)
        print("CHANGE DETECTION MODEL TRAINING (ML)")
        print("=" * 60)
        print(f"\nTraining set: {len(y_train):,} samples")
        print(f"  Change: {np.sum(y_train == 1):,}")
        print(f"  No-change: {np.sum(y_train == 0):,}")
        print(f"Test set: {len(y_test):,} samples")

    classifiers = get_change_classifiers()
    results = {}

    for name, pipeline in classifiers.items():
        result = train_change_model(
            pipeline, X_train, y_train, X_test, y_test, name, verbose
        )
        results[name] = result

    return results


def get_feature_importance(pipeline, feature_names=None, top_n=20, verbose=True):
    """
    Extract feature importance from a trained pipeline.

    Args:
        pipeline: Trained sklearn pipeline with tree-based classifier
        feature_names: List of feature names
        top_n: Number of top features to return
        verbose: Print results

    Returns:
        list of (feature_name, importance) tuples, sorted descending
    """
    classifier = pipeline.named_steps['classifier']

    if not hasattr(classifier, 'feature_importances_'):
        if verbose:
            print("  Model does not support feature importances")
        return []

    importances = classifier.feature_importances_

    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(importances))]

    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    if verbose:
        print(f"\n  Top {min(top_n, len(ranked))} features:")
        for name, imp in ranked[:top_n]:
            print(f"    {name}: {imp:.4f}")

    return ranked[:top_n]


def get_best_model(results):
    """
    Get the best performing model from results.

    Args:
        results: Dictionary of model results

    Returns:
        tuple: (best_model_name, best_results)
    """
    best_name = max(results, key=lambda x: results[x]['f1_macro'])
    return best_name, results[best_name]
