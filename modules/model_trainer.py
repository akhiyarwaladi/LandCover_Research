"""
Model Trainer Module
====================

Handles training and evaluation of classification models.
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Try importing optional libraries
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def get_classifiers(include_slow=True):
    """
    Get dictionary of classifiers to evaluate.

    Args:
        include_slow: Include computationally expensive classifiers

    Returns:
        dict: Classifier name -> Pipeline
    """
    classifiers = {}

    # Random Forest - Fast and accurate
    classifiers['Random Forest'] = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        ))
    ])

    # Extra Trees - Often faster than RF
    classifiers['Extra Trees'] = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()),
        ('classifier', ExtraTreesClassifier(
            n_estimators=200,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        ))
    ])

    # Logistic Regression - Fast baseline
    classifiers['Logistic Regression'] = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            multi_class='multinomial',
            max_iter=500,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        ))
    ])

    # Decision Tree - Fast and interpretable
    classifiers['Decision Tree'] = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        ))
    ])

    # Naive Bayes - Very fast
    classifiers['Naive Bayes'] = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()),
        ('classifier', GaussianNB())
    ])

    # SGD - Fast for large datasets
    classifiers['SGD'] = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler()),
        ('classifier', SGDClassifier(
            loss='modified_huber',
            penalty='l2',
            alpha=0.0001,
            max_iter=1000,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        ))
    ])

    # LightGBM (if available) - Very fast gradient boosting
    if HAS_LIGHTGBM:
        classifiers['LightGBM'] = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler()),
            ('classifier', lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=15,
                learning_rate=0.1,
                class_weight='balanced',
                n_jobs=-1,
                random_state=42,
                verbose=-1
            ))
        ])

    # XGBoost (if available)
    if HAS_XGBOOST and include_slow:
        classifiers['XGBoost'] = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler()),
            ('classifier', XGBClassifier(
                n_estimators=200,
                max_depth=15,
                learning_rate=0.1,
                n_jobs=-1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            ))
        ])

    return classifiers


def train_single_model(pipeline, X_train, y_train, X_test, y_test, model_name, verbose=True):
    """
    Train and evaluate a single model.

    Args:
        pipeline: Scikit-learn pipeline
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_name: Name of the model
        verbose: Print progress

    Returns:
        dict: Results including metrics and predictions
    """
    if verbose:
        print(f"\n--- Training {model_name} ---")

    start_time = time.time()

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    training_time = time.time() - start_time
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    if verbose:
        print(f"  Time: {training_time:.2f}s")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")

    return {
        'pipeline': pipeline,
        'y_test': y_test,
        'y_pred': y_pred,
        'training_time': training_time,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'report': classification_report(y_test, y_pred, zero_division=0)
    }


def train_all_models(X_train, y_train, X_test, y_test, include_slow=True, verbose=True):
    """
    Train and evaluate all available classifiers.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        include_slow: Include slower models
        verbose: Print progress

    Returns:
        dict: Results for each classifier
    """
    if verbose:
        print("\n" + "=" * 60)
        print("TRAINING AND EVALUATION")
        print("=" * 60)
        print(f"\nTraining set: {len(y_train):,} samples")
        print(f"Test set: {len(y_test):,} samples")

    # Get classifiers
    classifiers = get_classifiers(include_slow=include_slow)

    results = {}
    for name, pipeline in classifiers.items():
        result = train_single_model(
            pipeline, X_train, y_train, X_test, y_test, name, verbose
        )
        results[name] = result

    return results


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
