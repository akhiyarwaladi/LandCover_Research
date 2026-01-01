#!/usr/bin/env python3
"""
STEP 3: Classification
=======================

Loads preprocessed data, trains multiple classifiers, evaluates performance,
and generates visualizations.

Inputs:
- data/preprocessed/train_test_data.npz

Outputs:
- results/classification_results.csv
- results/classifier_comparison.png
- results/confusion_matrix_*.png
- results/feature_importance_*.png

Usage:
    python scripts/3_run_classification.py
"""

import sys
import os
import numpy as np

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.feature_engineering import get_all_feature_names
from modules.model_trainer import train_all_models, get_best_model
from modules.visualizer import generate_all_plots, export_results_to_csv


# Configuration
PREPROCESSED_DATA_PATH = 'data/preprocessed/train_test_data.npz'
OUTPUT_DIR = 'results'
INCLUDE_SLOW_MODELS = False  # Exclude XGBoost (has class label issues)


def main():
    print("=" * 70)
    print("STEP 3: CLASSIFICATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Preprocessed Data: {PREPROCESSED_DATA_PATH}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print(f"  Include Slow Models: {INCLUDE_SLOW_MODELS}")

    # Check if preprocessed data exists
    if not os.path.exists(PREPROCESSED_DATA_PATH):
        print("\n❌ ERROR: Preprocessed data not found!")
        print(f"   Expected: {PREPROCESSED_DATA_PATH}")
        print("\nPlease run preprocessing first:")
        print("  python scripts/2_preprocess_data.py")
        return 1

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------------
    # Load Preprocessed Data
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Loading Preprocessed Data...")
    print("-" * 70)

    data = np.load(PREPROCESSED_DATA_PATH)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    print(f"  Training set: {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
    print(f"  Test set: {X_test.shape[0]:,} samples × {X_test.shape[1]} features")

    # ------------------------------------------------------------------------
    # Train Classifiers
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Training Classifiers...")
    print("-" * 70)

    results = train_all_models(
        X_train, y_train,
        X_test, y_test,
        include_slow=INCLUDE_SLOW_MODELS,
        verbose=True
    )

    # ------------------------------------------------------------------------
    # Results Summary
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Export to CSV
    summary_df = export_results_to_csv(
        results,
        f'{OUTPUT_DIR}/classification_results.csv',
        verbose=True
    )

    print("\n" + summary_df.to_string(index=False))

    # Best classifier
    best_name, best_result = get_best_model(results)
    print(f"\n🏆 Best Classifier: {best_name}")
    print(f"   Accuracy: {best_result['accuracy']:.4f}")
    print(f"   F1 (macro): {best_result['f1_macro']:.4f}")
    print(f"   F1 (weighted): {best_result['f1_weighted']:.4f}")

    # Detailed report for best classifier
    print(f"\nClassification Report ({best_name}):")
    print(best_result['report'])

    # ------------------------------------------------------------------------
    # Generate Visualizations
    # ------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Generating Visualizations...")
    print("-" * 70)

    feature_names = get_all_feature_names()
    plot_paths = generate_all_plots(results, feature_names, OUTPUT_DIR, verbose=True)

    print(f"\nGenerated {len(plot_paths)} visualizations")

    # ------------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CLASSIFICATION COMPLETE!")
    print("=" * 70)
    print(f"\n✅ Results saved to: {OUTPUT_DIR}/")
    print(f"\nKey Outputs:")
    print(f"  1. classification_results.csv - Performance metrics")
    print(f"  2. classifier_comparison.png - Model comparison")
    print(f"  3. confusion_matrix_{best_name.lower().replace(' ', '_')}.png - Confusion matrix")
    print(f"  4. feature_importance_*.png - Feature importance plots")
    print(f"\nBest Model: {best_name}")
    print(f"  Accuracy: {best_result['accuracy']:.4f}")
    print(f"  F1 (macro): {best_result['f1_macro']:.4f}")
    print(f"  Training time: {best_result['training_time']:.2f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
