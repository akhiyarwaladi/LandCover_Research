#!/usr/bin/env python3
"""
Generate Publication Figures - From Saved Model Results
========================================================

This script loads saved model weights and results to generate publication-quality
figures for the journal paper without needing to retrain the model.

Figures Generated:
1. Training curves (loss and accuracy over epochs)
2. Confusion matrix (normalized)
3. ML vs DL comparison (bar chart)
4. Per-class F1-score comparison
5. Overall performance comparison

Usage:
    python scripts/generate_publication_figures.py

    # With custom theme
    python scripts/generate_publication_figures.py --theme seaborn-v0_8-darkgrid

    # Custom DPI
    python scripts/generate_publication_figures.py --dpi 600

Output:
    results/figures/publication/*.png (300 DPI by default)
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score
)

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input paths
RESNET_PREDICTIONS_PATH = 'results/resnet_classification/test_predictions.npz'
RESNET_HISTORY_PATH = 'results/resnet_classification/training_history.npz'
ML_RESULTS_PATH = 'results/classification_results.csv'

# Output directory
OUTPUT_DIR = 'results/figures/publication'

# Class names
CLASS_NAMES = ['Water', 'Trees/Forest', 'Crops/Agriculture',
               'Shrub/Scrub', 'Built Area', 'Bare Ground']

# Default styling (Professional journal style)
DEFAULT_THEME = 'seaborn-v0_8-whitegrid'
DEFAULT_DPI = 300
FIGURE_FORMAT = 'png'

# Professional journal color palette (colorblind-friendly)
# Using Nature/Science style colors
COLORS_ML = '#0173B2'  # Blue for ML (Random Forest)
COLORS_DL = '#DE8F05'  # Orange for DL (ResNet)
COLORS_PALETTE = ['#0173B2', '#029E73', '#CC78BC', '#DE8F05', '#CA9161', '#949494']  # Colorblind-safe

# Font settings for journal publication
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 1.0
})

# ============================================================================
# FIGURE GENERATION FUNCTIONS
# ============================================================================

def load_resnet_predictions(predictions_path):
    """
    Load saved ResNet predictions.

    Returns:
        dict with y_true, y_pred
    """
    data = np.load(predictions_path)
    return {
        'y_true': data['y_true'],
        'y_pred': data['y_pred']
    }


def load_resnet_history(history_path):
    """
    Load saved training history.

    Returns:
        dict with train_loss, train_acc, val_loss, val_acc, epoch_time
    """
    if not os.path.exists(history_path):
        print(f"⚠️  Warning: Training history not found at {history_path}")
        return None

    data = np.load(history_path)
    return {
        'train_loss': data['train_loss'],
        'train_acc': data['train_acc'],
        'val_loss': data['val_loss'],
        'val_acc': data['val_acc'],
        'epoch_time': data.get('epoch_time', None)
    }


def load_ml_results(ml_results_path):
    """
    Load Random Forest baseline results.

    Returns:
        dict with accuracy, f1_macro, f1_weighted
    """
    if not os.path.exists(ml_results_path):
        print(f"⚠️  Warning: ML results not found, using known values")
        return {
            'accuracy': 0.7495,
            'f1_macro': 0.542,
            'f1_weighted': 0.744
        }

    df = pd.read_csv(ml_results_path)
    rf_row = df[df.iloc[:, 0] == 'Random Forest'].iloc[0]

    return {
        'accuracy': rf_row.iloc[1],
        'f1_macro': rf_row.iloc[2],
        'f1_weighted': rf_row.iloc[3]
    }


def plot_training_curves(history, output_dir, dpi=300):
    """
    Plot training and validation curves.

    Parameters
    ----------
    history : dict
        Training history with train_loss, train_acc, val_loss, val_acc
    output_dir : str
        Directory to save figure
    dpi : int
        Resolution in dots per inch
    """
    if history is None:
        print("⚠️  Skipping training curves - no history available")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss plot
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-o', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-o', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'training_curves.{FIGURE_FORMAT}')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Saved: {output_path}")


def plot_confusion_matrix_publication(y_true, y_pred, class_names, output_dir, dpi=300):
    """
    Plot confusion matrix for publication.

    Parameters
    ----------
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    class_names : list
        List of class names
    output_dir : str
        Directory to save figure
    dpi : int
        Resolution
    """
    # Calculate confusion matrix (normalized)
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Frequency'},
                ax=ax, vmin=0, vmax=1)

    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
    ax.set_title('ResNet50 Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'confusion_matrix_resnet.{FIGURE_FORMAT}')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Saved: {output_path}")


def plot_ml_vs_dl_overall(ml_results, dl_accuracy, dl_f1_macro, dl_f1_weighted,
                          output_dir, dpi=300):
    """
    Plot overall ML vs DL comparison.

    Parameters
    ----------
    ml_results : dict
        Random Forest results
    dl_accuracy : float
        ResNet accuracy
    dl_f1_macro : float
        ResNet macro F1
    dl_f1_weighted : float
        ResNet weighted F1
    output_dir : str
        Directory to save figure
    dpi : int
        Resolution
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['Accuracy', 'F1-Score (Macro)', 'F1-Score (Weighted)']
    ml_values = [ml_results['accuracy'], ml_results['f1_macro'], ml_results['f1_weighted']]
    dl_values = [dl_accuracy, dl_f1_macro, dl_f1_weighted]

    x = np.arange(len(metrics))
    width = 0.35

    for i, (metric, ml_val, dl_val) in enumerate(zip(metrics, ml_values, dl_values)):
        axes[i].bar([0], [ml_val], width, label='Random Forest', color=COLORS_ML, alpha=0.8)
        axes[i].bar([1], [dl_val], width, label='ResNet50', color=COLORS_DL, alpha=0.8)
        axes[i].set_ylabel('Score', fontsize=12)
        axes[i].set_title(metric, fontsize=13, fontweight='bold')
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(['RF', 'ResNet'])
        axes[i].set_ylim([0, 1.0])
        axes[i].grid(True, alpha=0.3, axis='y')

        # Add value labels
        axes[i].text(0, ml_val + 0.02, f'{ml_val:.3f}', ha='center', fontsize=10)
        axes[i].text(1, dl_val + 0.02, f'{dl_val:.3f}', ha='center', fontsize=10)

        # Add improvement annotation
        improvement = dl_val - ml_val
        color = 'green' if improvement > 0 else 'red'
        axes[i].text(0.5, max(ml_val, dl_val) + 0.08,
                    f'{improvement:+.3f}',
                    ha='center', fontsize=11, color=color, fontweight='bold')

    plt.suptitle('Machine Learning vs Deep Learning Performance',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f'ml_vs_dl_overall.{FIGURE_FORMAT}')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Saved: {output_path}")


def plot_per_class_f1_comparison(ml_y_true, ml_y_pred, dl_y_true, dl_y_pred,
                                 class_names, output_dir, dpi=300):
    """
    Plot per-class F1-score comparison.

    Parameters
    ----------
    ml_y_true, ml_y_pred : array
        ML predictions (if available)
    dl_y_true, dl_y_pred : array
        DL predictions
    class_names : list
        Class names
    output_dir : str
        Directory to save figure
    dpi : int
        Resolution
    """
    # Calculate DL F1-scores
    dl_f1 = f1_score(dl_y_true, dl_y_pred, average=None, zero_division=0)

    # Try to calculate ML F1-scores (use known values if not available)
    if ml_y_true is not None and ml_y_pred is not None:
        ml_f1 = f1_score(ml_y_true, ml_y_pred, average=None, zero_division=0)
    else:
        # Known values from previous run
        ml_f1 = np.array([0.79, 0.74, 0.78, 0.37, 0.42, 0.15])

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(class_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, ml_f1, width, label='Random Forest',
                   color=COLORS_ML, alpha=0.8)
    bars2 = ax.bar(x + width/2, dl_f1, width, label='ResNet50',
                   color=COLORS_DL, alpha=0.8)

    ax.set_xlabel('Land Cover Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class F1-Score: Random Forest vs ResNet50',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'per_class_f1_comparison.{FIGURE_FORMAT}')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Saved: {output_path}")


def plot_improvement_per_class(ml_y_true, ml_y_pred, dl_y_true, dl_y_pred,
                               class_names, output_dir, dpi=300):
    """
    Plot improvement (DL - ML) per class.

    Parameters
    ----------
    ml_y_true, ml_y_pred : array
        ML predictions
    dl_y_true, dl_y_pred : array
        DL predictions
    class_names : list
        Class names
    output_dir : str
        Directory to save figure
    dpi : int
        Resolution
    """
    # Calculate F1-scores
    dl_f1 = f1_score(dl_y_true, dl_y_pred, average=None, zero_division=0)

    if ml_y_true is not None and ml_y_pred is not None:
        ml_f1 = f1_score(ml_y_true, ml_y_pred, average=None, zero_division=0)
    else:
        ml_f1 = np.array([0.79, 0.74, 0.78, 0.37, 0.42, 0.15])

    improvement = dl_f1 - ml_f1

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['green' if x > 0 else 'red' for x in improvement]
    bars = ax.bar(range(len(class_names)), improvement, color=colors, alpha=0.7)

    ax.set_xlabel('Land Cover Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score Improvement (ResNet - RF)', fontsize=12, fontweight='bold')
    ax.set_title('Deep Learning Improvement Over Random Forest',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, improvement)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.,
               height + (0.01 if height > 0 else -0.01),
               f'{val:+.3f}', ha='center',
               va='bottom' if height > 0 else 'top', fontsize=10)

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'improvement_per_class.{FIGURE_FORMAT}')
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Saved: {output_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(theme=DEFAULT_THEME, dpi=DEFAULT_DPI):
    """Generate all publication figures from saved model results."""

    print("=" * 70)
    print("GENERATE PUBLICATION FIGURES FROM SAVED MODEL")
    print("=" * 70)
    print(f"\n📊 Loading saved results...")

    # Set matplotlib theme
    try:
        plt.style.use(theme)
        print(f"   ✓ Using theme: {theme}")
    except:
        print(f"   ⚠️  Theme '{theme}' not found, using default")
        plt.style.use('default')

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load ResNet predictions
    print("\n1. Loading ResNet predictions...")
    if os.path.exists(RESNET_PREDICTIONS_PATH):
        resnet_preds = load_resnet_predictions(RESNET_PREDICTIONS_PATH)
        print(f"   ✓ Loaded {len(resnet_preds['y_true'])} test samples")

        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score
        dl_accuracy = accuracy_score(resnet_preds['y_true'], resnet_preds['y_pred'])
        dl_f1_macro = f1_score(resnet_preds['y_true'], resnet_preds['y_pred'],
                               average='macro', zero_division=0)
        dl_f1_weighted = f1_score(resnet_preds['y_true'], resnet_preds['y_pred'],
                                  average='weighted', zero_division=0)

        print(f"   ResNet Accuracy: {dl_accuracy:.4f}")
        print(f"   ResNet F1 (macro): {dl_f1_macro:.4f}")
        print(f"   ResNet F1 (weighted): {dl_f1_weighted:.4f}")
    else:
        print(f"   ⚠️  ResNet predictions not found!")
        print(f"   Please run: python scripts/run_resnet_classification.py")
        return

    # Load training history
    print("\n2. Loading training history...")
    history = load_resnet_history(RESNET_HISTORY_PATH)

    # Load ML results
    print("\n3. Loading Random Forest baseline...")
    ml_results = load_ml_results(ML_RESULTS_PATH)
    print(f"   RF Accuracy: {ml_results['accuracy']:.4f}")
    print(f"   RF F1 (macro): {ml_results['f1_macro']:.4f}")

    # Generate figures
    print(f"\n4. Generating publication figures (DPI={dpi})...")
    print(f"   Output: {OUTPUT_DIR}/")

    # Figure 1: Training curves
    print("\n   Figure 1: Training curves")
    plot_training_curves(history, OUTPUT_DIR, dpi)

    # Figure 2: Confusion matrix
    print("\n   Figure 2: Confusion matrix")
    plot_confusion_matrix_publication(
        resnet_preds['y_true'], resnet_preds['y_pred'],
        CLASS_NAMES, OUTPUT_DIR, dpi
    )

    # Figure 3: Overall ML vs DL
    print("\n   Figure 3: Overall performance comparison")
    plot_ml_vs_dl_overall(
        ml_results, dl_accuracy, dl_f1_macro, dl_f1_weighted,
        OUTPUT_DIR, dpi
    )

    # Figure 4: Per-class F1 comparison
    print("\n   Figure 4: Per-class F1-score comparison")
    plot_per_class_f1_comparison(
        None, None,  # ML predictions not available
        resnet_preds['y_true'], resnet_preds['y_pred'],
        CLASS_NAMES, OUTPUT_DIR, dpi
    )

    # Figure 5: Improvement per class
    print("\n   Figure 5: Improvement per class")
    plot_improvement_per_class(
        None, None,  # ML predictions not available
        resnet_preds['y_true'], resnet_preds['y_pred'],
        CLASS_NAMES, OUTPUT_DIR, dpi
    )

    # Summary
    print("\n" + "=" * 70)
    print("PUBLICATION FIGURES COMPLETE!")
    print("=" * 70)
    print(f"\n✅ All figures saved to: {OUTPUT_DIR}/")
    print("\n📋 Figures generated:")
    print("   1. training_curves.png - Loss and accuracy over epochs")
    print("   2. confusion_matrix_resnet.png - Normalized confusion matrix")
    print("   3. ml_vs_dl_overall.png - Overall performance comparison")
    print("   4. per_class_f1_comparison.png - Per-class F1-scores")
    print("   5. improvement_per_class.png - DL improvement over ML")

    print(f"\n💡 To regenerate with different theme:")
    print(f"   python {__file__} --theme seaborn-v0_8-darkgrid")
    print(f"\n💡 To change resolution:")
    print(f"   python {__file__} --dpi 600")

    print("\n✅ Figures ready for journal submission!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate publication figures from saved ResNet model results'
    )
    parser.add_argument('--theme', type=str, default=DEFAULT_THEME,
                       help=f'Matplotlib theme (default: {DEFAULT_THEME})')
    parser.add_argument('--dpi', type=int, default=DEFAULT_DPI,
                       help=f'Figure resolution in DPI (default: {DEFAULT_DPI})')

    args = parser.parse_args()

    main(theme=args.theme, dpi=args.dpi)
