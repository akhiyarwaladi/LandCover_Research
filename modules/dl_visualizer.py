#!/usr/bin/env python3
"""
Deep Learning Visualization Module
===================================

Visualization functions for deep learning land cover classification results.

Functions:
    - plot_training_curves: Training loss and accuracy
    - plot_confusion_matrix: Confusion matrix heatmap
    - plot_model_comparison: Compare multiple models
    - plot_spatial_predictions: Spatial prediction maps
    - generate_all_visualizations: Complete visualization pipeline

Author: Claude Sonnet 4.5
Date: 2026-01-03
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix
import os
import warnings
warnings.filterwarnings('ignore')


# Default color scheme (bright colors for Jambi)
DEFAULT_COLORS = {
    0: '#0066CC',  # Water - Bright Blue
    1: '#228B22',  # Trees/Forest - Forest Green
    2: '#90EE90',  # Crops - Light Green
    3: '#FF8C00',  # Shrub - Dark Orange
    4: '#FF1493',  # Built - Deep Pink/Magenta
    5: '#D2691E',  # Bare Ground - Chocolate Brown
}

DEFAULT_CLASS_NAMES = ['Water', 'Trees', 'Crops', 'Shrub', 'Built', 'Bare']


def plot_training_curves(history, save_path=None, baseline_acc=None, best_epoch=None, verbose=False):
    """
    Plot training loss and accuracy curves.

    Parameters
    ----------
    history : dict or numpy archive
        Training history with keys: train_loss, train_acc, val_loss, val_acc
    save_path : str, optional
        Path to save figure
    baseline_acc : float, optional
        Baseline accuracy to plot as horizontal line
    best_epoch : int, optional
        Best epoch to mark with vertical line
    verbose : bool, default=False
        Print save confirmation

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure

    Examples
    --------
    >>> history = np.load('training_history.npz')
    >>> fig = plot_training_curves(history, save_path='curves.png')
    """

    # Load history if path provided
    if isinstance(history, str):
        history = np.load(history)

    train_loss = history['train_loss']
    train_acc = history['train_acc']
    val_loss = history['val_loss']
    val_acc = history['val_acc']

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    epochs = range(1, len(train_loss) + 1)

    # Loss curve
    ax = axes[0]
    ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
    ax.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss')

    if best_epoch is not None:
        ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2,
                   alpha=0.5, label=f'Best Model (Epoch {best_epoch})')

    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax.set_title('Training Loss', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Accuracy curve
    ax = axes[1]
    ax.plot(epochs, np.array(train_acc) * 100, 'b-', linewidth=2, label='Training Accuracy')
    ax.plot(epochs, np.array(val_acc) * 100, 'r-', linewidth=2, label='Validation Accuracy')

    if best_epoch is not None:
        best_val_acc = val_acc[best_epoch - 1] * 100
        ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2,
                   alpha=0.5, label=f'Best Model ({best_val_acc:.2f}%)')

    if baseline_acc is not None:
        ax.axhline(y=baseline_acc, color='orange', linestyle='--', linewidth=2,
                   alpha=0.5, label=f'Baseline ({baseline_acc:.2f}%)')

    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Training Accuracy', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"✓ Saved: {save_path}")

    return fig


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None,
                         title='Confusion Matrix', accuracy=None, verbose=False):
    """
    Plot confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : list, optional
        Class names for labels
    save_path : str, optional
        Path to save figure
    title : str, default='Confusion Matrix'
        Plot title
    accuracy : float, optional
        Overall accuracy to display in title
    verbose : bool, default=False
        Print save confirmation

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure

    Examples
    --------
    >>> fig = plot_confusion_matrix(y_true, y_pred, save_path='cm.png')
    """

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Frequency'},
                ax=ax, vmin=0, vmax=1)

    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')

    if accuracy is not None:
        title = f'{title}\nAccuracy: {accuracy*100:.2f}%'

    ax.set_title(title, fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"✓ Saved: {save_path}")

    return fig


def plot_model_comparison(results_dict, save_path=None, verbose=False):
    """
    Compare performance of multiple models.

    Parameters
    ----------
    results_dict : dict
        Dictionary mapping model names to results
        e.g., {'Random Forest': {'accuracy': 0.75, 'f1_weighted': 0.74, ...},
               'ResNet50': {'accuracy': 0.80, 'f1_weighted': 0.79, ...}}
    save_path : str, optional
        Path to save figure
    verbose : bool, default=False
        Print save confirmation

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure

    Examples
    --------
    >>> results = {'RF': rf_results, 'ResNet': resnet_results}
    >>> fig = plot_model_comparison(results, save_path='comparison.png')
    """

    model_names = list(results_dict.keys())
    n_models = len(model_names)

    # Extract metrics
    accuracies = [results_dict[m]['accuracy'] * 100 for m in model_names]
    f1_weighted = [results_dict[m].get('f1_weighted', 0) * 100 for m in model_names]
    f1_macro = [results_dict[m].get('f1_macro', 0) * 100 for m in model_names]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    metrics = ['Accuracy', 'F1 (Weighted)', 'F1 (Macro)']
    x = np.arange(len(metrics))
    width = 0.8 / n_models

    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    for i, model in enumerate(model_names):
        values = [accuracies[i], f1_weighted[i], f1_macro[i]]
        offset = (i - n_models/2 + 0.5) * width

        bars = ax.bar(x + offset, values, width, label=model,
                     color=colors[i], alpha=0.8)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 100])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"✓ Saved: {save_path}")

    return fig


def plot_spatial_predictions(ground_truth, predictions, class_colors=None,
                            class_names=None, save_path=None, accuracy=None,
                            label_mapping=None, verbose=False):
    """
    Plot spatial prediction maps.

    Parameters
    ----------
    ground_truth : numpy.ndarray
        Ground truth labels (height, width)
    predictions : numpy.ndarray
        Predicted labels (height, width)
    class_colors : dict, optional
        Color mapping for classes
    class_names : list, optional
        Class names
    save_path : str, optional
        Path to save figure
    accuracy : float, optional
        Accuracy to display in title
    label_mapping : dict, optional
        Label remapping for ground truth
    verbose : bool, default=False
        Print save confirmation

    Returns
    -------
    fig : matplotlib.figure.Figure
        Created figure

    Examples
    --------
    >>> fig = plot_spatial_predictions(truth, preds, save_path='map.png')
    """

    if class_colors is None:
        class_colors = DEFAULT_COLORS

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    # Apply label mapping if provided
    if label_mapping is not None:
        gt_remapped = np.copy(ground_truth)
        for old, new in label_mapping.items():
            gt_remapped[ground_truth == old] = new
    else:
        gt_remapped = ground_truth

    # Create colormap
    n_classes = len(class_names)
    colors = [class_colors[i] for i in range(n_classes)]
    cmap = ListedColormap(colors)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Plot ground truth
    ax = axes[0]
    gt_plot = gt_remapped.astype(float)
    gt_plot[ground_truth == -1] = np.nan
    ax.imshow(gt_plot, cmap=cmap, vmin=0, vmax=n_classes-1, interpolation='nearest')
    ax.set_title('Ground Truth (KLHK 2024)', fontsize=16, fontweight='bold')
    ax.axis('off')

    # Plot predictions
    ax = axes[1]
    pred_plot = predictions.astype(float)
    pred_plot[predictions == -1] = np.nan

    title = 'Model Predictions'
    if accuracy is not None:
        title += f' (Accuracy: {accuracy*100:.2f}%)'

    ax.imshow(pred_plot, cmap=cmap, vmin=0, vmax=n_classes-1, interpolation='nearest')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')

    # Add legend
    legend_elements = [Patch(facecolor=class_colors[i], label=class_names[i])
                      for i in range(n_classes)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=n_classes,
              fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"✓ Saved: {save_path}")

    return fig


def generate_all_visualizations(training_history_path, test_results_path,
                                predictions_path, ground_truth, output_dir,
                                baseline_results=None, label_mapping=None,
                                class_names=None, class_colors=None, verbose=False):
    """
    Generate complete set of visualizations.

    This is the main function that creates all standard visualizations.

    Parameters
    ----------
    training_history_path : str
        Path to training history (.npz file)
    test_results_path : str
        Path to test results (.npz file)
    predictions_path : str or numpy.ndarray
        Path to predictions or prediction array
    ground_truth : numpy.ndarray
        Ground truth labels
    output_dir : str
        Directory to save visualizations
    baseline_results : dict, optional
        Baseline model results for comparison
    label_mapping : dict, optional
        Label remapping dictionary
    class_names : list, optional
        Class names
    class_colors : dict, optional
        Class colors
    verbose : bool, default=False
        Print progress

    Returns
    -------
    None

    Examples
    --------
    >>> generate_all_visualizations(
    ...     'results/training_history.npz',
    ...     'results/test_results.npz',
    ...     'results/predictions.npy',
    ...     ground_truth_array,
    ...     'results/visualizations'
    ... )
    """

    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)

    # Load data
    if verbose:
        print("\nLoading data...")

    history = np.load(training_history_path)
    test_data = np.load(test_results_path)
    y_true = test_data['targets']
    y_pred = test_data['predictions']

    if isinstance(predictions_path, str):
        predictions = np.load(predictions_path)
    else:
        predictions = predictions_path

    # Calculate best epoch
    val_acc = history['val_acc']
    best_epoch = np.argmax(val_acc) + 1
    best_val_acc = val_acc[best_epoch - 1]

    # Calculate test accuracy
    test_acc = (y_true == y_pred).mean()

    # 1. Training curves
    if verbose:
        print("\n1. Plotting training curves...")

    baseline_acc = baseline_results['accuracy'] * 100 if baseline_results else None

    plot_training_curves(
        history,
        save_path=os.path.join(output_dir, 'training_curves.png'),
        baseline_acc=baseline_acc,
        best_epoch=best_epoch,
        verbose=verbose
    )

    # 2. Confusion matrix
    if verbose:
        print("2. Plotting confusion matrix...")

    plot_confusion_matrix(
        y_true, y_pred,
        class_names=class_names,
        save_path=os.path.join(output_dir, 'confusion_matrix.png'),
        accuracy=test_acc,
        verbose=verbose
    )

    # 3. Model comparison (if baseline provided)
    if baseline_results:
        if verbose:
            print("3. Plotting model comparison...")

        # Calculate F1 scores
        from sklearn.metrics import f1_score
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')

        results_dict = {
            'Random Forest': baseline_results,
            'ResNet50': {
                'accuracy': test_acc,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted
            }
        }

        plot_model_comparison(
            results_dict,
            save_path=os.path.join(output_dir, 'model_comparison.png'),
            verbose=verbose
        )

    # 4. Spatial predictions
    if verbose:
        print("4. Plotting spatial predictions...")

    # Calculate spatial accuracy (without torch dependency)
    if label_mapping is not None:
        gt_remapped = np.copy(ground_truth)
        for old, new in label_mapping.items():
            gt_remapped[ground_truth == old] = new
    else:
        gt_remapped = ground_truth

    valid_mask = (predictions != -1) & (ground_truth != -1)
    n_valid = valid_mask.sum()
    spatial_acc = (predictions[valid_mask] == gt_remapped[valid_mask]).sum() / n_valid if n_valid > 0 else 0.0

    plot_spatial_predictions(
        ground_truth, predictions,
        class_colors=class_colors,
        class_names=class_names,
        save_path=os.path.join(output_dir, 'spatial_predictions.png'),
        accuracy=spatial_acc,
        label_mapping=label_mapping,
        verbose=verbose
    )

    if verbose:
        print("\n" + "="*80)
        print("VISUALIZATION COMPLETE!")
        print("="*80)
        print(f"\n✅ All visualizations saved to: {output_dir}/")
        print(f"\nGenerated files:")
        print(f"  - training_curves.png")
        print(f"  - confusion_matrix.png")
        if baseline_results:
            print(f"  - model_comparison.png")
        print(f"  - spatial_predictions.png")
