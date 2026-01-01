"""
Visualizer Module
=================

Generates visualizations and reports for classification results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from .data_loader import CLASS_NAMES


def plot_classifier_comparison(results, save_dir='results', verbose=True):
    """
    Create bar chart comparing classifier performance.

    Args:
        results: Dictionary of classifier results
        save_dir: Directory to save plot
        verbose: Print save location

    Returns:
        Path to saved plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = list(results.keys())
    accuracies = [results[n]['accuracy'] for n in names]
    f1_scores = [results[n]['f1_macro'] for n in names]
    times = [results[n]['training_time'] for n in names]

    # Accuracy/F1
    x = np.arange(len(names))
    width = 0.35

    axes[0].bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
    axes[0].bar(x + width/2, f1_scores, width, label='F1 (macro)', color='coral')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Classifier Performance Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', alpha=0.3)

    # Training time
    axes[1].bar(names, times, color='green', alpha=0.7)
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_title('Training Time')
    axes[1].set_xticklabels(names, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_path = f'{save_dir}/classifier_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"Saved: {output_path}")

    return output_path


def plot_confusion_matrix(y_test, y_pred, model_name, save_dir='results', verbose=True):
    """
    Create confusion matrix heatmap.

    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_dir: Directory to save plot
        verbose: Print save location

    Returns:
        Path to saved plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Get class labels
    classes_in_data = sorted(set(y_test) | set(y_pred))
    class_labels = [CLASS_NAMES.get(c, str(c)) for c in classes_in_data]

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}\n(Normalized by True Labels)')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    output_path = f'{save_dir}/confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"Saved: {output_path}")

    return output_path


def plot_feature_importance(pipeline, model_name, feature_names, save_dir='results', verbose=True):
    """
    Plot feature importance for tree-based models.

    Args:
        pipeline: Trained sklearn pipeline
        model_name: Name of the model
        feature_names: List of feature names
        save_dir: Directory to save plot
        verbose: Print save location

    Returns:
        Path to saved plot or None if not applicable
    """
    clf = pipeline.named_steps.get('classifier')

    if not hasattr(clf, 'feature_importances_'):
        return None

    fig, ax = plt.subplots(figsize=(10, 8))

    importance = clf.feature_importances_
    indices = np.argsort(importance)

    ax.barh(range(len(indices)), importance[indices], color='steelblue')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(f'Feature Importance - {model_name}')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    output_path = f'{save_dir}/feature_importance_{model_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"Saved: {output_path}")

    return output_path


def export_results_to_csv(results, save_path='results/classification_results.csv', verbose=True):
    """
    Export classification results to CSV.

    Args:
        results: Dictionary of classifier results
        save_path: Path to save CSV file
        verbose: Print save location

    Returns:
        pandas DataFrame with results
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data = []
    for name, result in results.items():
        data.append({
            'Classifier': name,
            'Accuracy': result['accuracy'],
            'F1_Macro': result['f1_macro'],
            'F1_Weighted': result['f1_weighted'],
            'Training_Time_Seconds': result['training_time']
        })

    df = pd.DataFrame(data)
    df = df.sort_values('F1_Macro', ascending=False)
    df.to_csv(save_path, index=False)

    if verbose:
        print(f"Saved: {save_path}")

    return df


def generate_all_plots(results, feature_names, save_dir='results', verbose=True):
    """
    Generate all visualization plots.

    Args:
        results: Dictionary of classifier results
        feature_names: List of feature names
        save_dir: Directory to save plots
        verbose: Print progress

    Returns:
        List of saved plot paths
    """
    os.makedirs(save_dir, exist_ok=True)

    saved_plots = []

    if verbose:
        print("\nGenerating visualizations...")

    # Classifier comparison
    path = plot_classifier_comparison(results, save_dir, verbose)
    saved_plots.append(path)

    # Best model confusion matrix
    best_name = max(results, key=lambda x: results[x]['f1_macro'])
    best_result = results[best_name]

    path = plot_confusion_matrix(
        best_result['y_test'],
        best_result['y_pred'],
        best_name,
        save_dir,
        verbose
    )
    saved_plots.append(path)

    # Feature importance plots
    for name, result in results.items():
        path = plot_feature_importance(
            result['pipeline'],
            name,
            feature_names,
            save_dir,
            verbose
        )
        if path:
            saved_plots.append(path)

    return saved_plots


def plot_aoa_map(aoa_map, di_map, threshold, save_dir='results', verbose=True):
    """
    Plot Area of Applicability (AOA) map.

    Args:
        aoa_map: Binary AOA map (1=inside, 0=outside)
        di_map: Dissimilarity Index map
        threshold: AOA threshold value
        save_dir: Directory to save plot
        verbose: Print save location

    Returns:
        Path to saved plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # AOA Binary Map
    im1 = axes[0].imshow(aoa_map, cmap='RdYlGn', interpolation='nearest')
    axes[0].set_title('Area of Applicability (AOA)\nGreen=Inside AOA, Red=Outside AOA')
    axes[0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('AOA (1=Inside, 0=Outside)')

    # Dissimilarity Index Map
    im2 = axes[1].imshow(di_map, cmap='YlOrRd', interpolation='nearest')
    axes[1].set_title(f'Dissimilarity Index (DI)\nThreshold={threshold:.3f}')
    axes[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('Dissimilarity Index')

    # Add statistics text
    pct_inside = (aoa_map.sum() / aoa_map.size) * 100
    stats_text = f"Inside AOA: {pct_inside:.1f}%\nOutside AOA: {100-pct_inside:.1f}%"
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    output_path = f'{save_dir}/aoa_map.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"Saved: {output_path}")

    return output_path


def plot_di_distribution(DI_train, DI_predict, threshold, save_dir='results', verbose=True):
    """
    Plot distribution of Dissimilarity Index for training and prediction data.

    Args:
        DI_train: Dissimilarity index for training data
        DI_predict: Dissimilarity index for prediction data
        threshold: AOA threshold value
        save_dir: Directory to save plot
        verbose: Print save location

    Returns:
        Path to saved plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histograms
    ax.hist(DI_train, bins=50, alpha=0.6, label='Training Data (CV)', color='blue', density=True)
    ax.hist(DI_predict, bins=50, alpha=0.6, label='Prediction Data', color='orange', density=True)

    # Threshold line
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2,
               label=f'AOA Threshold = {threshold:.3f}')

    ax.set_xlabel('Dissimilarity Index (DI)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Dissimilarity Index\nMeyer & Pebesma (2021) AOA Method')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_path = f'{save_dir}/di_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"Saved: {output_path}")

    return output_path


def plot_classification_with_aoa(classification_map, aoa_map, save_dir='results', verbose=True):
    """
    Plot classification results with AOA overlay.

    Args:
        classification_map: Classification prediction map
        aoa_map: Binary AOA map
        save_dir: Directory to save plot
        verbose: Print save location

    Returns:
        Path to saved plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Classification map
    im1 = axes[0].imshow(classification_map, cmap='tab10', interpolation='nearest')
    axes[0].set_title('Land Cover Classification')
    axes[0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label('Class')

    # AOA map
    im2 = axes[1].imshow(aoa_map, cmap='RdYlGn', interpolation='nearest')
    axes[1].set_title('Area of Applicability (AOA)')
    axes[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label('AOA (1=Inside, 0=Outside)')

    # Classification with AOA mask (set outside AOA to NaN for transparency)
    classification_masked = classification_map.copy().astype(float)
    classification_masked[aoa_map == 0] = np.nan

    im3 = axes[2].imshow(classification_masked, cmap='tab10', interpolation='nearest')
    axes[2].set_title('Classification (Inside AOA Only)')
    axes[2].axis('off')
    cbar3 = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    cbar3.set_label('Class')

    plt.tight_layout()

    output_path = f'{save_dir}/classification_with_aoa.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"Saved: {output_path}")

    return output_path


def plot_aoa_statistics(aoa_map, classification_map, save_dir='results', verbose=True):
    """
    Plot AOA statistics by class.

    Args:
        aoa_map: Binary AOA map
        classification_map: Classification prediction map
        save_dir: Directory to save plot
        verbose: Print save location

    Returns:
        Path to saved plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate percentage inside AOA for each class
    classes = sorted(np.unique(classification_map))
    pcts_inside = []
    class_labels = []

    for cls in classes:
        mask = (classification_map == cls)
        total_pixels = mask.sum()
        inside_aoa = (mask & (aoa_map == 1)).sum()

        if total_pixels > 0:
            pct = (inside_aoa / total_pixels) * 100
            pcts_inside.append(pct)
            class_labels.append(CLASS_NAMES.get(cls, f'Class {cls}'))

    # Bar plot
    x = np.arange(len(class_labels))
    bars = ax.bar(x, pcts_inside, color='steelblue')

    # Color bars based on percentage
    for i, (bar, pct) in enumerate(zip(bars, pcts_inside)):
        if pct >= 80:
            bar.set_color('green')
        elif pct >= 60:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    ax.set_ylabel('% Inside AOA')
    ax.set_title('Percentage of Predictions Inside AOA by Land Cover Class')
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=45, ha='right')
    ax.set_ylim(0, 100)
    ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='80% threshold')
    ax.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='60% threshold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    # Add value labels on bars
    for i, (bar, pct) in enumerate(zip(bars, pcts_inside)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    output_path = f'{save_dir}/aoa_by_class.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    if verbose:
        print(f"Saved: {output_path}")

    return output_path
