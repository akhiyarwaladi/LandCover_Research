"""
Visualizer Module
=================

Publication-quality plots for deforestation detection results.
All figures saved at 300 DPI with white background.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import seaborn as sns


# Consistent styling
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 10
DPI = 300
FIGCOLOR = 'white'

# Color scheme for change detection
CHANGE_COLORS = {
    0: '#228B22',  # No change (forest green)
    1: '#FF0000',  # Deforestation (red)
}

# Land cover colors (from parent project)
LANDCOVER_COLORS = {
    0: '#0077BE',  # Water (blue)
    1: '#228B22',  # Trees/Forest (green)
    4: '#FFD700',  # Crops (gold)
    5: '#DAA520',  # Shrub (dark goldenrod)
    6: '#DC143C',  # Built (crimson)
    7: '#D2B48C',  # Bare (tan)
}

# Year-based color palette for temporal analysis
YEAR_COLORS = plt.cm.viridis(np.linspace(0.1, 0.9, 7))


def plot_confusion_matrix(cm, title='Confusion Matrix',
                           class_names=None, save_path=None, verbose=True):
    """
    Plot normalized confusion matrix.

    Args:
        cm: Confusion matrix array
        title: Plot title
        class_names: List of class names
        save_path: Path to save figure
        verbose: Print save path
    """
    if class_names is None:
        class_names = ['No Change', 'Deforestation']

    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=FIGCOLOR)
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Proportion'})

    ax.set_xlabel('Predicted', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')
    ax.tick_params(labelsize=TICK_SIZE)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor=FIGCOLOR)
        if verbose:
            print(f"  Saved: {save_path}")

    plt.close()


def plot_confusion_matrices_comparison(results_dict, save_path=None, verbose=True):
    """
    Plot confusion matrices for all approaches side by side.

    Args:
        results_dict: Dict of approach_name -> results (with 'confusion_matrix')
        save_path: Path to save figure
        verbose: Print save path
    """
    n = len(results_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), facecolor=FIGCOLOR)
    if n == 1:
        axes = [axes]

    class_names = ['No Change', 'Deforestation']

    for ax, (name, results) in zip(axes, results_dict.items()):
        cm = results['confusion_matrix']
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, cbar=False)

        acc = results.get('accuracy', 0)
        f1 = results.get('f1_macro', 0)
        ax.set_title(f'{name}\nAcc={acc:.4f} | F1={f1:.4f}',
                     fontsize=TITLE_SIZE, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=LABEL_SIZE)
        ax.set_ylabel('Actual', fontsize=LABEL_SIZE)

    plt.suptitle('Change Detection Confusion Matrices',
                 fontsize=TITLE_SIZE + 2, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor=FIGCOLOR)
        if verbose:
            print(f"  Saved: {save_path}")

    plt.close()


def plot_training_curves(history, title='Training Curves', save_path=None, verbose=True):
    """
    Plot training loss and accuracy curves.

    Args:
        history: Dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        title: Plot title
        save_path: Path to save figure
        verbose: Print save path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=FIGCOLOR)

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=LABEL_SIZE)
    ax1.set_ylabel('Loss', fontsize=LABEL_SIZE)
    ax1.set_title('Loss', fontsize=TITLE_SIZE, fontweight='bold')
    ax1.legend(fontsize=TICK_SIZE)
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=LABEL_SIZE)
    ax2.set_ylabel('Accuracy', fontsize=LABEL_SIZE)
    ax2.set_title('Accuracy', fontsize=TITLE_SIZE, fontweight='bold')
    ax2.legend(fontsize=TICK_SIZE)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=TITLE_SIZE + 2, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor=FIGCOLOR)
        if verbose:
            print(f"  Saved: {save_path}")

    plt.close()


def plot_change_map(change_map, title='Deforestation Map', extent=None,
                     save_path=None, verbose=True):
    """
    Plot binary change detection map.

    Args:
        change_map: (H, W) binary array (1=deforestation)
        title: Plot title
        extent: [xmin, xmax, ymin, ymax] for georeferencing
        save_path: Path to save figure
        verbose: Print save path
    """
    fig, ax = plt.subplots(figsize=(12, 10), facecolor=FIGCOLOR)

    cmap = mcolors.ListedColormap(['#228B22', '#FF0000'])
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(change_map, cmap=cmap, norm=norm, extent=extent)

    legend_elements = [
        Patch(facecolor='#228B22', label='No Change'),
        Patch(facecolor='#FF0000', label='Deforestation'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=TICK_SIZE)

    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=LABEL_SIZE)
    ax.set_ylabel('Latitude', fontsize=LABEL_SIZE)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor=FIGCOLOR)
        if verbose:
            print(f"  Saved: {save_path}")

    plt.close()


def plot_bitemporal_comparison(rgb_t1, rgb_t2, change_map, year1, year2,
                                title=None, save_path=None, verbose=True):
    """
    Plot side-by-side: T1 RGB, T2 RGB, and change detection result.

    Args:
        rgb_t1: (H, W, 3) RGB image at time 1
        rgb_t2: (H, W, 3) RGB image at time 2
        change_map: (H, W) binary change map
        year1, year2: Year labels
        title: Optional title
        save_path: Path to save figure
        verbose: Print save path
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=FIGCOLOR)

    axes[0].imshow(rgb_t1)
    axes[0].set_title(f'Sentinel-2 RGB ({year1})', fontsize=LABEL_SIZE, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(rgb_t2)
    axes[1].set_title(f'Sentinel-2 RGB ({year2})', fontsize=LABEL_SIZE, fontweight='bold')
    axes[1].axis('off')

    cmap = mcolors.ListedColormap(['#228B22', '#FF0000'])
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    axes[2].imshow(change_map, cmap=cmap, norm=norm)
    axes[2].set_title(f'Change Detection ({year1}-{year2})',
                      fontsize=LABEL_SIZE, fontweight='bold')
    axes[2].axis('off')

    legend_elements = [
        Patch(facecolor='#228B22', label='No Change'),
        Patch(facecolor='#FF0000', label='Deforestation'),
    ]
    axes[2].legend(handles=legend_elements, loc='lower right', fontsize=TICK_SIZE - 1)

    if title:
        plt.suptitle(title, fontsize=TITLE_SIZE, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor=FIGCOLOR)
        if verbose:
            print(f"  Saved: {save_path}")

    plt.close()


def plot_annual_deforestation_trend(stats, save_path=None, verbose=True):
    """
    Plot annual deforestation as bar chart with trend line.

    Args:
        stats: Dict from compute_annual_deforestation_stats
        save_path: Path to save figure
        verbose: Print save path
    """
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=FIGCOLOR)

    years = stats['years']
    areas = stats['area_ha']

    bars = ax.bar(years, areas, color='#CC3333', alpha=0.8, edgecolor='darkred',
                  linewidth=0.5, label='Annual deforestation')

    # Add trend line if available
    if stats.get('trend') and stats['trend'] is not None:
        trend = stats['trend']
        x = np.array(years, dtype=float)
        y_trend = trend['slope_ha_per_year'] * x + trend['intercept']
        ax.plot(x, y_trend, 'k--', linewidth=2, alpha=0.7,
                label=f"Trend ({trend['slope_ha_per_year']:+.0f} ha/yr, "
                      f"R2={trend['r_squared']:.3f})")

    # Value labels on bars
    for bar, val in zip(bars, areas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(areas) * 0.01,
                f'{val:,.0f}', ha='center', va='bottom', fontsize=TICK_SIZE - 1,
                fontweight='bold')

    ax.set_xlabel('Year', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Deforestation Area (ha)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('Annual Deforestation in Jambi Province (2018-2024)',
                 fontsize=TITLE_SIZE, fontweight='bold')
    ax.legend(fontsize=TICK_SIZE)
    ax.set_xticks(years)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor=FIGCOLOR)
        if verbose:
            print(f"  Saved: {save_path}")

    plt.close()


def plot_cumulative_loss(stats, save_path=None, verbose=True):
    """
    Plot cumulative deforestation over time.

    Args:
        stats: Dict from compute_annual_deforestation_stats
        save_path: Path to save figure
        verbose: Print save path
    """
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=FIGCOLOR)

    years = stats['years']
    cumulative = stats['cumulative_ha']

    ax.fill_between(years, cumulative, alpha=0.3, color='#CC3333')
    ax.plot(years, cumulative, 'o-', color='#CC3333', linewidth=2.5,
            markersize=8, markeredgecolor='darkred', markerfacecolor='#FF4444')

    for y, c in zip(years, cumulative):
        ax.annotate(f'{c:,.0f} ha', (y, c), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=TICK_SIZE - 1,
                    fontweight='bold')

    ax.set_xlabel('Year', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Cumulative Deforestation (ha)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('Cumulative Forest Loss in Jambi Province (2018-2024)',
                 fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_xticks(years)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor=FIGCOLOR)
        if verbose:
            print(f"  Saved: {save_path}")

    plt.close()


def plot_transition_matrix(transition, title='Land Cover Transition Matrix',
                            save_path=None, verbose=True):
    """
    Plot transition matrix as heatmap.

    Args:
        transition: Dict from compute_transition_matrix
        title: Plot title
        save_path: Path to save figure
        verbose: Print save path
    """
    fig, ax = plt.subplots(figsize=(10, 8), facecolor=FIGCOLOR)

    matrix = transition['matrix']
    class_names = transition['class_names']

    # Normalize by row (from class)
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_norm = np.where(row_sums > 0, matrix / row_sums, 0)

    sns.heatmap(matrix_norm, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Transition Proportion'})

    ax.set_xlabel('To (T2)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('From (T1)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')
    ax.tick_params(labelsize=TICK_SIZE)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor=FIGCOLOR)
        if verbose:
            print(f"  Saved: {save_path}")

    plt.close()


def plot_approach_comparison_bar(results_dict, save_path=None, verbose=True):
    """
    Plot bar chart comparing metrics across approaches.

    Args:
        results_dict: Dict of approach_name -> results dict
        save_path: Path to save
        verbose: Print save path
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=FIGCOLOR)

    names = list(results_dict.keys())
    metrics = {
        'Accuracy': [results_dict[n].get('accuracy', 0) for n in names],
        'F1-Macro': [results_dict[n].get('f1_macro', 0) for n in names],
        'Kappa': [results_dict[n].get('kappa', 0) for n in names],
    }

    colors = ['#4472C4', '#ED7D31', '#70AD47']

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        bars = ax.bar(names, values, color=colors[:len(names)],
                      edgecolor='gray', linewidth=0.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=TICK_SIZE,
                    fontweight='bold')

        ax.set_ylabel(metric_name, fontsize=LABEL_SIZE, fontweight='bold')
        ax.set_title(metric_name, fontsize=TITLE_SIZE, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)

    plt.suptitle('Change Detection Approach Comparison',
                 fontsize=TITLE_SIZE + 2, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor=FIGCOLOR)
        if verbose:
            print(f"  Saved: {save_path}")

    plt.close()


def plot_feature_importance(importances, title='Feature Importance',
                             top_n=20, save_path=None, verbose=True):
    """
    Plot top feature importances as horizontal bar chart.

    Args:
        importances: List of (name, value) tuples
        title: Plot title
        top_n: Number of top features to show
        save_path: Path to save
        verbose: Print save path
    """
    importances = importances[:top_n]
    names = [x[0] for x in importances]
    values = [x[1] for x in importances]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.4)), facecolor=FIGCOLOR)

    bars = ax.barh(range(len(names)), values, color='#4472C4',
                   edgecolor='gray', linewidth=0.5)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=TICK_SIZE)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight', facecolor=FIGCOLOR)
        if verbose:
            print(f"  Saved: {save_path}")

    plt.close()


def create_rgb_from_sentinel(sentinel_data, bands=(2, 1, 0), percentile_clip=(2, 98)):
    """
    Create RGB image from Sentinel-2 data for visualization.

    Args:
        sentinel_data: (bands, H, W) Sentinel-2 data
        bands: Band indices for R, G, B (default: B4, B3, B2)
        percentile_clip: Percentile clipping range

    Returns:
        (H, W, 3) RGB image normalized to [0, 1]
    """
    rgb = np.stack([sentinel_data[b] for b in bands], axis=-1).astype(float)

    for i in range(3):
        band = rgb[:, :, i]
        valid = band[~np.isnan(band)]
        if len(valid) > 0:
            vmin = np.percentile(valid, percentile_clip[0])
            vmax = np.percentile(valid, percentile_clip[1])
            rgb[:, :, i] = np.clip((band - vmin) / (vmax - vmin + 1e-10), 0, 1)

    rgb = np.nan_to_num(rgb, nan=0)
    return rgb
