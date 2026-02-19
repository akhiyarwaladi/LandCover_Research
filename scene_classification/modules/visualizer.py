"""
Publication-Quality Visualization for Scene Classification

Generates figures at 300 DPI following IEEE/ISPRS standards.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR


# Consistent style
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.facecolor': 'white',
})


def plot_confusion_matrix(y_true, y_pred, class_names, title='',
                          save_path=None, figsize=None, normalize=True):
    """Plot a single confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
        vmax = 1.0
    else:
        fmt = 'd'
        vmax = None

    n = len(class_names)
    if figsize is None:
        figsize = (max(8, n * 0.4), max(6, n * 0.35))

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.heatmap(cm, annot=n <= 20, fmt=fmt, cmap='Blues', vmin=0, vmax=vmax,
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                annot_kws={'size': 7} if n > 15 else {})
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('True', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=13)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)
        print(f"  Saved: {save_path}")
    else:
        return fig


def plot_confusion_matrices_grid(results_dict, class_names, dataset_name,
                                 save_path=None):
    """Plot confusion matrices for multiple models in a grid."""
    model_names = list(results_dict.keys())
    n_models = len(model_names)
    ncols = min(4, n_models)
    nrows = (n_models + ncols - 1) // ncols
    n_classes = len(class_names)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 4.5 * nrows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, name in enumerate(model_names):
        res = results_dict[name]
        cm = confusion_matrix(res['y_true'], res['y_pred'])
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

        sns.heatmap(cm_norm, annot=n_classes <= 15, fmt='.2f', cmap='Blues',
                    vmin=0, vmax=1, ax=axes[i],
                    xticklabels=class_names if n_classes <= 15 else False,
                    yticklabels=class_names if n_classes <= 15 else False,
                    annot_kws={'size': 6})
        acc = res['accuracy']
        axes[i].set_title(f"{name}\nAcc: {acc:.2%}", fontweight='bold',
                          fontsize=10)

    # Hide empty subplots
    for j in range(n_models, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'Confusion Matrices - {dataset_name}',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)
        print(f"  Saved: {save_path}")


def plot_training_curves(histories, save_path=None):
    """
    Plot training curves for multiple models.

    Args:
        histories: dict of {model_name: {train_loss, train_acc, test_loss, test_acc}}
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    for (name, hist), color in zip(histories.items(), colors):
        epochs = range(1, len(hist['train_loss']) + 1)
        ax1.plot(epochs, hist['train_loss'], '--', color=color, alpha=0.5)
        ax1.plot(epochs, hist['test_loss'], '-', color=color, label=name)

        ax2.plot(epochs, hist['train_acc'], '--', color=color, alpha=0.5)
        ax2.plot(epochs, hist['test_acc'], '-', color=color, label=name)

    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('Training (--) and Test (-) Loss', fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Accuracy', fontweight='bold')
    ax2.set_title('Training (--) and Test (-) Accuracy', fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)
        print(f"  Saved: {save_path}")


def plot_accuracy_comparison(results_summary, save_path=None):
    """
    Bar chart comparing model accuracy across datasets.

    Args:
        results_summary: {dataset: {model: accuracy}}
    """
    datasets = list(results_summary.keys())
    models = list(results_summary[datasets[0]].keys())

    x = np.arange(len(models))
    width = 0.8 / len(datasets)
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(datasets)))

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.2), 6))

    for i, ds in enumerate(datasets):
        accs = [results_summary[ds].get(m, 0) for m in models]
        bars = ax.bar(x + i * width, [a * 100 for a in accs], width,
                      label=ds, color=colors[i], edgecolor='black',
                      linewidth=0.5)
        # Value labels
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f'{acc:.1%}', ha='center', va='bottom', fontsize=7,
                        rotation=45)

    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Model Accuracy Comparison Across Datasets', fontweight='bold')
    ax.set_xticks(x + width * (len(datasets) - 1) / 2)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)
        print(f"  Saved: {save_path}")


def plot_per_class_f1(results_dict, class_names, dataset_name,
                      save_path=None):
    """Grouped bar chart of per-class F1 scores."""
    models = list(results_dict.keys())
    n_classes = len(class_names)
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(max(12, n_classes * 0.5), 6))

    x = np.arange(n_classes)
    width = 0.8 / n_models
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    for i, name in enumerate(models):
        f1s = results_dict[name]['per_class']['f1']
        ax.bar(x + i * width, f1s, width, label=name, color=colors[i],
               edgecolor='black', linewidth=0.3)

    ax.set_xlabel('Class', fontweight='bold')
    ax.set_ylabel('F1-Score', fontweight='bold')
    ax.set_title(f'Per-Class F1-Score - {dataset_name}', fontweight='bold')
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=7)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)
        print(f"  Saved: {save_path}")


def plot_mcnemar_matrix(comparisons, model_names, save_path=None):
    """Heatmap of McNemar p-values."""
    n = len(model_names)
    pval_matrix = np.ones((n, n))

    name_to_idx = {name: i for i, name in enumerate(model_names)}
    for comp in comparisons:
        i = name_to_idx[comp['model_a']]
        j = name_to_idx[comp['model_b']]
        pval_matrix[i, j] = comp['p_value']
        pval_matrix[j, i] = comp['p_value']

    fig, ax = plt.subplots(figsize=(max(7, n * 0.8), max(6, n * 0.7)))

    mask = np.triu(np.ones_like(pval_matrix, dtype=bool), k=0)
    sns.heatmap(pval_matrix, mask=mask, annot=True, fmt='.4f',
                cmap='RdYlGn', vmin=0, vmax=0.1,
                xticklabels=model_names, yticklabels=model_names, ax=ax)
    ax.set_title("McNemar's Test p-values", fontweight='bold', fontsize=13)
    plt.xticks(rotation=30, ha='right')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)
        print(f"  Saved: {save_path}")


def plot_model_efficiency(model_info, accuracies, save_path=None):
    """Scatter plot: accuracy vs parameters (bubble = family)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    family_colors = {'cnn': '#2196F3', 'transformer': '#F44336',
                     'cnn_modern': '#4CAF50'}
    family_markers = {'cnn': 'o', 'transformer': 's', 'cnn_modern': 'D'}

    for name, info in model_info.items():
        if name not in accuracies:
            continue
        family = info.get('family', 'cnn')
        ax.scatter(info['params_m'], accuracies[name] * 100,
                   c=family_colors.get(family, 'gray'),
                   marker=family_markers.get(family, 'o'),
                   s=120, edgecolors='black', linewidth=0.8,
                   label=f"{family}" if family not in ax.get_legend_handles_labels()[1] else '',
                   zorder=3)
        ax.annotate(name, (info['params_m'], accuracies[name] * 100),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel('Parameters (M)', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Model Efficiency: Accuracy vs Parameters', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)
        print(f"  Saved: {save_path}")


def plot_sample_images(dataset_loader, class_names, dataset_name,
                       n_samples=3, save_path=None):
    """Show sample images from each class."""
    from torchvision.utils import make_grid
    import torchvision.transforms.functional as TF

    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, n_samples,
                             figsize=(n_samples * 2.5, n_classes * 2))

    class_counts = {i: 0 for i in range(n_classes)}
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for inputs, targets in dataset_loader:
        for img, target in zip(inputs, targets):
            t = target.item()
            if class_counts[t] < n_samples:
                col = class_counts[t]
                # Denormalize
                img_np = img.permute(1, 2, 0).numpy()
                img_np = img_np * std + mean
                img_np = np.clip(img_np, 0, 1)

                axes[t, col].imshow(img_np)
                axes[t, col].axis('off')
                if col == 0:
                    axes[t, col].set_ylabel(class_names[t], fontsize=8,
                                            rotation=0, ha='right',
                                            va='center')
                class_counts[t] += 1

        if all(c >= n_samples for c in class_counts.values()):
            break

    plt.suptitle(f'Sample Images - {dataset_name}',
                 fontweight='bold', fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)
        print(f"  Saved: {save_path}")
