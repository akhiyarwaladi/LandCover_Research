"""
Generate Publication Figures: Dataset Preview & Misclassified Samples

Creates two SINGLE-PAGE figures:
1. Dataset Preview - EuroSAT (left) | UC Merced (right)
2. Misclassified Samples - EuroSAT (left) | UC Merced (right)

Output: results/figures/dataset_preview.png
        results/figures/misclassified_samples.png
"""

import os
import sys
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASETS, RESULTS_DIR, TRAINING
from modules.dataset_loader import find_dataset_root, load_dataset_from_folders

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.facecolor': 'white',
})

FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_image(path):
    return np.array(Image.open(path).convert('RGB'))


def format_class_name(name):
    name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    return name.replace('_', ' ').title()


def get_test_split(dataset_name, seed=42):
    root_dir = find_dataset_root(dataset_name)
    image_paths, labels, class_names = load_dataset_from_folders(root_dir)
    train_ratio = DATASETS[dataset_name]['train_ratio']
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, test_idx = next(sss.split(image_paths, labels))
    test_paths = [image_paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    return test_paths, test_labels, class_names


def get_one_sample_per_class(dataset_name):
    root_dir = find_dataset_root(dataset_name)
    image_paths, labels, class_names = load_dataset_from_folders(root_dir)
    samples = {}
    for path, label in zip(image_paths, labels):
        if label not in samples:
            samples[label] = path
        if len(samples) == len(class_names):
            break
    return samples, class_names


def get_misclassified(dataset_name, model_name):
    results_path = os.path.join(RESULTS_DIR, 'models', dataset_name,
                                model_name, 'test_results.npz')
    data = np.load(results_path)
    y_true = data['y_true']
    y_pred = data['y_pred']
    wrong = np.where(y_true != y_pred)[0]
    return wrong, y_true, y_pred


# ============================================================
# Figure 1: Dataset Preview - COMPACT SINGLE PAGE
# ============================================================
def generate_dataset_preview():
    """
    Single page figure: EuroSAT (left) | UC Merced (right).
    EuroSAT: 2 rows x 5 cols (10 classes)
    UC Merced: 3 rows x 7 cols (21 classes)
    """
    print("=" * 60)
    print("Generating Dataset Preview Figure (single page)")
    print("=" * 60)

    eurosat_samples, eurosat_classes = get_one_sample_per_class('eurosat')
    ucmerced_samples, ucmerced_classes = get_one_sample_per_class('ucmerced')

    n_euro = len(eurosat_classes)   # 10
    n_uc = len(ucmerced_classes)    # 21

    # Compact layout: wider, shorter
    euro_rows, euro_cols = 2, 5    # 10 classes in 2x5
    uc_rows, uc_cols = 3, 7       # 21 classes in 3x7

    # Single page size (~A4 landscape proportions)
    fig = plt.figure(figsize=(15, 7.5))

    fig.suptitle('Dataset Samples Overview',
                 fontweight='bold', fontsize=14, y=0.99)

    # Vertical divider
    line = plt.Line2D([0.38, 0.38], [0.02, 0.92],
                      transform=fig.transFigure,
                      color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    fig.add_artist(line)

    # --- Left: EuroSAT (2 rows x 5 cols) ---
    left_gs = gridspec.GridSpec(euro_rows, euro_cols, figure=fig,
                                left=0.01, right=0.36, top=0.88, bottom=0.03,
                                hspace=0.35, wspace=0.10)

    fig.text(0.185, 0.935,
             f'EuroSAT ({n_euro} classes, 27,000 images, 64x64 px)',
             ha='center', va='center', fontsize=9, fontweight='bold',
             style='italic', color='#1a5276')

    for idx in range(n_euro):
        row = idx // euro_cols
        col = idx % euro_cols
        ax = fig.add_subplot(left_gs[row, col])
        img = load_image(eurosat_samples[idx])
        ax.imshow(img, interpolation='bilinear')
        ax.set_title(format_class_name(eurosat_classes[idx]),
                     fontsize=6, fontweight='bold', pad=2)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#cccccc')
            spine.set_linewidth(0.5)

    # --- Right: UC Merced (3 rows x 7 cols) ---
    right_gs = gridspec.GridSpec(uc_rows, uc_cols, figure=fig,
                                 left=0.41, right=0.99, top=0.88, bottom=0.03,
                                 hspace=0.35, wspace=0.08)

    fig.text(0.70, 0.935,
             f'UC Merced ({n_uc} classes, 2,100 images, 256x256 px)',
             ha='center', va='center', fontsize=9, fontweight='bold',
             style='italic', color='#1a5276')

    for idx in range(n_uc):
        row = idx // uc_cols
        col = idx % uc_cols
        ax = fig.add_subplot(right_gs[row, col])
        img = load_image(ucmerced_samples[idx])
        ax.imshow(img)
        ax.set_title(format_class_name(ucmerced_classes[idx]),
                     fontsize=5.5, fontweight='bold', pad=2)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#cccccc')
            spine.set_linewidth(0.5)

    save_path = os.path.join(FIGURES_DIR, 'dataset_preview.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
# Figure 2: Misclassified Samples - COMPACT SINGLE PAGE
# ============================================================
def generate_misclassified_figure():
    """
    Single page figure: EuroSAT misclassified (left) | UC Merced misclassified (right).
    EuroSAT: 3 rows x 3 cols (9 samples)
    UC Merced: 3 rows x 2 cols (5 samples + 1 empty)
    """
    print("=" * 60)
    print("Generating Misclassified Samples Figure (single page)")
    print("=" * 60)

    euro_model = 'convnext_tiny'
    uc_model = 'resnet101'

    euro_test_paths, _, euro_classes = get_test_split('eurosat')
    uc_test_paths, _, uc_classes = get_test_split('ucmerced')

    euro_wrong, euro_y_true, euro_y_pred = get_misclassified('eurosat', euro_model)
    uc_wrong, uc_y_true, uc_y_pred = get_misclassified('ucmerced', uc_model)

    print(f"  EuroSAT ({euro_model}): {len(euro_wrong)} misclassified")
    print(f"  UC Merced ({uc_model}): {len(uc_wrong)} misclassified")

    # Select 9 diverse EuroSAT samples (round-robin from different classes)
    n_euro_show = 9
    class_groups = defaultdict(list)
    for idx in euro_wrong:
        class_groups[euro_y_true[idx]].append(idx)

    euro_selected = []
    while len(euro_selected) < n_euro_show:
        added = False
        for cls in sorted(class_groups.keys()):
            if class_groups[cls] and len(euro_selected) < n_euro_show:
                euro_selected.append(class_groups[cls].pop(0))
                added = True
        if not added:
            break
    euro_selected = sorted(euro_selected,
                           key=lambda i: (euro_y_true[i], euro_y_pred[i]))

    uc_selected = list(uc_wrong)  # All 5

    # Layout
    euro_r, euro_c = 3, 3   # 9 images
    uc_r, uc_c = 3, 2       # 5 images + 1 empty cell

    fig = plt.figure(figsize=(15, 7.5))

    fig.suptitle('Misclassified Samples Analysis',
                 fontweight='bold', fontsize=14, y=0.99)

    # Vertical divider
    line = plt.Line2D([0.53, 0.53], [0.02, 0.92],
                      transform=fig.transFigure,
                      color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    fig.add_artist(line)

    # --- Left: EuroSAT (3x3) ---
    left_gs = gridspec.GridSpec(euro_r, euro_c, figure=fig,
                                left=0.02, right=0.50, top=0.86, bottom=0.02,
                                hspace=0.55, wspace=0.18)

    fig.text(0.26, 0.935,
             f'EuroSAT - ConvNeXt-Tiny (Best Model)\n'
             f'{len(euro_wrong)} misclassified / 5,400 test (Acc: 99.06%)',
             ha='center', va='center', fontsize=9, fontweight='bold',
             style='italic', color='#1a5276')

    for i, idx in enumerate(euro_selected):
        row = i // euro_c
        col = i % euro_c
        ax = fig.add_subplot(left_gs[row, col])
        img = load_image(euro_test_paths[idx])
        ax.imshow(img, interpolation='bilinear')
        true_cls = format_class_name(euro_classes[euro_y_true[idx]])
        pred_cls = format_class_name(euro_classes[euro_y_pred[idx]])
        ax.set_title(f'True: {true_cls}\nPred: {pred_cls}',
                     fontsize=6.5, fontweight='bold', pad=3,
                     color='#c0392b')
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#e74c3c')
            spine.set_linewidth(2)

    # --- Right: UC Merced (3x2, 5 images) ---
    right_gs = gridspec.GridSpec(uc_r, uc_c, figure=fig,
                                 left=0.56, right=0.98, top=0.86, bottom=0.02,
                                 hspace=0.55, wspace=0.15)

    fig.text(0.77, 0.935,
             f'UC Merced - ResNet-101\n'
             f'{len(uc_wrong)} misclassified / 420 test (Acc: 98.81%)',
             ha='center', va='center', fontsize=9, fontweight='bold',
             style='italic', color='#1a5276')

    for i, idx in enumerate(uc_selected):
        row = i // uc_c
        col = i % uc_c
        ax = fig.add_subplot(right_gs[row, col])
        img = load_image(uc_test_paths[idx])
        ax.imshow(img)
        true_cls = format_class_name(uc_classes[uc_y_true[idx]])
        pred_cls = format_class_name(uc_classes[uc_y_pred[idx]])
        ax.set_title(f'True: {true_cls}\nPred: {pred_cls}',
                     fontsize=6.5, fontweight='bold', pad=3,
                     color='#c0392b')
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#e74c3c')
            spine.set_linewidth(2)

    # Hide last empty cell (6th position in 3x2 grid)
    # No need - matplotlib only shows axes we create

    save_path = os.path.join(FIGURES_DIR, 'misclassified_samples.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ============================================================
if __name__ == '__main__':
    generate_dataset_preview()
    print()
    generate_misclassified_figure()
    print()
    print("=" * 60)
    print("All figures generated successfully!")
    print(f"  Output: {FIGURES_DIR}/")
    print("=" * 60)
