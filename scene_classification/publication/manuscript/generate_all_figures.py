"""Generate ALL publication figures at IEEE single-column width.

Professional journal-quality figures with readable fonts, clean layouts,
and consistent styling for reputable international journal standards.

Usage: python generate_all_figures.py
"""

import os
import sys
import json
import glob
import random
import numpy as np
from PIL import Image, ImageEnhance

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import chi2
from collections import Counter

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SC_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SC_ROOT)
from config import RESULTS_DIR, MODELS, DATASETS

EUROSAT_DIR = os.path.join(SC_ROOT, 'data', 'eurosat', 'eurosat', '2750')
UCMERCED_DIR = os.path.join(SC_ROOT, 'data', 'ucmerced', 'UCMerced_LandUse', 'Images')

random.seed(42)
np.random.seed(42)

# ============================================================
# IEEE Constants & Professional Style
# ============================================================
COL_W = 3.45   # IEEE single-column width in inches
FULL_W = 7.16  # IEEE two-column (full page) width in inches

# Professional 8-color palette (colorblind-friendly, muted tones)
MODEL_COLORS = [
    '#2166AC',  # ResNet-50     (deep blue)
    '#4393C3',  # ResNet-101    (medium blue)
    '#66BD63',  # DenseNet-121  (green)
    '#F4A582',  # EffNet-B0     (salmon)
    '#D6604D',  # EffNet-B3     (brick red)
    '#9970AB',  # ViT-B/16      (purple)
    '#E08214',  # Swin-T        (amber)
    '#1A9876',  # ConvNeXt-T    (teal)
]

MODEL_DISPLAY = {
    'resnet50': 'ResNet-50', 'resnet101': 'ResNet-101',
    'densenet121': 'DenseNet-121',
    'efficientnet_b0': 'EffNet-B0', 'efficientnet_b3': 'EffNet-B3',
    'vit_b_16': 'ViT-B/16', 'swin_t': 'Swin-T',
    'convnext_tiny': 'ConvNeXt-T',
}

FAMILY_COLORS = {
    'cnn': '#2166AC',
    'transformer': '#D6604D',
    'cnn_modern': '#1A9876',
}
FAMILY_MARKERS = {'cnn': 'o', 'transformer': 's', 'cnn_modern': 'D'}
FAMILY_LABELS = {
    'cnn': 'Classical CNN',
    'transformer': 'Transformer',
    'cnn_modern': 'Modernized CNN',
}

# Short labels for class names
EURO_SHORT = {
    'AnnualCrop': 'AnnCr', 'Forest': 'Forst',
    'HerbaceousVegetation': 'HerbV', 'Highway': 'Hway',
    'Industrial': 'Indst', 'Pasture': 'Past',
    'PermanentCrop': 'PermC', 'Residential': 'Resid',
    'River': 'River', 'SeaLake': 'SLake',
}

UCM_SHORT = {
    'agricultural': 'Agri', 'airplane': 'Airp', 'baseballdiamond': 'Base',
    'beach': 'Beach', 'buildings': 'Build', 'chaparral': 'Chap',
    'denseresidential': 'DenR', 'forest': 'Forst', 'freeway': 'Freew',
    'golfcourse': 'Golf', 'harbor': 'Harb', 'intersection': 'Inter',
    'mediumresidential': 'MedR', 'mobilehomepark': 'MobH',
    'overpass': 'Over', 'parkinglot': 'Park', 'river': 'River',
    'runway': 'Runw', 'sparseresidential': 'SpaR',
    'storagetanks': 'StoT', 'tenniscourt': 'Tenn',
}


def setup_style():
    """Professional matplotlib style for IEEE journal figures."""
    plt.rcParams.update({
        # Font: serif family for IEEE standards
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'dejavuserif',

        # Font sizes — enlarged for readability at column width
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.titleweight': 'bold',
        'axes.labelsize': 8,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'legend.title_fontsize': 7.5,

        # Legend
        'legend.framealpha': 0.95,
        'legend.edgecolor': '#AAAAAA',
        'legend.fancybox': False,
        'legend.frameon': True,

        # Figure
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.04,
        'figure.facecolor': 'white',

        # Axes
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 0.7,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,

        # Grid
        'grid.alpha': 0.25,
        'grid.linewidth': 0.5,
        'grid.color': '#CCCCCC',

        # Lines
        'lines.linewidth': 1.3,
        'lines.markersize': 5,

        # Ticks
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
    })


setup_style()


# ============================================================
# Helpers
# ============================================================
def model_order():
    return ['resnet50', 'resnet101', 'densenet121', 'efficientnet_b0',
            'efficientnet_b3', 'vit_b_16', 'swin_t', 'convnext_tiny']


def load_summary():
    with open(os.path.join(RESULTS_DIR, 'all_experiments_summary.json')) as f:
        return json.load(f)


def load_eval(ds):
    out = {}
    for m in model_order():
        tp = os.path.join(RESULTS_DIR, 'models', ds, m, 'test_results.npz')
        mp = os.path.join(RESULTS_DIR, 'models', ds, m, 'evaluation_metrics.json')
        if os.path.exists(tp) and os.path.exists(mp):
            npz = np.load(tp, allow_pickle=True)
            with open(mp) as f:
                met = json.load(f)
            out[m] = {
                'y_true': npz['y_true'], 'y_pred': npz['y_pred'],
                'accuracy': met['accuracy'], 'per_class': met['per_class'],
                'class_names': met.get('class_names', []),
            }
    return out


def load_hist(ds):
    out = {}
    for m in model_order():
        hp = os.path.join(RESULTS_DIR, 'models', ds, m, 'training_history.npz')
        if os.path.exists(hp):
            npz = np.load(hp)
            out[m] = {k: npz[k] for k in ['train_loss', 'train_acc', 'test_loss', 'test_acc']}
    return out


def savefig(fig, name):
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.04)
    plt.close(fig)
    print(f"  {name}: {os.path.getsize(p)/1024:.0f} KB")


KEY_MODELS = ['resnet50', 'efficientnet_b3', 'vit_b_16', 'convnext_tiny']


# ============================================================
# 1. EuroSAT Sample Images (2 cols x 5 rows = 10 classes)
# ============================================================
def gen_sample_eurosat():
    print("\n[1] EuroSAT sample images...")
    classes = DATASETS['eurosat']['class_names']
    fig, axes = plt.subplots(5, 2, figsize=(COL_W, 5.8))
    fig.suptitle('EuroSAT Sentinel-2 Satellite Samples',
                 fontsize=9, fontweight='bold', y=0.99)
    for i, cls in enumerate(classes):
        r, c = i // 2, i % 2
        d = os.path.join(EUROSAT_DIR, cls)
        imgs = sorted(glob.glob(os.path.join(d, '*.jpg')))
        if imgs:
            img = Image.open(random.choice(imgs)).convert('RGB')
            axes[r, c].imshow(np.array(img))
        axes[r, c].set_title(cls, fontsize=7.5, fontweight='bold', pad=3)
        axes[r, c].axis('off')
    fig.subplots_adjust(hspace=0.25, wspace=0.08, top=0.94)
    savefig(fig, 'sample_eurosat.pdf')


# ============================================================
# 2. UC Merced Sample Images (3 cols x 7 rows = 21 classes)
# ============================================================
def gen_sample_ucmerced():
    print("[2] UC Merced sample images...")
    classes = DATASETS['ucmerced']['class_names']
    fig, axes = plt.subplots(7, 3, figsize=(COL_W, 6.2))
    fig.suptitle('UC Merced Aerial Image Samples',
                 fontsize=9, fontweight='bold', y=0.99)
    for i, cls in enumerate(classes):
        r, c = i // 3, i % 3
        d = os.path.join(UCMERCED_DIR, cls)
        imgs = sorted(glob.glob(os.path.join(d, '*.tif')))
        if imgs:
            img = Image.open(random.choice(imgs)).convert('RGB')
            img = img.resize((128, 128), Image.LANCZOS)
            axes[r, c].imshow(np.array(img))
        axes[r, c].set_title(cls, fontsize=5.5, fontweight='bold', pad=2)
        axes[r, c].axis('off')
    fig.subplots_adjust(hspace=0.28, wspace=0.08, top=0.94)
    savefig(fig, 'sample_ucmerced.pdf')


# ============================================================
# 3. Data Augmentation (3 rows x 3 cols)
# ============================================================
def gen_augmentation():
    print("[3] Augmentation visualization...")
    classes_show = ['Forest', 'Residential', 'AnnualCrop']
    n_aug = 2

    def augment(img):
        img = img.resize((224, 224), Image.LANCZOS)
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img = img.rotate(random.uniform(-15, 15), fillcolor=(0, 0, 0))
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
        img = ImageEnhance.Color(img).enhance(random.uniform(0.9, 1.1))
        return img

    fig, axes = plt.subplots(len(classes_show), n_aug + 1,
                             figsize=(COL_W, 3.8))
    titles = ['Original'] + [f'Augmented #{i+1}' for i in range(n_aug)]
    for j, t in enumerate(titles):
        axes[0, j].set_title(t, fontsize=7.5, fontweight='bold', pad=4)

    for i, cls in enumerate(classes_show):
        d = os.path.join(EUROSAT_DIR, cls)
        imgs = sorted(glob.glob(os.path.join(d, '*.jpg')))
        img = Image.open(random.choice(imgs)).convert('RGB')
        axes[i, 0].imshow(np.array(img.resize((224, 224), Image.LANCZOS)))
        axes[i, 0].set_ylabel(cls, fontsize=7, fontweight='bold',
                               rotation=90, labelpad=6)
        for j in range(n_aug):
            axes[i, j+1].imshow(np.array(augment(img)))
        for j in range(n_aug + 1):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].yaxis.set_visible(j == 0)
            for sp in axes[i, j].spines.values():
                sp.set_visible(True)
                sp.set_color('#BBBBBB')
                sp.set_linewidth(0.5)

    fig.subplots_adjust(hspace=0.12, wspace=0.08, left=0.12, top=0.92)
    savefig(fig, 'augmentation.pdf')


# ============================================================
# 4. Confusion Matrices (2x2 grid)
# ============================================================
def gen_cm(ds_tag, ds_disp, eval_data, short_map):
    print(f"[4] Confusion matrices ({ds_disp})...")
    classes = eval_data[list(eval_data.keys())[0]]['class_names']
    n_cls = len(classes)
    models = [m for m in KEY_MODELS if m in eval_data]
    short_labels = [short_map.get(c, c[:5]) for c in classes]

    if n_cls <= 12:
        fig_w, fig_h = COL_W, 5.2
        annot_size = 5.5
        tick_size = 6
        title_size = 7.5
        suptitle_size = 9
        x_rot = 40
        hspace, wspace = 0.55, 0.45
    else:
        fig_w, fig_h = FULL_W, 5.8
        annot_size = 5.5
        tick_size = 6
        title_size = 8
        suptitle_size = 9.5
        x_rot = 45
        hspace, wspace = 0.55, 0.40

    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h))
    axes = axes.flatten()

    for ax in axes:
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

    for i, m in enumerate(models):
        d = eval_data[m]
        cm = confusion_matrix(d['y_true'], d['y_pred'])
        cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        annot_arr = np.empty_like(cm_n, dtype=object)
        for r in range(n_cls):
            for c in range(n_cls):
                val = cm_n[r, c]
                annot_arr[r, c] = f'{val:.2f}' if val >= 0.005 else ''

        sns.heatmap(cm_n, annot=annot_arr, fmt='', cmap='Blues',
                    vmin=0, vmax=1, ax=axes[i], cbar=False,
                    xticklabels=short_labels,
                    yticklabels=short_labels,
                    annot_kws={'size': annot_size, 'fontweight': 'medium'},
                    linewidths=0.3, linecolor='white', square=True)
        acc = d['accuracy']
        axes[i].set_title(f"{MODEL_DISPLAY.get(m, m)} ({acc:.2%})",
                          fontsize=title_size, fontweight='bold', pad=4)
        axes[i].tick_params(axis='both', labelsize=tick_size, length=2)
        axes[i].tick_params(axis='x', rotation=x_rot)
        axes[i].tick_params(axis='y', rotation=0)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

    for j in range(len(models), 4):
        axes[j].set_visible(False)

    fig.suptitle(f'Confusion Matrices on {ds_disp}',
                 fontsize=suptitle_size, fontweight='bold', y=1.01)
    fig.subplots_adjust(hspace=hspace, wspace=wspace, top=0.92)
    savefig(fig, f'cm_{ds_tag}.pdf')


# ============================================================
# 5. Training Curves (stacked: loss on top, accuracy below)
# ============================================================
def gen_curves(ds, histories):
    tag = ds
    ds_disp = 'EuroSAT' if ds == 'eurosat' else 'UC Merced'
    print(f"[5] Training curves ({ds_disp})...")

    models = [m for m in model_order() if m in histories]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 4.5), sharex=True)

    for idx, m in enumerate(models):
        h = histories[m]
        ep = range(1, len(h['train_loss']) + 1)
        c = MODEL_COLORS[idx]
        disp = MODEL_DISPLAY.get(m, m)
        # Only show test (validation) curves — dashed train lines removed for clarity
        ax1.plot(ep, h['test_loss'], '-', color=c, label=disp, lw=1.2)
        ax2.plot(ep, h['test_acc'], '-', color=c, label=disp, lw=1.2)

    ax1.set_ylabel('Validation Loss', fontsize=8, fontweight='bold')
    ax1.set_title(f'Training Dynamics on {ds_disp}', fontsize=9, fontweight='bold', pad=5)
    ax2.set_ylabel('Validation Accuracy', fontsize=8, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=8, fontweight='bold')

    # Single shared legend at top of loss subplot
    ax1.legend(fontsize=5.5, ncol=4, loc='upper right', framealpha=0.95,
               edgecolor='#CCCCCC', handlelength=1.2, columnspacing=0.6,
               handletextpad=0.4)
    ax2.grid(True, alpha=0.20, linewidth=0.4)
    ax1.grid(True, alpha=0.20, linewidth=0.4)
    ax1.tick_params(labelsize=7)
    ax2.tick_params(labelsize=7)

    fig.subplots_adjust(hspace=0.10)
    savefig(fig, f'curves_{tag}.pdf')


# ============================================================
# 6. Accuracy Comparison Bar Chart (both datasets)
# ============================================================
def gen_accuracy(data):
    print("[6] Accuracy comparison...")
    results = data['results']
    mo = model_order()
    datasets = [d for d in ['eurosat', 'ucmerced'] if d in results]
    ds_lbl = {'eurosat': 'EuroSAT', 'ucmerced': 'UC Merced'}
    ds_col = ['#2166AC', '#D6604D']

    fig, ax = plt.subplots(figsize=(COL_W, 3.5))
    x = np.arange(len(mo))
    w = 0.38

    for i, ds in enumerate(datasets):
        accs = [results[ds].get(m, {}).get('accuracy', 0) * 100 for m in mo]
        bars = ax.bar(x + i*w - w/2, accs, w, label=ds_lbl.get(ds, ds),
                      color=ds_col[i], edgecolor='white', lw=0.6, alpha=0.92)
        # Annotations inside bars (white text near top)
        for bar, a in zip(bars, accs):
            if a > 0:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() - 0.18,
                        f'{a:.1f}', ha='center', va='top',
                        fontsize=5, fontweight='bold', color='white')

    disp = [MODEL_DISPLAY.get(m, m) for m in mo]
    ax.set_xticks(x)
    ax.set_xticklabels(disp, rotation=30, ha='right', fontsize=6.5)
    ax.set_ylabel('Overall Accuracy (%)', fontsize=8, fontweight='bold')
    ax.set_title('Accuracy Comparison Across Datasets',
                 fontweight='bold', fontsize=9, pad=5)
    ax.set_ylim(97.8, 100.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.legend(fontsize=6.5, loc='lower right', edgecolor='#CCCCCC')
    ax.grid(axis='y', alpha=0.20, linewidth=0.4)
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    savefig(fig, 'accuracy_comparison.pdf')


# ============================================================
# 7. Per-Class F1 Heatmap (models x classes)
# ============================================================
def gen_f1_heatmap(ds, eval_data):
    tag = ds
    ds_disp = 'EuroSAT' if ds == 'eurosat' else 'UC Merced'
    print(f"[7] Per-class F1 heatmap ({ds_disp})...")

    classes = eval_data[list(eval_data.keys())[0]]['class_names']
    mo = [m for m in model_order() if m in eval_data]
    n_cls = len(classes)

    mat = np.zeros((len(mo), n_cls))
    for i, m in enumerate(mo):
        mat[i] = eval_data[m]['per_class']['f1']

    disp_models = [MODEL_DISPLAY.get(m, m) for m in mo]
    height = max(2.8, len(mo) * 0.38)

    if n_cls <= 12:
        show_annot = True
        annot_kws = {'size': 5.5, 'fontweight': 'medium'}
        x_labels = [EURO_SHORT.get(c, c[:5]) for c in classes]
        x_tick_size = 6
    else:
        show_annot = False
        annot_kws = {}
        x_labels = [UCM_SHORT.get(c, c[:4]) for c in classes]
        x_tick_size = 5

    fig, ax = plt.subplots(figsize=(COL_W, height))

    for sp in ax.spines.values():
        sp.set_visible(True)

    fmt = '.2f' if show_annot else ''
    sns.heatmap(mat, annot=show_annot, fmt=fmt, cmap='YlGn',
                vmin=0.9 if ds == 'eurosat' else 0.85, vmax=1.0,
                xticklabels=x_labels, yticklabels=disp_models, ax=ax,
                annot_kws=annot_kws, linewidths=0.4, linecolor='white',
                cbar_kws={'shrink': 0.8, 'label': 'F1'})
    ax.set_title(f'Per-Class F1-Score on {ds_disp}',
                 fontweight='bold', fontsize=9, pad=6)
    ax.tick_params(axis='x', rotation=40, labelsize=x_tick_size)
    ax.tick_params(axis='y', rotation=0, labelsize=6.5)
    plt.tight_layout()
    savefig(fig, f'per_class_f1_{tag}.pdf')


# ============================================================
# 8. McNemar p-value Heatmap
# ============================================================
def gen_mcnemar(ds, eval_data):
    tag = ds
    ds_disp = 'EuroSAT' if ds == 'eurosat' else 'UC Merced'
    print(f"[8] McNemar heatmap ({ds_disp})...")

    mo = [m for m in model_order() if m in eval_data]
    disp = [MODEL_DISPLAY.get(m, m) for m in mo]
    n = len(mo)
    pmat = np.ones((n, n))

    for i in range(n):
        for j in range(i+1, n):
            yt = eval_data[mo[i]]['y_true']
            pa = eval_data[mo[i]]['y_pred']
            pb = eval_data[mo[j]]['y_pred']
            ca, cb = (pa == yt), (pb == yt)
            n01, n10 = np.sum(ca & ~cb), np.sum(~ca & cb)
            if n01 + n10 > 0:
                chi2s = (abs(n01 - n10) - 1)**2 / (n01 + n10)
                pv = 1 - chi2.cdf(chi2s, df=1)
            else:
                pv = 1.0
            pmat[i, j] = pv
            pmat[j, i] = pv

    fig, ax = plt.subplots(figsize=(COL_W, 3.4))

    for sp in ax.spines.values():
        sp.set_visible(True)

    mask = np.triu(np.ones_like(pmat, dtype=bool), k=0)
    sns.heatmap(pmat, mask=mask, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=0.1, xticklabels=disp, yticklabels=disp, ax=ax,
                linewidths=0.6, linecolor='white', square=True,
                annot_kws={'size': 6, 'fontweight': 'bold'},
                cbar_kws={'shrink': 0.7, 'label': 'p-value'})
    ax.set_title(f"McNemar's Test p-values on {ds_disp}",
                 fontweight='bold', fontsize=9, pad=6)
    ax.tick_params(axis='x', rotation=30, labelsize=6)
    ax.tick_params(axis='y', rotation=0, labelsize=6)

    # Add significance asterisks
    for i in range(n):
        for j in range(i+1, n):
            p = pmat[i, j]
            mk = '**' if p < 0.01 else ('*' if p < 0.05 else '')
            if mk:
                ax.text(j + 0.5, i + 0.82, mk, ha='center', va='center',
                        fontsize=6, color='black', fontweight='bold')
    plt.tight_layout()
    savefig(fig, f'mcnemar_{tag}.pdf')


# ============================================================
# 9. Efficiency Scatter Plot
# ============================================================
def gen_efficiency(data):
    print("[9] Efficiency plot...")
    results = data['results']
    ds = 'eurosat'
    if ds not in results:
        return

    fig, ax = plt.subplots(figsize=(COL_W, 3.2))
    plotted = set()

    # Custom offsets to avoid label overlap (model_key -> (dx, dy))
    label_offsets = {
        'resnet50': (5, -8),
        'resnet101': (5, -8),
        'densenet121': (5, 4),
        'efficientnet_b0': (5, 4),
        'efficientnet_b3': (-40, 6),
        'vit_b_16': (5, -8),
        'swin_t': (5, 5),
        'convnext_tiny': (-48, 6),
    }

    for m in model_order():
        if m not in results[ds]:
            continue
        info = MODELS[m]
        acc = results[ds][m]['accuracy'] * 100
        fam = info.get('family', 'cnn')
        par = results[ds][m].get('params_m', info.get('params_m', 0))
        lbl = FAMILY_LABELS.get(fam, fam) if fam not in plotted else ''
        plotted.add(fam)
        ax.scatter(par, acc, c=FAMILY_COLORS.get(fam, 'gray'),
                   marker=FAMILY_MARKERS.get(fam, 'o'),
                   s=65, edgecolors='#333333', lw=0.7, label=lbl,
                   zorder=3, alpha=0.92)
        dx, dy = label_offsets.get(m, (5, 5))
        ax.annotate(MODEL_DISPLAY.get(m, m), (par, acc),
                    textcoords='offset points', xytext=(dx, dy), fontsize=5.5,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white',
                              ec='#BBBBBB', alpha=0.9, lw=0.4))

    ax.set_xlabel('Parameters (M)', fontsize=8, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=8, fontweight='bold')
    ax.set_title('Accuracy vs. Parameter Count on EuroSAT',
                 fontweight='bold', fontsize=9, pad=5)
    ax.grid(True, alpha=0.20, linewidth=0.4)
    ax.legend(fontsize=6, loc='lower right', edgecolor='#CCCCCC')
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    savefig(fig, 'efficiency_eurosat.pdf')


# ============================================================
# 10. Error / Prediction Analysis (stacked vertically)
# ============================================================
def gen_prediction(eval_data):
    print("[10] Prediction analysis...")
    classes = DATASETS['eurosat']['class_names']
    best = eval_data.get('convnext_tiny')
    if not best:
        print("  Skipped")
        return

    yt, yp = best['y_true'], best['y_pred']
    ok = (yt == yp)
    bad = ~ok

    short = EURO_SHORT
    short_classes = [short.get(c, c[:5]) for c in classes]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 4.6))

    # Top: per-class correct vs wrong
    corr = [np.sum(ok & (yt == c)) for c in range(len(classes))]
    wrng = [np.sum(bad & (yt == c)) for c in range(len(classes))]
    x = np.arange(len(classes))
    ax1.bar(x, corr, color='#27AE60', alpha=0.88, edgecolor='white',
            lw=0.4, label='Correct')
    ax1.bar(x, wrng, bottom=corr, color='#C0392B', alpha=0.88,
            edgecolor='white', lw=0.4, label='Misclassified')
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_classes, rotation=35, ha='right', fontsize=6)
    ax1.set_ylabel('Samples', fontsize=8, fontweight='bold')
    ax1.set_title('Per-Class Prediction Breakdown (ConvNeXt-T)',
                  fontweight='bold', fontsize=8, pad=5)
    ax1.legend(fontsize=6, edgecolor='#CCCCCC', loc='upper right')
    ax1.grid(axis='y', alpha=0.20, linewidth=0.4)
    ax1.tick_params(labelsize=7)

    # Bottom: top misclassification patterns
    cm = confusion_matrix(yt, yp)
    np.fill_diagonal(cm, 0)
    flat = np.argsort(cm.ravel())[::-1][:6]
    rows, cols = np.unravel_index(flat, cm.shape)
    lbs, vals = [], []
    for r, c in zip(rows, cols):
        if cm[r, c] > 0:
            tc_s = short.get(classes[r], classes[r][:5])
            pc_s = short.get(classes[c], classes[c][:5])
            lbs.append(f'{tc_s} \u2192 {pc_s}')
            vals.append(cm[r, c])
    if vals:
        yp2 = np.arange(len(vals))
        clrs = plt.cm.Reds(np.linspace(0.35, 0.75, len(vals)))
        ax2.barh(yp2, vals, color=clrs, edgecolor='white', lw=0.4, height=0.65)
        ax2.set_yticks(yp2)
        ax2.set_yticklabels(lbs, fontsize=6.5)
        ax2.set_xlabel('Count', fontsize=8, fontweight='bold')
        ax2.set_title('Top Misclassification Patterns',
                      fontweight='bold', fontsize=8, pad=5)
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.20, linewidth=0.4)
        ax2.tick_params(labelsize=7)
        for i, v in enumerate(vals):
            ax2.text(v + 0.3, i, str(int(v)), va='center',
                     fontsize=6.5, fontweight='bold')

    fig.subplots_adjust(hspace=0.50)
    savefig(fig, 'prediction_analysis.pdf')


# ============================================================
# 11. Misclassified Image Examples (side-by-side comparison)
# ============================================================
def reconstruct_test_paths(ds):
    """Reconstruct test image paths using the same deterministic split."""
    root = EUROSAT_DIR if ds == 'eurosat' else UCMERCED_DIR
    image_paths = []
    labels = []
    class_names = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])
    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(root, cls)
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff'):
            for img_path in glob.glob(os.path.join(cls_dir, ext)):
                image_paths.append(img_path)
                labels.append(idx)

    labels = np.array(labels)
    train_ratio = DATASETS[ds]['train_ratio']
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio,
                                  random_state=42)
    _, test_idx = next(sss.split(image_paths, labels))

    test_paths = [image_paths[i] for i in test_idx]
    test_labels = labels[test_idx]
    return test_paths, test_labels, class_names, image_paths, labels


def gen_misclassified(ds, eval_data):
    ds_disp = 'EuroSAT' if ds == 'eurosat' else 'UC Merced'
    short_map = EURO_SHORT if ds == 'eurosat' else UCM_SHORT
    print(f"[11] Misclassified examples ({ds_disp})...")

    test_paths, test_labels, class_names, all_paths, all_labels = \
        reconstruct_test_paths(ds)

    confusion_examples = {}
    for m in model_order():
        if m not in eval_data:
            continue
        d = eval_data[m]
        y_true, y_pred = d['y_true'], d['y_pred']
        if not np.array_equal(test_labels, y_true):
            print(f"  WARNING: test labels mismatch for {m}. Skipping.")
            return
        wrong = np.where(y_true != y_pred)[0]
        for idx in wrong:
            pair = (int(y_true[idx]), int(y_pred[idx]))
            if pair not in confusion_examples:
                confusion_examples[pair] = []
            confusion_examples[pair].append((idx, m))

    if not confusion_examples:
        print("  No misclassifications found. Skipping.")
        return

    sorted_pairs = sorted(confusion_examples.items(), key=lambda x: -len(x[1]))
    max_rows = min(6, len(sorted_pairs))
    pairs_to_show = sorted_pairs[:max_rows]
    n_rows = len(pairs_to_show)

    fig, axes = plt.subplots(n_rows, 2, figsize=(COL_W, n_rows * 1.35 + 0.6))
    if n_rows == 1:
        axes = axes.reshape(1, 2)

    fig.suptitle(f'Misclassified Examples on {ds_disp}',
                 fontsize=9, fontweight='bold', y=0.995)

    axes[0, 0].set_title('Misclassified Image', fontsize=7,
                         fontweight='bold', pad=4, color='#B03A2E')
    axes[0, 1].set_title('Example from Predicted Class', fontsize=7,
                         fontweight='bold', pad=4, color='#2471A3')

    for row, ((tc, pc), examples) in enumerate(pairs_to_show):
        seen_idx = set()
        unique_examples = []
        for tidx, mname in examples:
            if tidx not in seen_idx:
                seen_idx.add(tidx)
                unique_examples.append((tidx, mname))
        test_idx_use, model_name = unique_examples[0]
        n_models = len(examples)

        misc_img = Image.open(test_paths[test_idx_use]).convert('RGB')
        pred_class_imgs = [all_paths[i] for i in range(len(all_paths))
                           if all_labels[i] == pc]
        pred_ex = Image.open(pred_class_imgs[0]).convert('RGB')

        tc_short = short_map.get(class_names[tc], class_names[tc])
        pc_short = short_map.get(class_names[pc], class_names[pc])
        unique_models = len(set(mname for _, mname in examples))

        # Left: misclassified image (red border)
        axes[row, 0].imshow(np.array(misc_img))
        for sp in axes[row, 0].spines.values():
            sp.set_visible(True)
            sp.set_color('#C0392B')
            sp.set_linewidth(2.5)
        err_tag = f'({n_models} err, {unique_models} mdl)'
        axes[row, 0].set_ylabel(f'{tc_short}\n\u2192 {pc_short}\n{err_tag}',
                                fontsize=5.5, fontweight='bold', rotation=0,
                                labelpad=32, va='center', ha='center')

        # Right: example from predicted class (blue border)
        axes[row, 1].imshow(np.array(pred_ex))
        for sp in axes[row, 1].spines.values():
            sp.set_visible(True)
            sp.set_color('#2471A3')
            sp.set_linewidth(2.5)

        for j in range(2):
            axes[row, j].tick_params(left=False, bottom=False,
                                     labelleft=False, labelbottom=False)

    fig.subplots_adjust(hspace=0.22, wspace=0.12, top=0.91, left=0.18)
    savefig(fig, f'misclassified_{ds}.pdf')


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  PROFESSIONAL JOURNAL-QUALITY FIGURES (300 DPI)")
    print("=" * 60)

    data = load_summary()

    gen_sample_eurosat()
    gen_sample_ucmerced()
    gen_augmentation()

    ev_euro = load_eval('eurosat')
    ev_ucm = load_eval('ucmerced')

    if ev_euro:
        gen_cm('eurosat', 'EuroSAT', ev_euro, EURO_SHORT)
        gen_f1_heatmap('eurosat', ev_euro)
        gen_mcnemar('eurosat', ev_euro)
    if ev_ucm:
        gen_cm('ucmerced', 'UC Merced', ev_ucm, UCM_SHORT)
        gen_f1_heatmap('ucmerced', ev_ucm)
        gen_mcnemar('ucmerced', ev_ucm)

    for ds in ['eurosat', 'ucmerced']:
        hi = load_hist(ds)
        if hi:
            gen_curves(ds, hi)

    gen_accuracy(data)
    gen_efficiency(data)

    if ev_euro:
        gen_prediction(ev_euro)
        gen_misclassified('eurosat', ev_euro)
    if ev_ucm:
        gen_misclassified('ucmerced', ev_ucm)

    print("\n" + "=" * 60)
    print("  ALL FIGURES COMPLETE!")
    total = 0
    for f in sorted(os.listdir(FIG_DIR)):
        if f.endswith('.pdf'):
            s = os.path.getsize(os.path.join(FIG_DIR, f))
            total += s
    print(f"  Total: {total/1024:.0f} KB ({total/(1024*1024):.1f} MB)")


if __name__ == '__main__':
    main()
