"""Regenerate all publication figures as PDF vector format for LaTeX.

Usage: python regenerate_figures_pdf.py
Output: PDF vector figures in figures/ subdirectory.
"""

import os
import sys
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Add scene_classification root to path
SC_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SC_ROOT)

from config import RESULTS_DIR, MODELS, DATASETS
from modules.visualizer import (
    plot_confusion_matrices_grid, plot_training_curves,
    plot_accuracy_comparison, plot_per_class_f1, plot_mcnemar_matrix,
    plot_model_efficiency
)


def load_all_data():
    """Load all experiment results."""
    summary_path = os.path.join(RESULTS_DIR, 'all_experiments_summary.json')
    with open(summary_path) as f:
        data = json.load(f)
    return data


def main():
    print("=" * 60)
    print("REGENERATE FIGURES AS PDF VECTORS")
    print("=" * 60)

    data = load_all_data()
    results = data['results']

    for ds_name, model_results in results.items():
        if not model_results:
            continue

        print(f"\n--- {ds_name.upper()} ---")

        # Load evaluation data
        eval_data = {}
        histories = {}
        for m_name in model_results:
            test_path = os.path.join(RESULTS_DIR, 'models', ds_name,
                                     m_name, 'test_results.npz')
            hist_path = os.path.join(RESULTS_DIR, 'models', ds_name,
                                     m_name, 'training_history.npz')
            metrics_path = os.path.join(RESULTS_DIR, 'models', ds_name,
                                        m_name, 'evaluation_metrics.json')

            if os.path.exists(test_path) and os.path.exists(metrics_path):
                npz = np.load(test_path, allow_pickle=True)
                with open(metrics_path) as f:
                    metrics = json.load(f)
                eval_data[m_name] = {
                    'y_true': npz['y_true'],
                    'y_pred': npz['y_pred'],
                    'accuracy': metrics['accuracy'],
                    'per_class': metrics['per_class'],
                }

            if os.path.exists(hist_path):
                npz = np.load(hist_path)
                histories[m_name] = {
                    'train_loss': npz['train_loss'],
                    'train_acc': npz['train_acc'],
                    'test_loss': npz['test_loss'],
                    'test_acc': npz['test_acc'],
                }

        # Get class names
        class_names = None
        for m_name in model_results:
            mp = os.path.join(RESULTS_DIR, 'models', ds_name,
                              m_name, 'evaluation_metrics.json')
            if os.path.exists(mp):
                with open(mp) as f:
                    class_names = json.load(f).get('class_names')
                break

        if not class_names:
            continue

        # Confusion matrices (PDF)
        if eval_data:
            plot_confusion_matrices_grid(
                eval_data, class_names, ds_name,
                save_path=os.path.join(FIG_DIR, f'cm_{ds_name}.pdf'))

        # Training curves (PDF)
        if histories:
            plot_training_curves(
                histories,
                save_path=os.path.join(FIG_DIR, f'curves_{ds_name}.pdf'))

        # Per-class F1 (PDF)
        if eval_data:
            plot_per_class_f1(
                eval_data, class_names, ds_name,
                save_path=os.path.join(FIG_DIR, f'per_class_f1_{ds_name}.pdf'))

    # Cross-dataset accuracy comparison (PDF)
    acc_summary = {}
    for ds_name in results:
        acc_summary[ds_name] = {}
        for m_name in results[ds_name]:
            if 'error' not in results[ds_name][m_name]:
                acc_summary[ds_name][m_name] = results[ds_name][m_name].get(
                    'accuracy', 0)
    if acc_summary:
        plot_accuracy_comparison(
            acc_summary,
            save_path=os.path.join(FIG_DIR, 'accuracy_comparison.pdf'))

    # Model efficiency plot (PDF)
    for ds_name in results:
        accs = {m: r.get('accuracy', 0) for m, r in results[ds_name].items()
                if 'error' not in r}
        if accs:
            plot_model_efficiency(
                MODELS, accs,
                save_path=os.path.join(FIG_DIR, f'efficiency_{ds_name}.pdf'))
            break

    # McNemar heatmaps (need to recompute)
    from sklearn.metrics import confusion_matrix as sk_cm
    from scipy.stats import chi2

    for ds_name, model_results in results.items():
        eval_data_mcn = {}
        for m_name in model_results:
            test_path = os.path.join(RESULTS_DIR, 'models', ds_name,
                                     m_name, 'test_results.npz')
            if os.path.exists(test_path):
                npz = np.load(test_path, allow_pickle=True)
                eval_data_mcn[m_name] = {
                    'y_true': npz['y_true'],
                    'y_pred': npz['y_pred'],
                }

        if len(eval_data_mcn) < 2:
            continue

        model_names = list(eval_data_mcn.keys())
        comparisons = []
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                a, b = model_names[i], model_names[j]
                y_true = eval_data_mcn[a]['y_true']
                pred_a = eval_data_mcn[a]['y_pred']
                pred_b = eval_data_mcn[b]['y_pred']

                correct_a = (pred_a == y_true)
                correct_b = (pred_b == y_true)

                n01 = np.sum(correct_a & ~correct_b)
                n10 = np.sum(~correct_a & correct_b)

                if n01 + n10 > 0:
                    chi2_stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
                    p_value = 1 - chi2.cdf(chi2_stat, df=1)
                else:
                    chi2_stat = 0.0
                    p_value = 1.0

                comparisons.append({
                    'model_a': a, 'model_b': b,
                    'chi2': chi2_stat, 'p_value': p_value
                })

        plot_mcnemar_matrix(
            comparisons, model_names,
            save_path=os.path.join(FIG_DIR, f'mcnemar_{ds_name}.pdf'))

    print("\n" + "=" * 60)
    print("All PDF vector figures generated!")
    # List output files
    for f in sorted(os.listdir(FIG_DIR)):
        if f.endswith('.pdf'):
            size = os.path.getsize(os.path.join(FIG_DIR, f))
            print(f"  {f}: {size / 1024:.0f} KB")


if __name__ == '__main__':
    main()
