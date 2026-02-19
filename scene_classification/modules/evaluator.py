"""
Model Evaluation for Scene Classification

Computes comprehensive metrics: accuracy, F1, Kappa, per-class performance,
McNemar's test for pairwise model comparison.
"""

import os
import json
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    classification_report, confusion_matrix, precision_recall_fscore_support
)
from scipy.stats import chi2

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR


def evaluate_model(model, test_loader, class_names, device=None):
    """
    Evaluate a model on the test set.

    Returns:
        dict with predictions, targets, and all metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_wt = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    results = {
        'accuracy': float(acc),
        'f1_macro': float(f1_mac),
        'f1_weighted': float(f1_wt),
        'kappa': float(kappa),
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_probs': y_probs,
        'per_class': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist(),
        },
        'class_names': class_names,
    }

    return results


def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """
    McNemar's test for comparing two classifiers.

    Returns:
        chi2_stat, p_value
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # b01: A wrong, B right
    b01 = np.sum(~correct_a & correct_b)
    # b10: A right, B wrong
    b10 = np.sum(correct_a & ~correct_b)

    if b01 + b10 == 0:
        return 0.0, 1.0

    chi2_stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)

    return float(chi2_stat), float(p_value)


def run_pairwise_mcnemar(results_dict):
    """
    Run McNemar's test for all pairs of models.

    Args:
        results_dict: {model_name: evaluation_results}

    Returns:
        list of dicts with pairwise comparisons
    """
    model_names = list(results_dict.keys())
    comparisons = []

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name_a = model_names[i]
            name_b = model_names[j]
            res_a = results_dict[name_a]
            res_b = results_dict[name_b]

            chi2_stat, p_val = mcnemar_test(
                res_a['y_true'], res_a['y_pred'], res_b['y_pred'])

            sig = 'ns'
            if p_val < 0.001:
                sig = '***'
            elif p_val < 0.01:
                sig = '**'
            elif p_val < 0.05:
                sig = '*'

            comparisons.append({
                'model_a': name_a,
                'model_b': name_b,
                'chi2': chi2_stat,
                'p_value': p_val,
                'significance': sig,
                'acc_a': res_a['accuracy'],
                'acc_b': res_b['accuracy'],
            })

    return comparisons


def save_evaluation_results(results, model_name, dataset_name):
    """Save evaluation results to JSON and NPZ."""
    save_dir = os.path.join(RESULTS_DIR, 'models', dataset_name, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save predictions
    np.savez(os.path.join(save_dir, 'test_results.npz'),
             y_true=results['y_true'],
             y_pred=results['y_pred'],
             y_probs=results['y_probs'],
             confusion_matrix=results['confusion_matrix'])

    # Save metrics as JSON
    metrics = {
        'accuracy': results['accuracy'],
        'f1_macro': results['f1_macro'],
        'f1_weighted': results['f1_weighted'],
        'kappa': results['kappa'],
        'per_class': results['per_class'],
        'class_names': results['class_names'],
    }
    with open(os.path.join(save_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
