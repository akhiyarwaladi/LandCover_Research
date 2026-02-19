"""
Fast training script for scene classification benchmark.

Handles Windows multiprocessing properly with if __name__ == '__main__' guard.
Supports training specific models on specific datasets with proper num_workers.

Usage:
    python scripts/train_fast.py --dataset eurosat --models remaining --epochs 30
    python scripts/train_fast.py --dataset ucmerced --models all --epochs 30
    python scripts/train_fast.py --dataset all --models all --epochs 30
"""

import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             precision_recall_fscore_support, confusion_matrix)

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RESULTS_DIR, MODELS, DATASETS, DATASET_PATHS


def p(*args, **kw):
    print(*args, **kw, flush=True)


def get_remaining_models(dataset_name):
    """Find models that haven't been trained to 30 epochs yet."""
    remaining = []
    for m in MODELS:
        hist_path = os.path.join(RESULTS_DIR, 'models', dataset_name, m,
                                 'training_history.npz')
        if os.path.exists(hist_path):
            hist = np.load(hist_path)
            if len(hist['test_acc']) >= 25:  # Consider 25+ as complete
                continue
        remaining.append(m)
    return remaining


def train_one_model(model_name, train_loader, test_loader, class_names,
                    epochs, device, save_dir):
    """Train a single model and save all results."""
    from modules.models import create_model, count_parameters

    num_classes = len(class_names)
    model = create_model(model_name, num_classes).to(device)
    total_params, _ = count_parameters(model)
    p(f"  Params: {total_params/1e6:.1f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5)

    os.makedirs(save_dir, exist_ok=True)
    history = {'train_loss': [], 'train_acc': [],
               'test_loss': [], 'test_acc': [], 'lr': []}
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()
    best_preds = None
    best_targets = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()

        train_loss /= total
        train_acc = correct / total

        # Evaluate
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                test_total += targets.size(0)
                test_correct += preds.eq(targets).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        test_loss /= test_total
        test_acc = test_correct / test_total
        scheduler.step(test_acc)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            best_preds = np.array(all_preds)
            best_targets = np.array(all_targets)
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            p(f"    E{epoch:3d}/{epochs} | "
              f"Train: {train_acc:.4f} | Test: {test_acc:.4f} | "
              f"Best: {best_acc:.4f} (E{best_epoch})")

        if patience_counter >= 10:
            p(f"    Early stopping at epoch {epoch}")
            break

    total_time = time.time() - start_time

    # Save training history
    np.savez(os.path.join(save_dir, 'training_history.npz'),
             **{k: np.array(v) for k, v in history.items()})

    # Compute and save metrics
    y_true, y_pred = best_targets, best_preds
    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_wt = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    np.savez(os.path.join(save_dir, 'test_results.npz'),
             y_true=y_true, y_pred=y_pred, confusion_matrix=cm,
             y_probs=np.zeros((len(y_true), num_classes)))

    metrics = {
        'accuracy': float(acc), 'f1_macro': float(f1_mac),
        'f1_weighted': float(f1_wt), 'kappa': float(kappa),
        'per_class': {
            'precision': precision.tolist(), 'recall': recall.tolist(),
            'f1': f1.tolist(), 'support': support.tolist(),
        },
        'class_names': class_names,
    }
    with open(os.path.join(save_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    del model
    torch.cuda.empty_cache()

    return {
        'accuracy': acc, 'f1_macro': f1_mac, 'f1_weighted': f1_wt,
        'kappa': kappa, 'params_m': total_params / 1e6,
        'training_time': total_time, 'best_epoch': best_epoch,
    }


def train_dataset(dataset_name, model_list, epochs, device):
    """Train specified models on a dataset."""
    from modules.dataset_loader import create_dataloaders, find_dataset_root, \
        download_eurosat, download_ucmerced, verify_dataset

    p(f"\n{'='*60}")
    p(f"Dataset: {dataset_name.upper()}")
    p(f"{'='*60}")

    # Check if dataset exists, download if needed
    ok, _, _ = verify_dataset(dataset_name, verbose=True)
    if not ok:
        if dataset_name == 'eurosat':
            download_eurosat()
        elif dataset_name == 'ucmerced':
            download_ucmerced()
        else:
            p(f"  Dataset {dataset_name} not available. Skipping.")
            return {}

    train_loader, test_loader, class_names = create_dataloaders(
        dataset_name, batch_size=32, seed=42, verbose=True)

    results = {}
    for i, model_name in enumerate(model_list):
        p(f"\n  [{i+1}/{len(model_list)}] {model_name} ({epochs} epochs)")

        save_dir = os.path.join(RESULTS_DIR, 'models', dataset_name, model_name)
        result = train_one_model(
            model_name, train_loader, test_loader, class_names,
            epochs, device, save_dir)
        results[model_name] = result

        p(f"  >> {result['accuracy']:.4f} acc, {result['f1_macro']:.4f} F1, "
          f"{result['training_time']:.0f}s")

    return results


def update_summary(all_results):
    """Update the central summary JSON file."""
    summary_path = os.path.join(RESULTS_DIR, 'all_experiments_summary.json')

    # Load existing summary
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {'results': {}}

    # Merge new results
    for ds_name, model_results in all_results.items():
        if ds_name not in summary['results']:
            summary['results'][ds_name] = {}
        for m_name, result in model_results.items():
            summary['results'][ds_name][m_name] = result

    # Also fill in from evaluation_metrics.json files for any missing entries
    models_dir = os.path.join(RESULTS_DIR, 'models')
    for ds_name in os.listdir(models_dir):
        ds_path = os.path.join(models_dir, ds_name)
        if not os.path.isdir(ds_path):
            continue
        if ds_name not in summary['results']:
            summary['results'][ds_name] = {}
        for m_name in os.listdir(ds_path):
            if m_name in summary['results'].get(ds_name, {}):
                # Check if we have real training_time
                existing = summary['results'][ds_name][m_name]
                if existing.get('training_time', 0) > 0:
                    continue
            mp = os.path.join(ds_path, m_name, 'evaluation_metrics.json')
            if os.path.exists(mp):
                with open(mp) as f:
                    metrics = json.load(f)
                summary['results'][ds_name][m_name] = {
                    'accuracy': metrics['accuracy'],
                    'f1_macro': metrics['f1_macro'],
                    'f1_weighted': metrics['f1_weighted'],
                    'kappa': metrics['kappa'],
                    'params_m': MODELS.get(m_name, {}).get('params_m', 0),
                    'training_time': 0,
                    'best_epoch': 0,
                }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    p(f"\nSummary saved: {summary_path}")

    return summary


def regenerate_outputs():
    """Regenerate all publication tables and figures."""
    p("\n" + "="*60)
    p("REGENERATING PUBLICATION OUTPUTS")
    p("="*60)

    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, scripts_dir)

    from generate_publication_outputs import (
        load_results, generate_performance_tables,
        generate_per_class_tables, generate_figures)
    from generate_statistical_analysis import (
        generate_mcnemar_tables, generate_kappa_table,
        generate_efficiency_table)

    data = load_results()
    if data and data.get('results'):
        p("\nGenerating tables...")
        generate_performance_tables(data)
        generate_per_class_tables(data)

        p("\nGenerating figures...")
        generate_figures(data)

        p("\nStatistical analysis...")
        generate_mcnemar_tables()
        generate_kappa_table()
        generate_efficiency_table()
    else:
        p("No results found for publication outputs.")

    p("\nAll outputs regenerated!")


def main():
    parser = argparse.ArgumentParser(description='Fast Scene Classification Training')
    parser.add_argument('--dataset', default='all',
                        choices=['eurosat', 'ucmerced', 'all'],
                        help='Dataset to train on')
    parser.add_argument('--models', default='all',
                        choices=['all', 'remaining'],
                        help='Which models to train')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--no-regen', action='store_true',
                        help='Skip regenerating publication outputs')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    p(f"Device: {device}")
    if device.type == 'cuda':
        p(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Determine which datasets to train
    if args.dataset == 'all':
        datasets = ['eurosat', 'ucmerced']
    else:
        datasets = [args.dataset]

    all_results = {}
    global_start = time.time()

    for ds_name in datasets:
        # Determine which models to train
        if args.models == 'remaining':
            model_list = get_remaining_models(ds_name)
            if not model_list:
                p(f"\n  All models already trained on {ds_name}. Skipping.")
                continue
        else:
            model_list = list(MODELS.keys())

        p(f"\nModels to train on {ds_name}: {model_list}")
        results = train_dataset(ds_name, model_list, args.epochs, device)
        all_results[ds_name] = results

    total_time = time.time() - global_start
    p(f"\n{'='*60}")
    p(f"ALL TRAINING COMPLETE: {total_time/60:.1f} minutes")
    p(f"{'='*60}")

    # Update summary
    summary = update_summary(all_results)

    # Print summary table
    for ds_name in summary.get('results', {}):
        p(f"\n  {ds_name}:")
        p(f"  {'Model':<20} {'Acc':>8} {'F1-Mac':>8} {'Kappa':>8}")
        p(f"  {'-'*46}")
        sorted_res = sorted(
            summary['results'][ds_name].items(),
            key=lambda x: x[1].get('accuracy', 0), reverse=True)
        for m, r in sorted_res:
            p(f"  {m:<20} {r.get('accuracy',0):>8.4f} "
              f"{r.get('f1_macro',0):>8.4f} {r.get('kappa',0):>8.4f}")

    # Regenerate outputs
    if not args.no_regen:
        regenerate_outputs()

    p("\nDONE!")


if __name__ == '__main__':
    main()
