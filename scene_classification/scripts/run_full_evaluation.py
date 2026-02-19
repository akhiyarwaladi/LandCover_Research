"""
Full End-to-End Evaluation Pipeline

Downloads EuroSAT (if not present), trains all 8 models, evaluates,
and generates all publication outputs.

Usage:
    python scripts/run_full_evaluation.py
    python scripts/run_full_evaluation.py --quick    # Fewer epochs for testing
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR, MODELS, DATASETS, TRAINING


def try_load_eurosat():
    """Try to load EuroSAT via torchvision or from local files."""
    from config import DATASET_PATHS

    # Try local files first
    from modules.dataset_loader import verify_dataset, create_dataloaders
    ok, n_cls, n_img = verify_dataset('eurosat', verbose=False)
    if ok:
        print("Found local EuroSAT dataset")
        return create_dataloaders('eurosat')

    # Try torchvision EuroSAT
    try:
        from torchvision.datasets import EuroSAT
        from torchvision import transforms
        from torch.utils.data import random_split

        data_dir = DATASET_PATHS['eurosat']
        os.makedirs(data_dir, exist_ok=True)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        dataset = EuroSAT(root=data_dir, download=True, transform=transform)
        n_total = len(dataset)
        n_train = int(0.8 * n_total)
        n_test = n_total - n_train

        train_set, test_set = random_split(
            dataset, [n_train, n_test],
            generator=torch.Generator().manual_seed(42))

        # Apply training augmentation
        train_set.dataset.transform = train_transform

        class_names = DATASETS['eurosat']['class_names']

        train_loader = DataLoader(train_set, batch_size=32, shuffle=True,
                                  num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False,
                                 num_workers=2, pin_memory=True)

        print(f"Loaded EuroSAT via torchvision: {n_train} train, {n_test} test")
        return train_loader, test_loader, class_names

    except Exception as e:
        print(f"Could not load EuroSAT via torchvision: {e}")
        return None


def create_synthetic_dataset(num_classes=10, num_train=2000, num_test=500):
    """Create a synthetic dataset for testing the pipeline."""
    print(f"Creating synthetic dataset: {num_classes} classes, "
          f"{num_train} train, {num_test} test")

    # Create synthetic images (3, 224, 224) with class-specific patterns
    np.random.seed(42)
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for cls in range(num_classes):
        n_tr = num_train // num_classes
        n_te = num_test // num_classes

        # Each class gets a unique mean pattern
        base = np.random.randn(3, 224, 224).astype(np.float32) * 0.1
        base[cls % 3] += 0.5  # Channel bias per class
        # Add spatial pattern
        freq = (cls + 1) * 2
        x = np.linspace(0, freq * np.pi, 224)
        pattern = np.sin(x[None, :]) * np.cos(x[:, None])
        base += pattern[None, :, :] * 0.3

        for _ in range(n_tr):
            img = base + np.random.randn(3, 224, 224).astype(np.float32) * 0.2
            train_images.append(img)
            train_labels.append(cls)

        for _ in range(n_te):
            img = base + np.random.randn(3, 224, 224).astype(np.float32) * 0.2
            test_images.append(img)
            test_labels.append(cls)

    train_images = torch.tensor(np.array(train_images))
    train_labels = torch.tensor(train_labels)
    test_images = torch.tensor(np.array(test_images))
    test_labels = torch.tensor(test_labels)

    train_loader = DataLoader(
        TensorDataset(train_images, train_labels),
        batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(
        TensorDataset(test_images, test_labels),
        batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    class_names = [f'Class_{i}' for i in range(num_classes)]
    return train_loader, test_loader, class_names


def train_and_evaluate(model, train_loader, test_loader, class_names,
                       model_name, dataset_name, epochs, device):
    """Train and evaluate a single model."""
    from modules.models import count_parameters

    model = model.to(device)
    total_params, _ = count_parameters(model)
    num_classes = len(class_names)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5)

    save_dir = os.path.join(RESULTS_DIR, 'models', dataset_name, model_name)
    os.makedirs(save_dir, exist_ok=True)

    history = {'train_loss': [], 'train_acc': [],
               'test_loss': [], 'test_acc': [], 'lr': []}
    best_acc = 0.0
    best_epoch = 0
    start_time = time.time()

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
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'best_model.pth'))
            best_preds = np.array(all_preds)
            best_targets = np.array(all_targets)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"    E{epoch:3d}/{epochs} | "
                  f"Train: {train_acc:.4f} | "
                  f"Test: {test_acc:.4f} | "
                  f"Best: {best_acc:.4f} (E{best_epoch})")

    total_time = time.time() - start_time

    # Save history
    np.savez(os.path.join(save_dir, 'training_history.npz'),
             **{k: np.array(v) for k, v in history.items()})

    # Compute final metrics using best model predictions
    y_true = best_targets
    y_pred = best_preds
    acc = accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_wt = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Save test results
    np.savez(os.path.join(save_dir, 'test_results.npz'),
             y_true=y_true, y_pred=y_pred, confusion_matrix=cm,
             y_probs=np.zeros((len(y_true), len(class_names))))

    metrics = {
        'accuracy': float(acc),
        'f1_macro': float(f1_mac),
        'f1_weighted': float(f1_wt),
        'kappa': float(kappa),
        'per_class': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist(),
        },
        'class_names': class_names,
    }
    with open(os.path.join(save_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    return {
        'accuracy': acc,
        'f1_macro': f1_mac,
        'f1_weighted': f1_wt,
        'kappa': kappa,
        'params_m': total_params / 1e6,
        'training_time': total_time,
        'best_epoch': best_epoch,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (fewer epochs)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Specific models to train')
    args = parser.parse_args()

    epochs = 5 if args.quick else TRAINING['epochs']
    model_names = args.models or list(MODELS.keys())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Try to load EuroSAT
    print("\n" + "=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    result = try_load_eurosat()
    if result is not None:
        train_loader, test_loader, class_names = result
        dataset_name = 'eurosat'
    else:
        print("\nFalling back to synthetic dataset for pipeline testing...")
        train_loader, test_loader, class_names = create_synthetic_dataset(
            num_classes=10, num_train=2000, num_test=500)
        dataset_name = 'synthetic'

    num_classes = len(class_names)
    print(f"\nDataset: {dataset_name} ({num_classes} classes)")
    print(f"Models: {len(model_names)}")
    print(f"Epochs: {epochs}")

    # Train all models
    print("\n" + "=" * 60)
    print("TRAINING ALL MODELS")
    print("=" * 60)

    from modules.models import create_model

    all_results = {}
    global_start = time.time()

    for i, model_name in enumerate(model_names):
        print(f"\n--- [{i+1}/{len(model_names)}] {model_name} ---")
        try:
            model = create_model(model_name, num_classes)
            result = train_and_evaluate(
                model, train_loader, test_loader, class_names,
                model_name, dataset_name, epochs, device)
            all_results[model_name] = result
            print(f"  >> {result['accuracy']:.4f} acc, "
                  f"{result['f1_macro']:.4f} F1, "
                  f"{result['training_time']:.1f}s")

            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[model_name] = {'error': str(e)}

    total_time = time.time() - global_start

    # Save summary
    summary = {
        'results': {dataset_name: {}},
        'total_time_seconds': total_time,
        'device': str(device),
        'epochs': epochs,
        'dataset': dataset_name,
    }
    for m_name, m_res in all_results.items():
        summary['results'][dataset_name][m_name] = {
            k: v for k, v in m_res.items()
        }

    summary_path = os.path.join(RESULTS_DIR, 'all_experiments_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nDataset: {dataset_name}")
    print(f"Total time: {total_time/60:.1f} min")
    print(f"\n{'Model':<20} {'Acc':>8} {'F1-Mac':>8} {'Kappa':>8} "
          f"{'Params':>8} {'Time':>8}")
    print("-" * 60)

    sorted_results = sorted(
        [(m, r) for m, r in all_results.items() if 'error' not in r],
        key=lambda x: x[1]['accuracy'], reverse=True)

    for m_name, res in sorted_results:
        print(f"{m_name:<20} {res['accuracy']:>8.4f} {res['f1_macro']:>8.4f} "
              f"{res['kappa']:>8.4f} {res['params_m']:>7.1f}M "
              f"{res['training_time']:>7.1f}s")

    best_name, best_res = sorted_results[0]
    print(f"\nBest model: {best_name} ({best_res['accuracy']:.4f} accuracy)")

    # Generate publication outputs
    print("\n" + "=" * 60)
    print("GENERATING PUBLICATION OUTPUTS")
    print("=" * 60)

    try:
        from scripts.generate_publication_outputs import (
            generate_performance_tables, generate_per_class_tables,
            generate_figures
        )
        data = {'results': summary['results']}
        print("\nTables...")
        generate_performance_tables(data)
        generate_per_class_tables(data)
        print("\nFigures...")
        generate_figures(data)
    except Exception as e:
        print(f"Publication output error: {e}")
        # Try importing differently
        try:
            sys.path.insert(0, os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'scripts'))
            from generate_publication_outputs import (
                generate_performance_tables, generate_per_class_tables,
                generate_figures
            )
            data = {'results': summary['results']}
            generate_performance_tables(data)
            generate_per_class_tables(data)
            generate_figures(data)
        except Exception as e2:
            print(f"  Could not generate publication outputs: {e2}")
            print("  Run separately: python scripts/generate_publication_outputs.py")

    # Generate statistical analysis
    try:
        from scripts.generate_statistical_analysis import (
            generate_mcnemar_tables, generate_kappa_table,
            generate_efficiency_table
        )
        print("\nStatistical analysis...")
        generate_mcnemar_tables()
        generate_kappa_table()
        generate_efficiency_table()
    except Exception as e:
        print(f"  Statistical analysis: {e}")
        print("  Run separately: python scripts/generate_statistical_analysis.py")

    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    print(f"\nResults: {RESULTS_DIR}")
    print(f"  Models:  results/models/{dataset_name}/")
    print(f"  Tables:  results/tables/")
    print(f"  Figures: results/figures/")


if __name__ == '__main__':
    main()
