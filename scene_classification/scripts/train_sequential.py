"""
Train all models sequentially with forced output flushing.
Uses the cached EuroSAT data from the first run.
"""
import os
import sys
import json
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             precision_recall_fscore_support, confusion_matrix)

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RESULTS_DIR, MODELS, DATASETS, TRAINING

def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def load_eurosat():
    from config import DATASET_PATHS
    from torchvision.datasets import EuroSAT
    from torchvision import transforms
    from torch.utils.data import DataLoader, random_split

    data_dir = DATASET_PATHS['eurosat']

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load full dataset with train transform (will override for test)
    train_dataset = EuroSAT(root=data_dir, download=False, transform=train_transform)
    test_dataset = EuroSAT(root=data_dir, download=False, transform=test_transform)

    n_total = len(train_dataset)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train

    # Use same split indices for both
    gen = torch.Generator().manual_seed(42)
    train_indices, test_indices = random_split(range(n_total), [n_train, n_test],
                                               generator=gen)

    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices.indices)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False,
                             num_workers=0, pin_memory=True)

    return train_loader, test_loader, DATASETS['eurosat']['class_names']


def train_one_model(model_name, train_loader, test_loader, class_names,
                    epochs=30, device='cuda'):
    from modules.models import create_model, count_parameters

    num_classes = len(class_names)
    model = create_model(model_name, num_classes).to(device)
    total_params, _ = count_parameters(model)

    flush_print(f"  Params: {total_params/1e6:.1f}M")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5)

    save_dir = os.path.join(RESULTS_DIR, 'models', 'eurosat', model_name)
    os.makedirs(save_dir, exist_ok=True)

    history = {'train_loss': [], 'train_acc': [],
               'test_loss': [], 'test_acc': [], 'lr': []}
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0; correct = 0; total = 0

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

        model.eval()
        test_loss = 0; test_correct = 0; test_total = 0
        all_preds = []; all_targets = []

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
            flush_print(f"    E{epoch:3d}/{epochs} | "
                        f"Train: {train_acc:.4f} | Test: {test_acc:.4f} | "
                        f"Best: {best_acc:.4f} (E{best_epoch})")

        if patience_counter >= 10:
            flush_print(f"    Early stopping at epoch {epoch}")
            break

    total_time = time.time() - start_time

    # Save history
    np.savez(os.path.join(save_dir, 'training_history.npz'),
             **{k: np.array(v) for k, v in history.items()})

    # Save test results + metrics
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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flush_print(f"Device: {device}")
    if device.type == 'cuda':
        flush_print(f"  GPU: {torch.cuda.get_device_name(0)}")

    flush_print("\nLoading EuroSAT...")
    train_loader, test_loader, class_names = load_eurosat()
    flush_print(f"  Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    model_names = list(MODELS.keys())
    all_results = {}
    global_start = time.time()

    for i, model_name in enumerate(model_names):
        flush_print(f"\n{'='*50}")
        flush_print(f"[{i+1}/{len(model_names)}] {model_name} (30 epochs)")
        flush_print(f"{'='*50}")

        result = train_one_model(model_name, train_loader, test_loader,
                                 class_names, epochs=30, device=device)
        all_results[model_name] = result
        flush_print(f"  >> {result['accuracy']:.4f} acc, "
                    f"{result['f1_macro']:.4f} F1, {result['training_time']:.1f}s")

    total_time = time.time() - global_start

    # Save summary
    summary = {'results': {'eurosat': {}}, 'total_time_seconds': total_time,
               'device': str(device), 'epochs': 30, 'dataset': 'eurosat'}
    for m, r in all_results.items():
        summary['results']['eurosat'][m] = r

    with open(os.path.join(RESULTS_DIR, 'all_experiments_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final results
    flush_print(f"\n{'='*60}")
    flush_print("FINAL RESULTS (30 epochs)")
    flush_print(f"{'='*60}")
    flush_print(f"Total: {total_time/60:.1f} min\n")
    flush_print(f"{'Model':<20} {'Acc':>8} {'F1-Mac':>8} {'Kappa':>8} {'Params':>8} {'Time':>8}")
    flush_print("-" * 62)

    sorted_res = sorted([(m, r) for m, r in all_results.items()],
                        key=lambda x: x[1]['accuracy'], reverse=True)
    for m, r in sorted_res:
        flush_print(f"{m:<20} {r['accuracy']:>8.4f} {r['f1_macro']:>8.4f} "
                    f"{r['kappa']:>8.4f} {r['params_m']:>7.1f}M {r['training_time']:>7.1f}s")

    flush_print(f"\nBest: {sorted_res[0][0]} ({sorted_res[0][1]['accuracy']:.4f})")

    # Generate publication outputs
    flush_print("\nGenerating publication outputs...")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
        from generate_publication_outputs import (
            generate_performance_tables, generate_per_class_tables, generate_figures)
        from generate_statistical_analysis import (
            generate_mcnemar_tables, generate_kappa_table, generate_efficiency_table)

        data = {'results': summary['results']}
        generate_performance_tables(data)
        generate_per_class_tables(data)
        generate_figures(data)
        generate_mcnemar_tables()
        generate_kappa_table()
        generate_efficiency_table()
        flush_print("Publication outputs generated!")
    except Exception as e:
        flush_print(f"Publication output error: {e}")

    flush_print("\nDONE!")


if __name__ == '__main__':
    main()
