"""
Training Loop for Scene Classification Models

Handles training, validation, early stopping, and checkpointing.
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAINING, RESULTS_DIR


def train_model(model, train_loader, test_loader, num_classes,
                model_name='model', dataset_name='dataset',
                epochs=None, lr=None, device=None, verbose=True):
    """
    Train a model with early stopping and LR scheduling.

    Args:
        model: nn.Module
        train_loader: training DataLoader
        test_loader: test DataLoader
        num_classes: number of classes
        model_name: name for saving
        dataset_name: dataset name for saving
        epochs: max epochs (default from config)
        lr: learning rate (default from config)
        device: torch device
        verbose: print progress

    Returns:
        dict with training history and best metrics
    """
    if epochs is None:
        epochs = TRAINING['epochs']
    if lr is None:
        lr = TRAINING['learning_rate']
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr,
                      weight_decay=TRAINING['weight_decay'])
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max',
        patience=TRAINING['scheduler_patience'],
        factor=TRAINING['scheduler_factor'],
    )

    # Output directory
    save_dir = os.path.join(RESULTS_DIR, 'models', dataset_name, model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'lr': [],
    }

    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- Evaluate ---
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        test_loss /= test_total
        test_acc = test_correct / test_total

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(test_acc)

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['lr'].append(current_lr)

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'best_model.pth'))
        else:
            patience_counter += 1

        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train: {train_acc:.4f} | "
                  f"Test: {test_acc:.4f} | "
                  f"Best: {best_acc:.4f} (E{best_epoch}) | "
                  f"LR: {current_lr:.1e}")

        # Early stopping
        if patience_counter >= TRAINING['early_stopping_patience']:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    total_time = time.time() - start_time

    # Save training history
    np.savez(os.path.join(save_dir, 'training_history.npz'),
             **{k: np.array(v) for k, v in history.items()})

    # Load best model for final evaluation
    model.load_state_dict(
        torch.load(os.path.join(save_dir, 'best_model.pth'),
                    map_location=device, weights_only=True))

    result = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'total_epochs': len(history['train_loss']),
        'training_time': total_time,
        'history': history,
    }

    # Save summary
    summary = {k: v for k, v in result.items() if k != 'history'}
    with open(os.path.join(save_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"  Finished: {best_acc:.4f} acc @ epoch {best_epoch} "
              f"({total_time:.1f}s)")

    return result
