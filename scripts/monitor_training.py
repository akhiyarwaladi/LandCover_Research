#!/usr/bin/env python3
"""
Quick Training Monitor - Check Current Training Progress
"""

import numpy as np
import os
import time

results_dir = 'results/models/resnet50'

print("🔍 Monitoring ResNet-50 Training Progress...\n")

while True:
    history_file = f'{results_dir}/training_history.npz'

    if os.path.exists(history_file):
        history = np.load(history_file)

        train_loss = history['train_loss']
        train_acc = history['train_acc']
        val_loss = history['val_loss']
        val_acc = history['val_acc']

        epochs_completed = len(train_loss)

        print(f"\r[{time.strftime('%H:%M:%S')}] Epochs: {epochs_completed}/30 | "
              f"Train Acc: {train_acc[-1]*100:.2f}% | "
              f"Val Acc: {val_acc[-1]*100:.2f}% | "
              f"Train Loss: {train_loss[-1]:.4f} | "
              f"Val Loss: {val_loss[-1]:.4f}", end='', flush=True)

        if epochs_completed >= 30:
            print("\n\n✅ Training complete!")
            break
    else:
        print(f"\r[{time.strftime('%H:%M:%S')}] Waiting for training to start...", end='', flush=True)

    time.sleep(5)  # Check every 5 seconds
