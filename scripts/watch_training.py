#!/usr/bin/env python3
"""
Simple text-based training monitor
"""

import numpy as np
import os
import time

def watch_training(model_name='resnet50'):
    """Watch training progress in terminal."""

    history_file = f'results/models/{model_name}/training_history.npz'

    print(f"\n{'='*80}")
    print(f"  WATCHING {model_name.upper()} TRAINING PROGRESS")
    print(f"{'='*80}\n")

    last_epoch = 0

    while True:
        if os.path.exists(history_file):
            history = np.load(history_file)

            train_loss = history['train_loss']
            train_acc = history['train_acc']
            val_loss = history['val_loss']
            val_acc = history['val_acc']

            current_epoch = len(train_loss)

            # Print header every 10 epochs
            if current_epoch == 1 or current_epoch % 10 == 0:
                print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>10} | {'Val Acc':>9} | {'Best Val':>9} |")
                print(f"{'-'*80}")

            # Only print if new data
            if current_epoch > last_epoch:
                best_val = max(val_acc) * 100
                marker = " ✓" if val_acc[-1] == max(val_acc) else ""

                print(f"{current_epoch:5d} | {train_loss[-1]:10.4f} | {train_acc[-1]*100:8.2f}% | "
                      f"{val_loss[-1]:10.4f} | {val_acc[-1]*100:8.2f}% | {best_val:8.2f}% {marker}")

                last_epoch = current_epoch

                # Check if complete
                if current_epoch >= 30:
                    print(f"\n{'='*80}")
                    print(f"  ✅ TRAINING COMPLETE!")
                    print(f"  Best Validation Accuracy: {max(val_acc)*100:.2f}% (Epoch {np.argmax(val_acc)+1})")
                    print(f"{'='*80}\n")
                    break
        else:
            print(f"\r⏳ Waiting for {model_name} training to start...", end='', flush=True)

        time.sleep(5)  # Check every 5 seconds


if __name__ == '__main__':
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else 'resnet50'

    try:
        watch_training(model)
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped.\n")
