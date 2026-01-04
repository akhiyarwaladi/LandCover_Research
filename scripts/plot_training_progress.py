#!/usr/bin/env python3
"""
Real-time Training Progress Plotter
Monitors training_history.npz and plots accuracy/loss curves
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time

def plot_training_curves(model_name='resnet50', refresh_interval=10):
    """
    Plot training curves in real-time.

    Args:
        model_name: Model being trained
        refresh_interval: Seconds between plot updates
    """

    history_file = f'results/models/{model_name}/training_history.npz'

    plt.ion()  # Interactive mode
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{model_name.upper()} Training Progress (Real-time)', fontsize=16, fontweight='bold')

    print(f"📊 Monitoring training progress for {model_name}...")
    print(f"   Refreshing every {refresh_interval} seconds")
    print(f"   Press Ctrl+C to stop\n")

    last_epoch = 0

    try:
        while True:
            if os.path.exists(history_file):
                history = np.load(history_file)

                train_loss = history['train_loss']
                train_acc = history['train_acc']
                val_loss = history['val_loss']
                val_acc = history['val_acc']

                epochs = np.arange(1, len(train_loss) + 1)
                current_epoch = len(train_loss)

                # Only update if new data
                if current_epoch > last_epoch:
                    # Clear previous plots
                    ax1.clear()
                    ax2.clear()
                    ax3.clear()
                    ax4.clear()

                    # Plot 1: Training & Validation Accuracy
                    ax1.plot(epochs, train_acc * 100, 'b-o', label='Train Accuracy', linewidth=2, markersize=6)
                    ax1.plot(epochs, val_acc * 100, 'r-s', label='Val Accuracy', linewidth=2, markersize=6)
                    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
                    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
                    ax1.set_title('Accuracy over Epochs', fontsize=14, fontweight='bold')
                    ax1.legend(fontsize=11)
                    ax1.grid(True, alpha=0.3)
                    ax1.set_ylim([0, 100])

                    # Plot 2: Training & Validation Loss
                    ax2.plot(epochs, train_loss, 'b-o', label='Train Loss', linewidth=2, markersize=6)
                    ax2.plot(epochs, val_loss, 'r-s', label='Val Loss', linewidth=2, markersize=6)
                    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
                    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
                    ax2.set_title('Loss over Epochs', fontsize=14, fontweight='bold')
                    ax2.legend(fontsize=11)
                    ax2.grid(True, alpha=0.3)

                    # Plot 3: Accuracy Difference (Val - Train)
                    acc_diff = (val_acc - train_acc) * 100
                    colors = ['green' if x >= 0 else 'red' for x in acc_diff]
                    ax3.bar(epochs, acc_diff, color=colors, alpha=0.6)
                    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
                    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
                    ax3.set_ylabel('Val Acc - Train Acc (%)', fontsize=12, fontweight='bold')
                    ax3.set_title('Generalization Gap', fontsize=14, fontweight='bold')
                    ax3.grid(True, alpha=0.3, axis='y')

                    # Plot 4: Current Stats Table
                    ax4.axis('off')

                    stats_text = f"""
                    CURRENT STATS (Epoch {current_epoch}/30)
                    ═══════════════════════════════════════

                    Training Accuracy:     {train_acc[-1]*100:6.2f}%
                    Validation Accuracy:   {val_acc[-1]*100:6.2f}%

                    Training Loss:         {train_loss[-1]:8.4f}
                    Validation Loss:       {val_loss[-1]:8.4f}

                    ───────────────────────────────────────
                    Best Val Accuracy:     {max(val_acc)*100:6.2f}%
                    Best Val Epoch:        {np.argmax(val_acc)+1:6d}

                    Avg Train Acc:         {np.mean(train_acc)*100:6.2f}%
                    Avg Val Acc:           {np.mean(val_acc)*100:6.2f}%
                    """

                    ax4.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                            verticalalignment='center', bbox=dict(boxstyle='round',
                            facecolor='wheat', alpha=0.3))

                    plt.tight_layout()
                    plt.pause(0.1)

                    # Console update
                    print(f"\r[Epoch {current_epoch:2d}/30] "
                          f"Train: {train_acc[-1]*100:5.2f}% | "
                          f"Val: {val_acc[-1]*100:5.2f}% | "
                          f"Best: {max(val_acc)*100:5.2f}%", end='', flush=True)

                    last_epoch = current_epoch

                    # Check if training complete
                    if current_epoch >= 30:
                        print("\n\n✅ Training complete! Final plot displayed.")
                        plt.ioff()
                        plt.show()
                        break
            else:
                print(f"\r⏳ Waiting for training to start...", end='', flush=True)

            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped by user.")
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Monitor and plot training progress')
    parser.add_argument('--model', default='resnet50', help='Model name to monitor')
    parser.add_argument('--interval', type=int, default=10, help='Refresh interval in seconds')

    args = parser.parse_args()

    plot_training_curves(args.model, args.interval)
