"""
Train All Models on All Available Datasets

Runs the full benchmark: 8 models x available datasets.
Saves all results for publication output generation.

Usage:
    python scripts/train_all_experiments.py
    python scripts/train_all_experiments.py --dataset eurosat
    python scripts/train_all_experiments.py --models resnet50 vit_b_16
"""

import os
import sys
import json
import time
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASETS, MODELS, TRAINING, RESULTS_DIR
from modules.dataset_loader import create_dataloaders, verify_dataset
from modules.models import create_model, count_parameters
from modules.trainer import train_model
from modules.evaluator import evaluate_model, save_evaluation_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, nargs='+', default=None,
                        help='Specific datasets to run')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Specific models to run')
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Determine datasets and models
    dataset_names = args.dataset or list(DATASETS.keys())
    model_names = args.models or list(MODELS.keys())

    # Check which datasets are available
    available_datasets = []
    print("\nDataset availability:")
    for ds_name in dataset_names:
        ok, n_cls, n_img = verify_dataset(ds_name)
        if ok:
            available_datasets.append(ds_name)

    if not available_datasets:
        print("\nNo datasets available! Run scripts/download_datasets.py first.")
        return

    print(f"\nWill train {len(model_names)} models on {len(available_datasets)} datasets")
    total_experiments = len(model_names) * len(available_datasets)
    print(f"Total experiments: {total_experiments}")

    # Run experiments
    all_results = {}
    experiment_idx = 0
    global_start = time.time()

    for ds_name in available_datasets:
        print(f"\n{'='*60}")
        print(f"DATASET: {ds_name}")
        print(f"{'='*60}")

        ds_config = DATASETS[ds_name]
        train_loader, test_loader, class_names = create_dataloaders(ds_name)

        all_results[ds_name] = {}

        for model_name in model_names:
            experiment_idx += 1
            print(f"\n--- [{experiment_idx}/{total_experiments}] "
                  f"{model_name} on {ds_name} ---")

            try:
                # Create model
                model = create_model(model_name, ds_config['num_classes'])
                total_params, _ = count_parameters(model)
                print(f"  Parameters: {total_params/1e6:.1f}M")

                # Train
                train_result = train_model(
                    model, train_loader, test_loader,
                    num_classes=ds_config['num_classes'],
                    model_name=model_name,
                    dataset_name=ds_name,
                    epochs=args.epochs,
                    device=device,
                )

                # Evaluate
                eval_result = evaluate_model(
                    model, test_loader, class_names, device)
                save_evaluation_results(eval_result, model_name, ds_name)

                all_results[ds_name][model_name] = {
                    'accuracy': eval_result['accuracy'],
                    'f1_macro': eval_result['f1_macro'],
                    'f1_weighted': eval_result['f1_weighted'],
                    'kappa': eval_result['kappa'],
                    'params_m': total_params / 1e6,
                    'training_time': train_result['training_time'],
                    'best_epoch': train_result['best_epoch'],
                    'per_class_f1': eval_result['per_class']['f1'],
                }

                print(f"  Result: {eval_result['accuracy']:.4f} acc, "
                      f"{eval_result['f1_macro']:.4f} F1-macro")

                # Free GPU memory
                del model
                torch.cuda.empty_cache() if device.type == 'cuda' else None

            except Exception as e:
                print(f"  ERROR: {e}")
                all_results[ds_name][model_name] = {'error': str(e)}

    total_time = time.time() - global_start

    # Save combined results
    summary_path = os.path.join(RESULTS_DIR, 'all_experiments_summary.json')
    save_data = {
        'results': {},
        'total_time_seconds': total_time,
        'device': str(device),
    }
    for ds_name, models_dict in all_results.items():
        save_data['results'][ds_name] = {}
        for m_name, m_res in models_dict.items():
            save_data['results'][ds_name][m_name] = {
                k: v for k, v in m_res.items()
                if k != 'per_class_f1'  # Don't save lists to summary
            }

    with open(summary_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")

    for ds_name in available_datasets:
        print(f"\n{ds_name}:")
        for m_name in model_names:
            if m_name in all_results.get(ds_name, {}):
                res = all_results[ds_name][m_name]
                if 'error' in res:
                    print(f"  {m_name:20s}: ERROR - {res['error']}")
                else:
                    print(f"  {m_name:20s}: {res['accuracy']:.4f} acc, "
                          f"{res['f1_macro']:.4f} F1")

    print(f"\nResults saved to: {summary_path}")


if __name__ == '__main__':
    main()
