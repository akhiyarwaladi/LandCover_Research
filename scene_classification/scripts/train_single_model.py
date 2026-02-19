"""
Train a Single Model on a Single Dataset

Usage:
    python scripts/train_single_model.py --model resnet50 --dataset eurosat
    python scripts/train_single_model.py --model vit_b_16 --dataset nwpu_resisc45
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASETS, MODELS, TRAINING
from modules.dataset_loader import create_dataloaders
from modules.models import create_model, count_parameters
from modules.trainer import train_model
from modules.evaluator import evaluate_model, save_evaluation_results


def main():
    parser = argparse.ArgumentParser(description='Train a scene classifier')
    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODELS.keys()))
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASETS.keys()))
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    ds_config = DATASETS[args.dataset]
    train_loader, test_loader, class_names = create_dataloaders(
        args.dataset,
        batch_size=args.batch_size or TRAINING['batch_size'],
    )

    # Create model
    model = create_model(args.model, ds_config['num_classes'])
    total, trainable = count_parameters(model)
    print(f"\nModel: {args.model}")
    print(f"  Parameters: {total:,} ({total/1e6:.1f}M)")

    # Train
    print(f"\nTraining on {args.dataset}...")
    result = train_model(
        model, train_loader, test_loader,
        num_classes=ds_config['num_classes'],
        model_name=args.model,
        dataset_name=args.dataset,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )

    # Evaluate
    print(f"\nEvaluating...")
    eval_results = evaluate_model(model, test_loader, class_names, device)
    save_evaluation_results(eval_results, args.model, args.dataset)

    print(f"\nResults:")
    print(f"  Accuracy:    {eval_results['accuracy']:.4f}")
    print(f"  F1-Macro:    {eval_results['f1_macro']:.4f}")
    print(f"  F1-Weighted: {eval_results['f1_weighted']:.4f}")
    print(f"  Kappa:       {eval_results['kappa']:.4f}")


if __name__ == '__main__':
    main()
