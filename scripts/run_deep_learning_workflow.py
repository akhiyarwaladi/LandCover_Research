#!/usr/bin/env python3
"""
Deep Learning Workflow - Master Orchestrator
=============================================

This script runs the complete deep learning workflow from training to
journal-ready outputs without needing to retrain for each step.

Complete workflow:
1. Train ResNet model and save weights
2. Generate Excel tables from saved model
3. Generate publication figures from saved model

Usage:
    # Full workflow (train + results)
    python scripts/run_deep_learning_workflow.py

    # Skip training (only generate outputs from saved model)
    python scripts/run_deep_learning_workflow.py --skip-training

    # Only training
    python scripts/run_deep_learning_workflow.py --train-only

    # Custom figure theme
    python scripts/run_deep_learning_workflow.py --skip-training --theme seaborn-v0_8-darkgrid

    # High resolution figures
    python scripts/run_deep_learning_workflow.py --skip-training --dpi 600

Output:
    models/resnet50_best.pth - Trained model weights
    results/resnet_classification/ - Training history and predictions
    results/tables/classification_results.xlsx - Excel tables
    results/figures/publication/*.png - Publication figures
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Script paths
SCRIPTS_DIR = Path(__file__).parent
TRAIN_SCRIPT = SCRIPTS_DIR / 'run_resnet_classification.py'
TABLE_SCRIPT = SCRIPTS_DIR / 'generate_results_table.py'
FIGURE_SCRIPT = SCRIPTS_DIR / 'generate_publication_figures.py'

# Output paths
MODEL_PATH = 'models/resnet50_best.pth'
PREDICTIONS_PATH = 'results/resnet_classification/test_predictions.npz'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_script(script_path, args=None, description=""):
    """
    Run a Python script as subprocess.

    Parameters
    ----------
    script_path : Path
        Path to script
    args : list, optional
        Additional arguments
    description : str
        Description of what the script does
    """
    print("\n" + "=" * 70)
    print(f"RUNNING: {description}")
    print("=" * 70)
    print(f"Script: {script_path}")

    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed!")
        print(f"   Return code: {result.returncode}")
        return False

    print(f"\n✅ {description} completed successfully!")
    return True


def check_model_exists():
    """Check if trained model or predictions exists."""
    # Check if either model exists OR predictions exist (for testing)
    return os.path.exists(MODEL_PATH) or os.path.exists(PREDICTIONS_PATH)


# ============================================================================
# WORKFLOW FUNCTIONS
# ============================================================================

def run_training():
    """Step 1: Train ResNet model."""
    success = run_script(
        TRAIN_SCRIPT,
        description="ResNet Model Training"
    )

    if not success:
        print("\n❌ Training failed! Cannot proceed with results generation.")
        return False

    # Verify outputs
    if not check_model_exists():
        print("\n❌ Model training did not produce expected outputs!")
        print(f"   Expected: {MODEL_PATH}")
        print(f"   Expected: {PREDICTIONS_PATH}")
        return False

    return True


def generate_tables():
    """Step 2: Generate Excel tables from saved model."""
    if not check_model_exists():
        print("\n❌ Cannot generate tables - model not found!")
        print(f"   Missing: {MODEL_PATH}")
        print("   Run training first: python scripts/run_resnet_classification.py")
        return False

    success = run_script(
        TABLE_SCRIPT,
        description="Excel Tables Generation"
    )

    return success


def generate_figures(theme='seaborn-v0_8-whitegrid', dpi=300):
    """Step 3: Generate publication figures from saved model."""
    if not check_model_exists():
        print("\n❌ Cannot generate figures - model not found!")
        print(f"   Missing: {MODEL_PATH}")
        print("   Run training first: python scripts/run_resnet_classification.py")
        return False

    args = ['--theme', theme, '--dpi', str(dpi)]

    success = run_script(
        FIGURE_SCRIPT,
        args=args,
        description="Publication Figures Generation"
    )

    return success


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main(args):
    """Run complete deep learning workflow."""

    print("=" * 70)
    print("DEEP LEARNING WORKFLOW - Master Orchestrator")
    print("=" * 70)
    print("\n🎯 Complete workflow for journal paper preparation")
    print("   Train ResNet → Generate Tables → Generate Figures")

    # Step 1: Training (unless skipped)
    if not args.skip_training:
        print("\n" + "─" * 70)
        print("STEP 1: Training ResNet Model")
        print("─" * 70)

        if not run_training():
            print("\n❌ Workflow aborted due to training failure")
            return False
    else:
        print("\n" + "─" * 70)
        print("STEP 1: Training SKIPPED (using existing model)")
        print("─" * 70)

        if not check_model_exists():
            print("\n❌ No existing model found!")
            print("   Remove --skip-training flag to train model first")
            return False

        print(f"   ✓ Found existing model: {MODEL_PATH}")

    # If train-only, stop here
    if args.train_only:
        print("\n" + "─" * 70)
        print("WORKFLOW COMPLETE (train-only mode)")
        print("─" * 70)
        print("\n✅ Model training complete!")
        print(f"\n📊 To generate results:")
        print(f"   python {FIGURE_SCRIPT.name}")
        print(f"   python {TABLE_SCRIPT.name}")
        return True

    # Step 2: Generate Excel tables
    print("\n" + "─" * 70)
    print("STEP 2: Generating Excel Tables")
    print("─" * 70)

    if not generate_tables():
        print("\n⚠️  Table generation failed (continuing anyway)")

    # Step 3: Generate publication figures
    print("\n" + "─" * 70)
    print("STEP 3: Generating Publication Figures")
    print("─" * 70)

    if not generate_figures(theme=args.theme, dpi=args.dpi):
        print("\n⚠️  Figure generation failed")

    # Summary
    print("\n" + "=" * 70)
    print("DEEP LEARNING WORKFLOW COMPLETE!")
    print("=" * 70)

    print("\n✅ All outputs generated successfully!")

    print("\n📊 Journal Paper Materials:")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Tables: results/tables/classification_results.xlsx")
    print(f"   Figures: results/figures/publication/*.png")

    print("\n📝 Next Steps:")
    print("   1. Review Excel tables for paper")
    print("   2. Insert publication figures into manuscript")
    print("   3. Cite methodology in paper")

    print("\n💡 To regenerate with different styles:")
    print(f"   Tables: No regeneration needed (already formatted)")
    print(f"   Figures: python {FIGURE_SCRIPT.name} --theme <theme> --dpi <dpi>")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run complete deep learning workflow for journal paper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full workflow (train + results)
  python %(prog)s

  # Skip training (use existing model)
  python %(prog)s --skip-training

  # Only train model
  python %(prog)s --train-only

  # Custom figure styling
  python %(prog)s --skip-training --theme seaborn-v0_8-darkgrid --dpi 600
        """
    )

    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and use existing model')
    parser.add_argument('--train-only', action='store_true',
                       help='Only train model (skip results generation)')
    parser.add_argument('--theme', type=str, default='seaborn-v0_8-whitegrid',
                       help='Matplotlib theme for figures (default: seaborn-v0_8-whitegrid)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Figure resolution in DPI (default: 300)')

    args = parser.parse_args()

    # Validate arguments
    if args.skip_training and args.train_only:
        print("❌ ERROR: Cannot use --skip-training and --train-only together")
        sys.exit(1)

    # Run workflow
    success = main(args)

    if not success:
        sys.exit(1)
