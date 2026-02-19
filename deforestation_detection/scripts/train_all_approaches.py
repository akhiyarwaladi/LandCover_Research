"""
Train All Change Detection Approaches

Runs all 3 approaches sequentially:
    A. Post-Classification Comparison (PCC) with ResNet-101
    B. Siamese CNN with ResNet-50 backbone
    C. Random Forest with stacked temporal features

Usage:
    python scripts/train_all_approaches.py

Output:
    results/models/pcc_resnet101/
    results/models/siamese_resnet50/
    results/models/rf_change/
"""

import os
import sys
import time
import subprocess
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, 'scripts')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def run_script(script_name, args=None):
    """
    Run a training script and capture output.

    Args:
        script_name: Script filename
        args: Optional list of command-line arguments

    Returns:
        (success: bool, elapsed_time: float)
    """
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)

    print(f"\n{'='*60}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd, capture_output=False, text=True, cwd=BASE_DIR
        )
        elapsed = time.time() - start_time
        success = result.returncode == 0

        if success:
            print(f"\n  COMPLETED in {elapsed:.1f}s")
        else:
            print(f"\n  FAILED (exit code {result.returncode}) after {elapsed:.1f}s")

        return success, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n  ERROR: {e}")
        return False, elapsed


def main():
    """Train all 3 change detection approaches."""
    print("=" * 60)
    print("MULTI-TEMPORAL DEFORESTATION DETECTION")
    print("Training All Approaches")
    print("=" * 60)

    total_start = time.time()
    results = {}

    # Approach A: PCC with ResNet-101
    print("\n" + "#" * 60)
    print("# APPROACH A: Post-Classification Comparison (PCC)")
    print("#" * 60)
    success, elapsed = run_script('train_pcc_resnet.py', ['--variant', 'resnet101'])
    results['PCC-ResNet101'] = {'success': success, 'time': elapsed}

    # Approach B: Siamese CNN
    print("\n" + "#" * 60)
    print("# APPROACH B: Siamese CNN")
    print("#" * 60)
    success, elapsed = run_script('train_siamese_network.py', ['--backbone', 'resnet50'])
    results['Siamese-ResNet50'] = {'success': success, 'time': elapsed}

    # Approach C: Random Forest
    print("\n" + "#" * 60)
    print("# APPROACH C: Random Forest")
    print("#" * 60)
    success, elapsed = run_script('train_rf_change_detection.py')
    results['RF-Change'] = {'success': success, 'time': elapsed}

    # Summary
    total_time = time.time() - total_start

    print("\n" + "=" * 60)
    print("ALL APPROACHES COMPLETE")
    print("=" * 60)
    print(f"\n{'Approach':<25} {'Status':<10} {'Time':>10}")
    print("-" * 50)

    for name, info in results.items():
        status = 'OK' if info['success'] else 'FAILED'
        time_str = f"{info['time']:.1f}s"
        print(f"{name:<25} {status:<10} {time_str:>10}")

    print(f"\n  Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, 'all_approaches_summary.json')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary = {
        'approaches': results,
        'total_time_seconds': total_time,
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Summary saved: {summary_path}")

    # Check for results
    print("\n  Model outputs:")
    for approach_dir in ['pcc_resnet101', 'siamese_resnet50', 'rf_change']:
        model_dir = os.path.join(RESULTS_DIR, 'models', approach_dir)
        if os.path.exists(model_dir):
            files = os.listdir(model_dir)
            print(f"    {approach_dir}/: {len(files)} files")
        else:
            print(f"    {approach_dir}/: NOT FOUND")


if __name__ == '__main__':
    main()
