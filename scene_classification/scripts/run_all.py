"""
Run Everything: Train -> Evaluate -> Tables -> Figures

Single script that runs the entire experiment pipeline:
  1. Preflight checks (imports, GPU, disk space)
  2. Verify/download datasets
  3. Train all 8 models on all available datasets
  4. Evaluate all models and save results
  5. Generate publication Excel tables
  6. Generate all 16 PDF figures for the manuscript
  7. Compile LaTeX manuscript (optional)

Usage:
    python scripts/run_all.py                          # Full pipeline
    python scripts/run_all.py --check                  # Dry run: verify everything
    python scripts/run_all.py --skip-training           # Only regenerate outputs
    python scripts/run_all.py --dataset eurosat          # One dataset only
    python scripts/run_all.py --models resnet50 vit_b_16 # Specific models
    python scripts/run_all.py --force                   # Retrain even if models exist
    python scripts/run_all.py --fresh                   # Wipe data + results, redownload, retrain
    python scripts/run_all.py --log                     # Save output to log file

Adding new models or datasets:
    - Models: add entry to MODELS dict in config.py, then add architecture
      in modules/models.py create_model()
    - Datasets: add entry to DATASETS and DATASET_PATHS in config.py, then
      add folder-finding logic in modules/dataset_loader.py find_dataset_root()
    - This script picks up changes from config.py automatically.
"""

import os
import sys
import json
import time
import shutil
import logging
import argparse
import traceback
import subprocess
from datetime import datetime

# Setup path so we can import from project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from config import DATASETS, MODELS, TRAINING, RESULTS_DIR, DATA_DIR


# ============================================================
# Logging setup
# ============================================================

def setup_logging(use_log_file=False):
    """Configure logging to console and optionally to a file."""
    log_format = '%(message)s'
    handlers = [logging.StreamHandler(sys.stdout)]

    if use_log_file:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(RESULTS_DIR, f'pipeline_{timestamp}.log')
        handlers.append(logging.FileHandler(log_path, encoding='utf-8'))
        print(f"  Logging to: {log_path}")

    logging.basicConfig(level=logging.INFO, format=log_format, handlers=handlers)
    return logging.getLogger('pipeline')


def step_header(step_num, title, log):
    """Print a formatted step header with timestamp."""
    now = datetime.now().strftime('%H:%M:%S')
    log.info("")
    log.info("=" * 60)
    log.info(f"  STEP {step_num}: {title}  [{now}]")
    log.info("=" * 60)


# ============================================================
# Step 0: Preflight Checks
# ============================================================

def step_preflight(log):
    """Verify all imports and hardware before starting."""
    step_header(0, "PREFLIGHT CHECKS", log)

    errors = []

    # Check Python packages
    packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'torchvision'),
        ('sklearn', 'scikit-learn'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('openpyxl', 'openpyxl'),
    ]
    for pkg_import, pkg_name in packages:
        try:
            __import__(pkg_import)
            log.info(f"  [OK] {pkg_name}")
        except ImportError:
            errors.append(pkg_name)
            log.info(f"  [MISSING] {pkg_name}")

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            log.info(f"  [OK] GPU: {gpu_name} ({mem_gb:.1f} GB)")
        else:
            log.info("  [WARN] No GPU detected, will use CPU (slow)")
    except Exception as e:
        log.info(f"  [WARN] GPU check failed: {e}")

    # Check project modules
    for module_name in ['modules.dataset_loader', 'modules.models',
                        'modules.trainer', 'modules.evaluator']:
        try:
            __import__(module_name)
            log.info(f"  [OK] {module_name}")
        except ImportError as e:
            errors.append(module_name)
            log.info(f"  [MISSING] {module_name}: {e}")

    # Check subprocess scripts exist
    for script_rel in ['scripts/generate_publication_outputs.py',
                       'publication/manuscript/generate_all_figures.py']:
        script_path = os.path.join(PROJECT_ROOT, script_rel)
        if os.path.exists(script_path):
            log.info(f"  [OK] {script_rel}")
        else:
            log.info(f"  [WARN] {script_rel} not found (will skip)")

    if errors:
        log.info(f"\n  MISSING: {', '.join(errors)}")
        log.info("  Install with: pip install " + ' '.join(
            e.lower() for e in errors if not e.startswith('modules')))
        return False

    log.info("\n  All preflight checks passed.")
    return True


# ============================================================
# Step 0b: Fresh Download (wipe + redownload)
# ============================================================

def step_fresh_download(dataset_names, log):
    """Wipe data and results folders, then redownload datasets."""
    from modules.dataset_loader import (
        download_eurosat, download_ucmerced,
        download_nwpu_resisc45, download_aid
    )

    step_header("0b", "FRESH DOWNLOAD (wipe + redownload)", log)

    # --- Wipe data folder ---
    for ds_name in dataset_names:
        from config import DATASET_PATHS
        ds_path = DATASET_PATHS.get(ds_name)
        if ds_path and os.path.isdir(ds_path):
            log.info(f"  Deleting {ds_path}/ ...")
            shutil.rmtree(ds_path)
            log.info(f"  Deleted.")

    # --- Wipe results folder ---
    if os.path.isdir(RESULTS_DIR):
        log.info(f"  Deleting {RESULTS_DIR}/ ...")
        shutil.rmtree(RESULTS_DIR)
        log.info(f"  Deleted.")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Download each dataset ---
    download_funcs = {
        'eurosat': download_eurosat,
        'ucmerced': download_ucmerced,
        'nwpu_resisc45': download_nwpu_resisc45,
        'aid': download_aid,
    }

    for ds_name in dataset_names:
        log.info(f"\n  Downloading {ds_name}...")
        func = download_funcs.get(ds_name)
        if func:
            try:
                func()
                log.info(f"  [OK] {ds_name} downloaded.")
            except Exception as e:
                log.info(f"  [FAIL] {ds_name}: {e}")
                log.info(traceback.format_exc())
        else:
            log.info(f"  [SKIP] {ds_name}: no download function available")

    log.info("")


# ============================================================
# Step 1: Verify Datasets
# ============================================================

def step_verify_datasets(dataset_names, log):
    """Check that requested datasets exist on disk."""
    from modules.dataset_loader import (
        verify_dataset, download_eurosat, download_ucmerced
    )

    step_header(1, "VERIFY DATASETS", log)

    available = []
    for ds_name in dataset_names:
        ok, n_cls, n_img = verify_dataset(ds_name, verbose=False)
        if ok:
            available.append(ds_name)
            log.info(f"  [OK] {ds_name}: {n_img} images, {n_cls} classes")
        else:
            log.info(f"  [--] {ds_name} not found, attempting download...")
            try:
                if ds_name == 'eurosat':
                    download_eurosat()
                elif ds_name == 'ucmerced':
                    download_ucmerced()
                else:
                    log.info(f"  [SKIP] {ds_name}: no auto-download available")
                    continue
                ok, n_cls, n_img = verify_dataset(ds_name, verbose=False)
                if ok:
                    available.append(ds_name)
                    log.info(f"  [OK] {ds_name}: {n_img} images, {n_cls} classes")
                else:
                    log.info(f"  [FAIL] {ds_name}: download completed but "
                             f"verification failed")
            except Exception as e:
                log.info(f"  [FAIL] {ds_name}: {e}")

    if not available:
        log.info("\n  No datasets available! Cannot proceed.")
        sys.exit(1)

    return available


# ============================================================
# Step 2: Train All Models
# ============================================================

def model_already_trained(dataset_name, model_name):
    """Check if a model has already been trained and evaluated."""
    model_dir = os.path.join(RESULTS_DIR, 'models', dataset_name, model_name)
    required = ['best_model.pth', 'evaluation_metrics.json', 'test_results.npz']
    return all(os.path.exists(os.path.join(model_dir, f)) for f in required)


def rebuild_summary_from_disk():
    """Scan results/models/ and rebuild all_experiments_summary.json."""
    summary_path = os.path.join(RESULTS_DIR, 'all_experiments_summary.json')
    results = {}
    models_base = os.path.join(RESULTS_DIR, 'models')
    if not os.path.isdir(models_base):
        return
    for ds_dir in sorted(os.listdir(models_base)):
        ds_path = os.path.join(models_base, ds_dir)
        if not os.path.isdir(ds_path):
            continue
        for m_dir in sorted(os.listdir(ds_path)):
            eval_path = os.path.join(ds_path, m_dir,
                                     'evaluation_metrics.json')
            ts_path = os.path.join(ds_path, m_dir,
                                   'training_summary.json')
            if not os.path.exists(eval_path):
                continue
            with open(eval_path) as f:
                ev = json.load(f)
            train_time = 0
            params_m = MODELS.get(m_dir, {}).get('params_m', 0)
            best_epoch = 0
            if os.path.exists(ts_path):
                with open(ts_path) as f:
                    ts = json.load(f)
                train_time = ts.get('training_time', 0)
                best_epoch = ts.get('best_epoch', 0)
                if ts.get('params_m'):
                    params_m = ts['params_m']
            # Use actual params from evaluation if available
            if ev.get('params_m'):
                params_m = ev['params_m']
            results.setdefault(ds_dir, {})[m_dir] = {
                'accuracy': ev['accuracy'],
                'f1_macro': ev['f1_macro'],
                'f1_weighted': ev['f1_weighted'],
                'kappa': ev.get('kappa', 0),
                'training_time': train_time,
                'params_m': params_m,
                'best_epoch': best_epoch,
            }
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    return results


def step_train_all(available_datasets, model_names, epochs=None,
                   force=False, log=None):
    """Train all requested models on all available datasets."""
    import torch
    from modules.dataset_loader import create_dataloaders
    from modules.models import create_model, count_parameters
    from modules.trainer import train_model
    from modules.evaluator import evaluate_model, save_evaluation_results

    step_header(2, "TRAIN & EVALUATE ALL MODELS", log)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"  Device: {device}")
    if device.type == 'cuda':
        log.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        log.info(f"  Memory: {mem_gb:.1f} GB")

    total_experiments = len(model_names) * len(available_datasets)
    log.info(f"  Models: {len(model_names)} ({', '.join(model_names)})")
    log.info(f"  Datasets: {len(available_datasets)} "
             f"({', '.join(available_datasets)})")
    log.info(f"  Total experiments: {total_experiments}")
    if not force:
        log.info("  Mode: skip already-trained (use --force to retrain)")

    all_results = {}
    experiment_idx = 0
    skipped = 0
    failed = []
    global_start = time.time()

    for ds_name in available_datasets:
        log.info(f"\n{'─' * 60}")
        log.info(f"  DATASET: {ds_name.upper()}")
        log.info(f"{'─' * 60}")

        ds_config = DATASETS[ds_name]
        train_loader, test_loader, class_names = create_dataloaders(ds_name)
        all_results[ds_name] = {}

        for model_name in model_names:
            experiment_idx += 1
            exp_start = time.time()

            # Skip if already trained (unless --force)
            if not force and model_already_trained(ds_name, model_name):
                log.info(f"\n  [{experiment_idx}/{total_experiments}] "
                         f"{model_name} on {ds_name} -- SKIP (already trained)")
                skipped += 1

                # Load existing results for the summary
                model_dir = os.path.join(
                    RESULTS_DIR, 'models', ds_name, model_name)
                try:
                    with open(os.path.join(
                            model_dir, 'evaluation_metrics.json')) as f:
                        saved = json.load(f)

                    # Also load training summary for time/epoch
                    train_time = 0
                    best_epoch = 0
                    params_m = MODELS.get(model_name, {}).get('params_m', 0)
                    summary_path = os.path.join(
                        model_dir, 'training_summary.json')
                    if os.path.exists(summary_path):
                        with open(summary_path) as f:
                            ts = json.load(f)
                        train_time = ts.get('training_time', 0)
                        best_epoch = ts.get('best_epoch', 0)

                    all_results[ds_name][model_name] = {
                        'accuracy': saved['accuracy'],
                        'f1_macro': saved['f1_macro'],
                        'f1_weighted': saved['f1_weighted'],
                        'kappa': saved['kappa'],
                        'params_m': params_m,
                        'training_time': train_time,
                        'best_epoch': best_epoch,
                        'skipped': True,
                    }
                    log.info(f"    Loaded: {saved['accuracy']:.4f} acc, "
                             f"{saved['f1_macro']:.4f} F1")
                except Exception:
                    pass
                continue

            log.info(f"\n  [{experiment_idx}/{total_experiments}] "
                     f"{model_name} on {ds_name}")

            try:
                model = create_model(model_name, ds_config['num_classes'])
                total_params, trainable_params = count_parameters(model)
                log.info(f"    Parameters: {total_params / 1e6:.1f}M "
                         f"(trainable: {trainable_params / 1e6:.1f}M)")

                # Train
                train_result = train_model(
                    model, train_loader, test_loader,
                    num_classes=ds_config['num_classes'],
                    model_name=model_name,
                    dataset_name=ds_name,
                    epochs=epochs,
                    device=device,
                )

                # Evaluate
                eval_result = evaluate_model(
                    model, test_loader, class_names, device)
                save_evaluation_results(eval_result, model_name, ds_name)

                exp_time = time.time() - exp_start
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

                acc = eval_result['accuracy']
                f1 = eval_result['f1_macro']
                log.info(f"    Done: {acc:.4f} acc, {f1:.4f} F1, "
                         f"{exp_time:.0f}s total")

                # Free GPU memory
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            except Exception as e:
                log.info(f"    ERROR: {e}")
                log.info(f"    Traceback:\n{traceback.format_exc()}")
                all_results[ds_name][model_name] = {'error': str(e)}
                failed.append(f"{model_name}/{ds_name}")

                # Still try to free GPU
                try:
                    del model
                except NameError:
                    pass
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

    total_time = time.time() - global_start

    # Save combined summary (merge with existing results from other datasets)
    summary_path = os.path.join(RESULTS_DIR, 'all_experiments_summary.json')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load existing summary to preserve results from datasets not in this run
    existing_results = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                existing_results = json.load(f).get('results', {})
        except (json.JSONDecodeError, KeyError):
            pass

    # Also scan disk for any datasets with saved evaluation files
    models_base = os.path.join(RESULTS_DIR, 'models')
    if os.path.isdir(models_base):
        for ds_dir in os.listdir(models_base):
            ds_path = os.path.join(models_base, ds_dir)
            if not os.path.isdir(ds_path) or ds_dir in all_results:
                continue
            for m_dir in os.listdir(ds_path):
                eval_path = os.path.join(ds_path, m_dir,
                                         'evaluation_metrics.json')
                ts_path = os.path.join(ds_path, m_dir,
                                       'training_summary.json')
                if os.path.exists(eval_path):
                    with open(eval_path) as f:
                        ev = json.load(f)
                    train_time = 0
                    params_m = 0
                    best_epoch = 0
                    if os.path.exists(ts_path):
                        with open(ts_path) as f:
                            ts = json.load(f)
                        train_time = ts.get('training_time', 0)
                        best_epoch = ts.get('best_epoch', 0)
                        params_m = ts.get('params_m', 0)
                    existing_results.setdefault(ds_dir, {})[m_dir] = {
                        'accuracy': ev['accuracy'],
                        'f1_macro': ev['f1_macro'],
                        'f1_weighted': ev['f1_weighted'],
                        'kappa': ev.get('kappa', 0),
                        'training_time': train_time,
                        'params_m': params_m,
                        'best_epoch': best_epoch,
                    }

    save_data = {
        'timestamp': datetime.now().isoformat(),
        'results': existing_results,
        'total_time_seconds': total_time,
        'device': str(device),
    }
    # Overwrite with current run results (authoritative)
    for ds_name, models_dict in all_results.items():
        save_data['results'][ds_name] = {}
        for m_name, m_res in models_dict.items():
            save_data['results'][ds_name][m_name] = {
                k: v for k, v in m_res.items()
                if k not in ('per_class_f1', 'skipped')
            }

    with open(summary_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    # Print training summary
    log.info(f"\n{'─' * 60}")
    log.info(f"  TRAINING COMPLETE  ({total_time / 60:.1f} minutes)")
    log.info(f"  Trained: {total_experiments - skipped - len(failed)}, "
             f"Skipped: {skipped}, Failed: {len(failed)}")
    log.info(f"{'─' * 60}")

    for ds_name in available_datasets:
        log.info(f"\n  {ds_name}:")
        for m_name in model_names:
            if m_name in all_results.get(ds_name, {}):
                res = all_results[ds_name][m_name]
                if 'error' in res:
                    log.info(f"    {m_name:20s}  FAILED")
                elif res.get('skipped'):
                    log.info(f"    {m_name:20s}  "
                             f"{res['accuracy']:.4f} acc  "
                             f"{res['f1_macro']:.4f} F1  (cached)")
                else:
                    log.info(f"    {m_name:20s}  "
                             f"{res['accuracy']:.4f} acc  "
                             f"{res['f1_macro']:.4f} F1  "
                             f"{res['training_time']:.0f}s")

    if failed:
        log.info(f"\n  FAILED experiments: {', '.join(failed)}")

    log.info(f"\n  Summary saved: {summary_path}")
    return all_results


# ============================================================
# Step 3: Generate Publication Tables
# ============================================================

def step_generate_tables(log):
    """Generate Excel tables from results."""
    step_header(3, "GENERATE PUBLICATION TABLES", log)

    script = os.path.join(SCRIPT_DIR, 'generate_publication_outputs.py')
    if not os.path.exists(script):
        log.info(f"  Script not found: {script}")
        log.info("  Skipping table generation.")
        return

    result = subprocess.run(
        [sys.executable, script],
        cwd=PROJECT_ROOT,
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode == 0:
        # Show key output lines
        for line in result.stdout.splitlines():
            if line.strip():
                log.info(f"  {line.strip()}")
        log.info("  Tables generated successfully.")
    else:
        log.info(f"  Table generation failed (exit code {result.returncode})")
        if result.stderr:
            for line in result.stderr.splitlines()[-10:]:
                log.info(f"    {line}")


# ============================================================
# Step 3b: Generate LaTeX Tables + Manuscript Values
# ============================================================

def step_generate_latex_tables(log):
    """Generate LaTeX table snippets and manuscript value reference."""
    log.info("\n  --- Generating LaTeX tables + manuscript values ---")

    script = os.path.join(SCRIPT_DIR, 'generate_latex_tables.py')
    if not os.path.exists(script):
        log.info(f"  Script not found: {script}")
        return

    result = subprocess.run(
        [sys.executable, script],
        cwd=PROJECT_ROOT,
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if line.strip():
                log.info(f"  {line.strip()}")
    else:
        log.info(f"  LaTeX table generation failed")
        if result.stderr:
            for line in result.stderr.splitlines()[-5:]:
                log.info(f"    {line}")


# ============================================================
# Step 4: Generate Publication Figures
# ============================================================

def step_generate_figures(log):
    """Generate all 16 PDF figures for the manuscript."""
    step_header(4, "GENERATE PUBLICATION FIGURES", log)

    script = os.path.join(
        PROJECT_ROOT, 'publication', 'manuscript', 'generate_all_figures.py')
    if not os.path.exists(script):
        log.info(f"  Script not found: {script}")
        log.info("  Skipping figure generation.")
        return

    result = subprocess.run(
        [sys.executable, script],
        cwd=os.path.dirname(script),
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if any(k in line for k in ['COMPLETE', 'Saved', 'Total',
                                        'Generated', 'saved']):
                log.info(f"  {line.strip()}")
        log.info("  Figures generated successfully.")
    else:
        log.info(f"  Figure generation failed (exit code {result.returncode})")
        if result.stderr:
            for line in result.stderr.splitlines()[-15:]:
                log.info(f"    {line}")


# ============================================================
# Step 5: Compile LaTeX (optional)
# ============================================================

def step_compile_latex(log):
    """Compile the manuscript PDF."""
    step_header(5, "COMPILE LATEX MANUSCRIPT", log)

    manuscript_dir = os.path.join(
        PROJECT_ROOT, 'publication', 'manuscript')

    # Try humanized version first, then original
    for tex_name in ['manuscript_humanized', 'manuscript']:
        tex_file = os.path.join(manuscript_dir, f'{tex_name}.tex')
        if os.path.exists(tex_file):
            break
    else:
        log.info("  No .tex file found. Skipping.")
        return

    log.info(f"  Compiling: {tex_name}.tex")

    try:
        for pass_num in range(3):
            cmd = ['pdflatex', '-interaction=nonstopmode', tex_file]
            subprocess.run(cmd, cwd=manuscript_dir,
                           capture_output=True, timeout=120)
            if pass_num == 0:
                subprocess.run(['bibtex', tex_name], cwd=manuscript_dir,
                               capture_output=True, timeout=60)
        log.info(f"  Compiled: {tex_name}.pdf")
    except FileNotFoundError:
        log.info("  pdflatex not found. Install MiKTeX or TeX Live.")
    except subprocess.TimeoutExpired:
        log.info("  LaTeX compilation timed out.")
    except Exception as e:
        log.info(f"  LaTeX compilation failed: {e}")


# ============================================================
# Step 6: Final Summary
# ============================================================

def step_final_summary(available_datasets, log):
    """Print where everything is saved."""
    step_header(6, "PIPELINE COMPLETE", log)

    fig_dir = os.path.join(
        PROJECT_ROOT, 'publication', 'manuscript', 'figures')
    table_dir = os.path.join(RESULTS_DIR, 'tables')
    models_dir = os.path.join(RESULTS_DIR, 'models')

    # Count outputs
    n_figures = 0
    if os.path.isdir(fig_dir):
        n_figures = len([f for f in os.listdir(fig_dir)
                         if f.endswith('.pdf')])

    n_tables = 0
    if os.path.isdir(table_dir):
        for root, dirs, files in os.walk(table_dir):
            n_tables += len([f for f in files if f.endswith('.xlsx')])

    n_models = 0
    if os.path.isdir(models_dir):
        for ds in os.listdir(models_dir):
            ds_path = os.path.join(models_dir, ds)
            if os.path.isdir(ds_path):
                for m in os.listdir(ds_path):
                    if os.path.exists(
                            os.path.join(ds_path, m, 'best_model.pth')):
                        n_models += 1

    log.info(f"""
  Trained models ({n_models}):
    results/models/{{dataset}}/{{model}}/
      best_model.pth          - Trained weights
      training_history.npz    - Loss/accuracy curves
      training_summary.json   - Training metadata
      evaluation_metrics.json - All metrics
      test_results.npz        - Predictions (y_true, y_pred, y_probs)

  Publication figures ({n_figures} PDFs):
    publication/manuscript/figures/

  Publication tables ({n_tables} Excel):
    results/tables/performance/
    results/tables/statistical/

  LaTeX tables (ready to paste into manuscript):
    results/tables/latex/table_eurosat.tex
    results/tables/latex/table_ucmerced.tex
    results/tables/latex/table_efficiency.tex
    results/tables/latex/manuscript_values.tex   <- \\newcommand defs
    results/tables/latex/quick_reference.txt     <- all numbers at a glance

  Summary:
    results/all_experiments_summary.json

  To pick results for the paper, load:
    evaluation_metrics.json  -> accuracy, F1, kappa, per-class
    test_results.npz         -> y_true, y_pred for custom analysis
    training_history.npz     -> plot training curves
""")


# ============================================================
# Check mode (dry run)
# ============================================================

def run_check(dataset_names, model_names, log):
    """Dry run: verify everything is ready without training."""
    log.info("\n  MODE: CHECK ONLY (no training)")

    # Preflight
    ok = step_preflight(log)
    if not ok:
        return

    # Datasets
    available = step_verify_datasets(dataset_names, log)

    # Check existing results
    step_header("X", "EXISTING RESULTS", log)
    for ds_name in available:
        log.info(f"\n  {ds_name}:")
        for model_name in model_names:
            trained = model_already_trained(ds_name, model_name)
            status = "TRAINED" if trained else "NOT TRAINED"
            detail = ""
            if trained:
                metrics_path = os.path.join(
                    RESULTS_DIR, 'models', ds_name, model_name,
                    'evaluation_metrics.json')
                try:
                    with open(metrics_path) as f:
                        m = json.load(f)
                    detail = (f"  acc={m['accuracy']:.4f}  "
                              f"F1={m['f1_macro']:.4f}")
                except Exception:
                    detail = "  (metrics unreadable)"
            log.info(f"    {model_name:20s}  [{status}]{detail}")

    # Check scripts
    step_header("X", "PIPELINE SCRIPTS", log)
    scripts = [
        ('generate_publication_outputs.py', SCRIPT_DIR),
        ('generate_all_figures.py',
         os.path.join(PROJECT_ROOT, 'publication', 'manuscript')),
    ]
    for name, directory in scripts:
        path = os.path.join(directory, name)
        status = "OK" if os.path.exists(path) else "MISSING"
        log.info(f"  [{status}] {name}")

    # LaTeX
    for tex in ['manuscript_humanized.tex', 'manuscript.tex']:
        path = os.path.join(PROJECT_ROOT, 'publication', 'manuscript', tex)
        if os.path.exists(path):
            log.info(f"  [OK] {tex}")

    log.info("\n  Check complete. Use 'python scripts/run_all.py' to run.")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run entire experiment pipeline: train + evaluate + outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python scripts/run_all.py                    # Full pipeline (skip trained)
  python scripts/run_all.py --check            # Dry run, verify everything
  python scripts/run_all.py --fresh            # Wipe everything, redownload, retrain
  python scripts/run_all.py --force            # Retrain all (keep existing data)
  python scripts/run_all.py --skip-training    # Regenerate tables & figures only
  python scripts/run_all.py --dataset eurosat  # One dataset
  python scripts/run_all.py --models resnet50 vit_b_16  # Specific models
  python scripts/run_all.py --log              # Save output to log file
        """)
    parser.add_argument('--dataset', type=str, nargs='+', default=None,
                        help='Datasets to use (default: eurosat ucmerced)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Models to train (default: all 8)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override max epochs')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training, only regenerate tables and figures')
    parser.add_argument('--skip-figures', action='store_true',
                        help='Skip figure generation')
    parser.add_argument('--skip-latex', action='store_true',
                        help='Skip LaTeX compilation')
    parser.add_argument('--force', action='store_true',
                        help='Retrain models even if results already exist')
    parser.add_argument('--fresh', action='store_true',
                        help='Wipe data + results, redownload datasets, '
                             'retrain everything from scratch')
    parser.add_argument('--check', action='store_true',
                        help='Dry run: verify setup without training')
    parser.add_argument('--log', action='store_true',
                        help='Save output to results/pipeline_TIMESTAMP.log')
    args = parser.parse_args()

    dataset_names = args.dataset or ['eurosat', 'ucmerced']
    model_names = args.models or list(MODELS.keys())

    # Validate model names early
    invalid = [m for m in model_names if m not in MODELS]
    if invalid:
        print(f"ERROR: Unknown model(s): {', '.join(invalid)}")
        print(f"Available: {', '.join(MODELS.keys())}")
        sys.exit(1)

    # Validate dataset names early
    invalid_ds = [d for d in dataset_names if d not in DATASETS]
    if invalid_ds:
        print(f"ERROR: Unknown dataset(s): {', '.join(invalid_ds)}")
        print(f"Available: {', '.join(DATASETS.keys())}")
        sys.exit(1)

    log = setup_logging(use_log_file=args.log)
    start_time = time.time()

    log.info("=" * 60)
    log.info("  SCENE CLASSIFICATION - FULL PIPELINE")
    log.info("=" * 60)
    log.info(f"  Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Datasets: {', '.join(dataset_names)}")
    log.info(f"  Models:   {len(model_names)} architectures")
    log.info(f"  Epochs:   {args.epochs or TRAINING['epochs']}")

    # Check mode
    if args.check:
        run_check(dataset_names, model_names, log)
        return

    # --fresh implies --force (results wiped, must retrain)
    if args.fresh:
        args.force = True
        log.info("  Mode:     FRESH (wipe data + results, redownload, retrain)")
    elif args.skip_training:
        log.info("  Mode:     OUTPUTS ONLY (skip training)")

    # Step 0: Preflight
    ok = step_preflight(log)
    if not ok:
        log.info("\n  Preflight failed. Fix missing packages and retry.")
        sys.exit(1)

    # Step 0b: Fresh download (wipe + redownload)
    if args.fresh:
        step_fresh_download(dataset_names, log)

    # Step 1: Verify datasets
    available = step_verify_datasets(dataset_names, log)

    # Step 2: Train (unless skipped)
    if not args.skip_training:
        step_train_all(available, model_names, args.epochs,
                       force=args.force, log=log)
    else:
        log.info("\n  Skipping training (--skip-training)")

    # Rebuild summary from all results on disk
    rebuild_summary_from_disk()

    # Step 3: Generate tables (Excel + LaTeX)
    step_generate_tables(log)
    step_generate_latex_tables(log)

    # Step 4: Generate figures
    if not args.skip_figures:
        step_generate_figures(log)
    else:
        log.info("\n  Skipping figures (--skip-figures)")

    # Step 5: Compile LaTeX
    if not args.skip_latex:
        step_compile_latex(log)
    else:
        log.info("\n  Skipping LaTeX (--skip-latex)")

    # Step 6: Summary
    step_final_summary(available, log)

    total = time.time() - start_time
    log.info(f"  Total pipeline time: {total / 60:.1f} minutes")
    log.info(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)


if __name__ == '__main__':
    main()
