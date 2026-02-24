"""
Generate LaTeX Tables and Manuscript Values from Training Results

Reads all_experiments_summary.json + per-model evaluation files
and generates:
  1. LaTeX table snippets (Tables 2, 3, 4 for the manuscript)
  2. LaTeX commands file (\newcommand for every number in the paper)
  3. Quick-reference text summary

After retraining, run this script and either:
  - Copy-paste the generated tables into manuscript.tex, OR
  - Use \\input{} to include them automatically

Usage:
    python scripts/generate_latex_tables.py
    python scripts/generate_latex_tables.py --update-manuscript
"""

import os
import sys
import json
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from config import DATASETS, MODELS, RESULTS_DIR

# Display names for the manuscript
MODEL_DISPLAY = {
    'resnet50': 'ResNet-50',
    'resnet101': 'ResNet-101',
    'densenet121': 'DenseNet-121',
    'efficientnet_b0': 'EffNet-B0',
    'efficientnet_b3': 'EffNet-B3',
    'vit_b_16': 'ViT-B/16',
    'swin_t': 'Swin-T',
    'convnext_tiny': 'ConvNeXt-T',
}

# Param counts from Table 1 (backbone only, before head replacement)
# These are stable across runs, but we also read actual values from results
MODEL_PARAMS_DISPLAY = {
    'resnet50': '23.5M',
    'resnet101': '42.5M',
    'densenet121': '7.0M',
    'efficientnet_b0': '4.0M',
    'efficientnet_b3': '10.7M',
    'vit_b_16': '85.8M',
    'swin_t': '27.5M',
    'convnext_tiny': '27.8M',
}


def load_summary():
    """Load results, preferring per-model evaluation files over the summary.

    Per-model evaluation_metrics.json files are the ground truth (written
    directly after evaluation). The all_experiments_summary.json is a
    convenience aggregation that may be stale if models were retrained
    individually. We merge both sources, with per-model files taking priority.
    """
    # Start with summary as base
    summary_path = os.path.join(RESULTS_DIR, 'all_experiments_summary.json')
    data = {'results': {}}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            data = json.load(f)

    # Override metrics with per-model evaluation files (authoritative).
    # Keep training_time and params_m from summary (may be the only source
    # if training_summary.json doesn't exist yet).
    models_dir = os.path.join(RESULTS_DIR, 'models')
    if not os.path.isdir(models_dir):
        if not data['results']:
            print(f"ERROR: No results found. Run training first.")
            sys.exit(1)
        return data

    for ds_name in os.listdir(models_dir):
        ds_path = os.path.join(models_dir, ds_name)
        if not os.path.isdir(ds_path):
            continue
        if ds_name not in data['results']:
            data['results'][ds_name] = {}

        for model_name in os.listdir(ds_path):
            model_path = os.path.join(ds_path, model_name)
            eval_path = os.path.join(model_path, 'evaluation_metrics.json')

            if not os.path.exists(eval_path):
                continue

            with open(eval_path) as f:
                eval_data = json.load(f)

            # Keep existing summary data as base (has training_time, params)
            existing = data['results'].get(ds_name, {}).get(model_name, {})

            # Try training_summary.json for time/epoch (written by new trainer)
            summary_json = os.path.join(model_path, 'training_summary.json')
            if os.path.exists(summary_json):
                with open(summary_json) as f:
                    ts = json.load(f)
                train_time = ts.get('training_time', 0)
                best_epoch = ts.get('best_epoch', 0)
            else:
                # Fall back to summary JSON values
                train_time = existing.get('training_time', 0)
                best_epoch = existing.get('best_epoch', 0)

            # Override only metrics (accuracy, F1, kappa) from eval file
            entry = {
                'accuracy': eval_data['accuracy'],
                'f1_macro': eval_data['f1_macro'],
                'f1_weighted': eval_data['f1_weighted'],
                'kappa': eval_data['kappa'],
                'params_m': existing.get('params_m', 0),
                'training_time': train_time,
                'best_epoch': best_epoch,
            }
            data['results'][ds_name][model_name] = entry

    if not data['results']:
        print(f"ERROR: No results found. Run training first.")
        sys.exit(1)

    return data


def load_eval_metrics(dataset_name, model_name):
    """Load per-model evaluation metrics."""
    path = os.path.join(
        RESULTS_DIR, 'models', dataset_name, model_name,
        'evaluation_metrics.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def fmt_acc(val):
    """Format accuracy as percentage with 2 decimals."""
    return f"{val * 100:.2f}"


def fmt_f1(val):
    """Format F1/kappa with 4 decimals."""
    return f"{val:.4f}"


def fmt_time(val):
    """Format training time as integer seconds with LaTeX thousands separator."""
    t = int(round(val))
    if t >= 1000:
        return f"{t // 1000}{{,}}{t % 1000:03d}"
    return str(t)


def fmt_time_plain(val):
    """Format training time as integer seconds (no LaTeX)."""
    return str(int(round(val)))


def generate_performance_table(data, dataset_name, caption, label,
                                test_count):
    """Generate a LaTeX performance table (Tables 2/3 in manuscript)."""
    models = data['results'].get(dataset_name, {})
    if not models:
        return f"% No results for {dataset_name}\n"

    # Sort by accuracy descending
    sorted_models = sorted(
        models.items(),
        key=lambda x: x[1].get('accuracy', 0),
        reverse=True
    )

    # Find best values for bolding
    best_acc = max(m['accuracy'] for _, m in sorted_models)
    best_f1 = max(m['f1_macro'] for _, m in sorted_models)
    best_kappa = max(m['kappa'] for _, m in sorted_models)

    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption} ({test_count:,} test images). "
                 f"Best results in bold.}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{OA (\%)} & \textbf{F1-Mac} "
                 r"& \textbf{$\kappa$} & \textbf{Params} \\")
    lines.append(r"\midrule")

    for model_key, metrics in sorted_models:
        name = MODEL_DISPLAY.get(model_key, model_key)
        # Use actual params from results (includes dataset-specific head)
        # Fall back to static display value if not in results
        if 'params_m' in metrics and metrics['params_m'] > 0:
            params = f"{metrics['params_m']:.1f}M"
        else:
            params = MODEL_PARAMS_DISPLAY.get(model_key, '?M')

        acc = fmt_acc(metrics['accuracy'])
        f1 = fmt_f1(metrics['f1_macro'])
        kappa = fmt_f1(metrics['kappa'])

        # Bold the best
        if metrics['accuracy'] == best_acc:
            acc = f"\\textbf{{{acc}}}"
        if metrics['f1_macro'] == best_f1:
            f1 = f"\\textbf{{{f1}}}"
        if metrics['kappa'] == best_kappa:
            kappa = f"\\textbf{{{kappa}}}"

        lines.append(f"{name:15s} & {acc} & {f1} & {kappa} & {params} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_efficiency_table(data):
    """Generate Table 4: training time comparison."""
    eurosat = data['results'].get('eurosat', {})
    ucmerced = data['results'].get('ucmerced', {})

    if not eurosat:
        return "% No EuroSAT results for efficiency table\n"

    # Sort by EuroSAT training time ascending
    sorted_models = sorted(
        eurosat.items(),
        key=lambda x: x[1].get('training_time', 0)
    )

    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Training time comparison (seconds). "
                 r"Models sorted by EuroSAT time.}")
    lines.append(r"\label{tab:efficiency}")
    lines.append(r"\begin{tabular}{lrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{EuroSAT} "
                 r"& \textbf{UC Merced} \\")
    lines.append(r"\midrule")

    for model_key, metrics in sorted_models:
        name = MODEL_DISPLAY.get(model_key, model_key)
        t_euro = fmt_time(metrics['training_time'])
        t_ucm = ""
        if model_key in ucmerced:
            t_ucm = fmt_time(ucmerced[model_key]['training_time'])
        lines.append(f"{name:15s} & {t_euro} & {t_ucm} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_latex_commands(data):
    """Generate \\newcommand definitions for every manuscript number."""
    cmds = []
    cmds.append("% Auto-generated manuscript values")
    cmds.append("% Run: python scripts/generate_latex_tables.py")
    cmds.append(f"% Generated from: results/all_experiments_summary.json")
    cmds.append("")

    for ds_name in ['eurosat', 'ucmerced']:
        ds_data = data['results'].get(ds_name, {})
        if not ds_data:
            continue

        ds_tag = ds_name.capitalize()
        cmds.append(f"% --- {ds_tag} ---")

        # Find best model
        best_model = max(ds_data.items(),
                         key=lambda x: x[1].get('accuracy', 0))
        best_key, best_metrics = best_model

        cmds.append(f"\\newcommand{{\\best{ds_tag}Model}}"
                    f"{{{MODEL_DISPLAY[best_key]}}}")
        cmds.append(f"\\newcommand{{\\best{ds_tag}Acc}}"
                    f"{{{fmt_acc(best_metrics['accuracy'])}}}")
        cmds.append(f"\\newcommand{{\\best{ds_tag}Fone}}"
                    f"{{{fmt_f1(best_metrics['f1_macro'])}}}")
        cmds.append(f"\\newcommand{{\\best{ds_tag}Kappa}}"
                    f"{{{fmt_f1(best_metrics['kappa'])}}}")

        # Find worst model
        worst_model = min(ds_data.items(),
                          key=lambda x: x[1].get('accuracy', 0))
        worst_key, worst_metrics = worst_model
        cmds.append(f"\\newcommand{{\\worst{ds_tag}Model}}"
                    f"{{{MODEL_DISPLAY[worst_key]}}}")
        cmds.append(f"\\newcommand{{\\worst{ds_tag}Acc}}"
                    f"{{{fmt_acc(worst_metrics['accuracy'])}}}")

        # Range
        acc_range = (best_metrics['accuracy'] -
                     worst_metrics['accuracy']) * 100
        cmds.append(f"\\newcommand{{\\range{ds_tag}}}"
                    f"{{{acc_range:.2f}}}")

        # Per-model
        for model_key, metrics in ds_data.items():
            tag = model_key.replace('_', '')
            cmds.append(f"\\newcommand{{\\acc{tag}{ds_tag}}}"
                        f"{{{fmt_acc(metrics['accuracy'])}}}")
            cmds.append(f"\\newcommand{{\\fone{tag}{ds_tag}}}"
                        f"{{{fmt_f1(metrics['f1_macro'])}}}")

        # Training times
        fastest = min(ds_data.items(),
                      key=lambda x: x[1].get('training_time', float('inf')))
        slowest = max(ds_data.items(),
                      key=lambda x: x[1].get('training_time', 0))
        cmds.append(f"\\newcommand{{\\fastest{ds_tag}Model}}"
                    f"{{{MODEL_DISPLAY[fastest[0]]}}}")
        cmds.append(f"\\newcommand{{\\fastest{ds_tag}Time}}"
                    f"{{{fmt_time_plain(fastest[1]['training_time'])}}}")
        cmds.append(f"\\newcommand{{\\slowest{ds_tag}Model}}"
                    f"{{{MODEL_DISPLAY[slowest[0]]}}}")
        cmds.append(f"\\newcommand{{\\slowest{ds_tag}Time}}"
                    f"{{{fmt_time_plain(slowest[1]['training_time'])}}}")

        cmds.append("")

    return "\n".join(cmds)


def generate_quick_reference(data):
    """Generate a plain-text summary of all manuscript numbers."""
    lines = []
    lines.append("=" * 65)
    lines.append("  MANUSCRIPT QUICK REFERENCE - ALL NUMBERS")
    lines.append("  (Copy these into the manuscript after retraining)")
    lines.append("=" * 65)

    for ds_name in ['eurosat', 'ucmerced']:
        ds_data = data['results'].get(ds_name, {})
        if not ds_data:
            continue

        lines.append(f"\n{'─' * 65}")
        lines.append(f"  {ds_name.upper()}")
        lines.append(f"{'─' * 65}")

        # Sort by accuracy
        sorted_models = sorted(
            ds_data.items(),
            key=lambda x: x[1].get('accuracy', 0),
            reverse=True
        )

        lines.append(f"\n  {'Model':<20s} {'OA (%)':<10s} {'F1-Mac':<10s} "
                     f"{'Kappa':<10s} {'Time (s)':<10s}")
        lines.append(f"  {'─' * 60}")

        for model_key, m in sorted_models:
            name = MODEL_DISPLAY.get(model_key, model_key)
            lines.append(
                f"  {name:<20s} "
                f"{m['accuracy']*100:<10.2f} "
                f"{m['f1_macro']:<10.4f} "
                f"{m['kappa']:<10.4f} "
                f"{m['training_time']:<10.0f}"
            )

        best = sorted_models[0]
        worst = sorted_models[-1]
        acc_range = (best[1]['accuracy'] - worst[1]['accuracy']) * 100

        lines.append(f"\n  Best:  {MODEL_DISPLAY[best[0]]} "
                     f"({best[1]['accuracy']*100:.2f}%)")
        lines.append(f"  Worst: {MODEL_DISPLAY[worst[0]]} "
                     f"({worst[1]['accuracy']*100:.2f}%)")
        lines.append(f"  Range: {acc_range:.2f} percentage points")

    # Cross-dataset highlights for abstract/conclusion
    lines.append(f"\n{'─' * 65}")
    lines.append("  ABSTRACT / CONCLUSION KEY NUMBERS")
    lines.append(f"{'─' * 65}")

    for ds_name in ['eurosat', 'ucmerced']:
        ds_data = data['results'].get(ds_name, {})
        if not ds_data:
            continue
        best = max(ds_data.items(),
                   key=lambda x: x[1].get('accuracy', 0))
        lines.append(f"\n  {ds_name.upper()} best: "
                     f"{MODEL_DISPLAY[best[0]]} = "
                     f"{best[1]['accuracy']*100:.2f}%")

    # EfficientNet-B0 (lightweight highlight)
    for ds_name in ['eurosat', 'ucmerced']:
        ds_data = data['results'].get(ds_name, {})
        if 'efficientnet_b0' in ds_data:
            m = ds_data['efficientnet_b0']
            lines.append(f"  EffNet-B0 on {ds_name}: "
                         f"{m['accuracy']*100:.2f}% "
                         f"({m['params_m']:.1f}M params)")

    # Depth paradox: ResNet-101 vs ResNet-50
    lines.append(f"\n  DEPTH PARADOX (ResNet-101 vs ResNet-50):")
    for ds_name in ['eurosat', 'ucmerced']:
        ds_data = data['results'].get(ds_name, {})
        r50 = ds_data.get('resnet50', {})
        r101 = ds_data.get('resnet101', {})
        if r50 and r101:
            lines.append(
                f"    {ds_name}: R50={r50['accuracy']*100:.2f}% vs "
                f"R101={r101['accuracy']*100:.2f}%")

    # Efficiency comparison
    lines.append(f"\n  EFFICIENCY (for Discussion section):")
    euro = data['results'].get('eurosat', {})
    if 'efficientnet_b0' in euro and 'vit_b_16' in euro:
        b0 = euro['efficientnet_b0']
        vit = euro['vit_b_16']
        ratio = vit['params_m'] / b0['params_m'] if b0['params_m'] else 0
        gap = (vit['accuracy'] - b0['accuracy']) * 100
        lines.append(f"    ViT params / EffNet-B0 params = "
                     f"{ratio:.0f}x")
        lines.append(f"    Accuracy gap: {gap:.2f} pp")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Generate LaTeX tables from training results')
    parser.add_argument('--update-manuscript', action='store_true',
                        help='Auto-replace tables in manuscript.tex')
    args = parser.parse_args()

    data = load_summary()

    # Output directory
    out_dir = os.path.join(RESULTS_DIR, 'tables', 'latex')
    os.makedirs(out_dir, exist_ok=True)

    # Count test images from dataset config
    eurosat_test = int(27000 * 0.2)   # 5400
    ucmerced_test = int(2100 * 0.2)   # 420

    # 1. Performance tables
    table_eurosat = generate_performance_table(
        data, 'eurosat',
        'Classification performance on EuroSAT',
        'tab:eurosat', eurosat_test)

    table_ucmerced = generate_performance_table(
        data, 'ucmerced',
        'Classification performance on UC Merced',
        'tab:ucmerced', ucmerced_test)

    table_efficiency = generate_efficiency_table(data)

    # 2. LaTeX commands
    latex_commands = generate_latex_commands(data)

    # 3. Quick reference
    quick_ref = generate_quick_reference(data)

    # Save all files
    files = {
        'table_eurosat.tex': table_eurosat,
        'table_ucmerced.tex': table_ucmerced,
        'table_efficiency.tex': table_efficiency,
        'manuscript_values.tex': latex_commands,
        'quick_reference.txt': quick_ref,
    }

    for fname, content in files.items():
        path = os.path.join(out_dir, fname)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content + '\n')
        print(f"  Saved: {path}")

    # Print quick reference
    print(quick_ref)

    # Optionally update manuscript
    if args.update_manuscript:
        update_manuscript(table_eurosat, table_ucmerced, table_efficiency)


def update_manuscript(table_eurosat, table_ucmerced, table_efficiency):
    """Replace table environments in manuscript.tex with fresh data."""
    tex_path = os.path.join(
        PROJECT_ROOT, 'publication', 'manuscript', 'manuscript.tex')

    if not os.path.exists(tex_path):
        print(f"  Manuscript not found: {tex_path}")
        return

    with open(tex_path, 'r', encoding='utf-8') as f:
        content = f.read()

    replacements = {
        'tab:eurosat': table_eurosat,
        'tab:ucmerced': table_ucmerced,
        'tab:efficiency': table_efficiency,
    }

    for label, new_table in replacements.items():
        # Find the table environment containing this label
        marker = f"\\label{{{label}}}"
        if marker not in content:
            print(f"  WARNING: {marker} not found in manuscript")
            continue

        # Find \\begin{table} before and \\end{table} after
        label_pos = content.index(marker)

        # Search backward for \begin{table}
        begin_search = content.rfind(r'\begin{table}', 0, label_pos)
        if begin_search == -1:
            print(f"  WARNING: \\begin{{table}} not found before {label}")
            continue

        # Search forward for \end{table}
        end_search = content.find(r'\end{table}', label_pos)
        if end_search == -1:
            print(f"  WARNING: \\end{{table}} not found after {label}")
            continue
        end_search += len(r'\end{table}')

        old_table = content[begin_search:end_search]
        content = content[:begin_search] + new_table + content[end_search:]
        print(f"  Updated {label} in manuscript.tex")

    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"  Manuscript updated: {tex_path}")


if __name__ == '__main__':
    main()
