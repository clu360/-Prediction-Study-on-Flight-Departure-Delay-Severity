"""Build and optionally execute Phase IV analysis using only published aggregates."""

import argparse
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "04_weather_mlp_analysis.ipynb"


def build():
    markdown = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    return nbf.v4.new_notebook(
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        cells=[
            markdown("""# Phase IV — Weather and MLP comparisons

The primary Phase III results remain the pre-existing study reference. The Phase IV fit window is August–October 2025. Each arm scores November and December with one frozen model and transformer. G4 and weighting comparisons are exploratory because those holdout months were previously examined. The MLP architecture is selected only on April–October development folds.

This notebook reads published aggregates. When a Phase IV aggregate is unavailable, its section reports that evidence is pending."""),
            code("""from pathlib import Path
import csv
import json
from IPython.display import Markdown, display
import matplotlib.pyplot as plt

here = Path.cwd()
repo = next(p for p in (here, *here.parents) if (p / 'reports/phase3').is_dir())
phase3 = repo / 'reports/phase3'
phase4 = repo / 'reports/phase4'

def csv_rows(path):
    if path.exists():
        with path.open(encoding='utf-8', newline='') as file:
            return list(csv.DictReader(file))
    return []

def table(rows, columns):
    if rows:
        lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
        lines.extend('| ' + ' | '.join(str(r.get(c, '')).replace('|', '/') for c in columns) + ' |' for r in rows)
        display(Markdown('\\n'.join(lines)))
    else:
        print('Verified aggregate evidence is pending.')

prior = csv_rows(phase3 / 'metrics_summary.csv')
historical = [r for r in prior if r['model'] == 'xgboost'
              and r['feature_view'] in {'G1_G2', 'G1_G2_G3'}
              and r['airport'] == 'overall' and 'pooled' in r['scored_month'].lower()]
table(historical, ['feature_view', 'scored_month', 'n', 'macro_f1', 'log_loss', 'severe_precision', 'severe_recall'])"""),
            markdown("""## Architecture and paired models

Six unweighted candidates screen on April, July, and October. The top two are compared across all seven development folds by equal-month log loss. Inner validation selects epochs within each training window. The median stopping epoch of the selected candidate fixes the final training length.

XGBoost uses the Phase III G1–G3 settings for both fresh fits. MLP-A and MLP-B use G1–G3. MLP-C and MLP-D add G4. MLP-B and MLP-D use inverse-frequency class weighting."""),
            code("""tuning = csv_rows(phase4 / 'mlp_tuning_summary.csv')
if not tuning:
    tuning = csv_rows(phase4 / 'mlp_development_progress.csv')
    if tuning:
        print('Completed development scores. The architecture lock is pending.')
table(tuning, ['candidate_id', 'month', 'epochs', 'log_loss', 'macro_f1', 'stop_reason'])
lock_path = phase4 / 'mlp_selection.json'
if lock_path.exists():
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    fit_config = dict(candidate_id=lock['candidate_id'], **lock['candidate'], epochs=lock['epochs'],
                      final_seeds=lock['settings']['final_seeds'])
    table([fit_config], list(fit_config))
    print(lock['epoch_rule'])
    curves = csv_rows(phase4 / 'mlp_learning_curves.csv')
    curves = [r for r in curves if int(r['candidate_id']) == lock['candidate_id']
              and r['stage'] == 'inner']
    if curves:
        fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
        for month in sorted({r['month'] for r in curves}, key=int):
            line = [r for r in curves if r['month'] == month]
            ax.plot([int(r['epoch']) for r in line], [float(r['stop_loss']) for r in line],
                    marker='.', label=f'Scored month {int(month):02}')
        ax.set(xlabel='Inner training epoch', ylabel='Chronological stop-validation log loss')
        ax.legend(ncols=2, fontsize='small')
        plt.show()
seed_summary = csv_rows(phase4 / 'seed_summary.csv')
table([r for r in seed_summary if r['subgroup'] == 'overall'],
      ['arm', 'month', 'seed_count', 'macro_f1', 'log_loss', 'severe_precision', 'severe_recall',
       'severe_pr_auc_ap', 'actual_severe_prevalence', 'mean_predicted_severe_probability', 'argmax_severe_share'])
comparisons = csv_rows(phase4 / 'model_contrasts.csv')
if comparisons:
    table(comparisons, ['month', 'arm', 'reference', 'metric', 'raw_difference'])"""),
            markdown("""## Severe risk and paired contrasts

Calibration compares mean severe probability with observed severe frequency within each bin. The argmax severe share is the proportion classified as severe. The precision–recall curves are saved numerical aggregates. Factorial contrasts are raw model-minus-comparator differences, including for log loss."""),
            code("""details_path = phase4 / 'metric_details.json'
details = json.loads(details_path.read_text(encoding='utf-8')) if details_path.exists() else {}
pooled = {key: values for key, values in details.items() if key.endswith('/pooled/overall')}
if pooled:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for key, values in sorted(pooled.items()):
        label = '/'.join(key.split('/')[:2])
        bins = [r for r in values['calibration'] if r['class_id'] == 2 and r['n']]
        axes[0].plot([r['mean_probability'] for r in bins], [r['observed_rate'] for r in bins], label=label)
        curve = values.get('severe_pr_curve', [])
        if curve:
            recall, precision = zip(*curve)
            axes[1].plot(recall, precision, label=label)
    axes[0].plot([0, 1], [0, 1], color='gray', linestyle=':')
    axes[0].set(xlabel='Mean severe probability', ylabel='Observed severe fraction')
    axes[1].set(xlabel='Severe recall', ylabel='Severe precision')
    axes[1].legend(fontsize='small')
    plt.show()
else:
    print('Verified severe-risk traces are pending.')

contrasts = csv_rows(phase4 / 'factorial_contrasts.csv')
table([r for r in contrasts if r['subgroup'] == 'overall' and r['month'] == 'pooled'
       and r['metric'] in {'macro_f1', 'log_loss', 'severe_recall', 'severe_precision'}],
      ['seed', 'metric', 'B_minus_A', 'C_minus_A', 'D_minus_C', 'D_minus_B', 'interaction'])"""),
            markdown("""## Airports, lead time, weather and research questions

Lead-time bands use the interval from prediction to scheduled departure. The adverse band includes any MVFR-or-worse origin or destination, frozen precipitation, or thunderstorm. Benign requires those four signals to be observed and negative. The remaining cases are unknown.

These comparisons describe predictive performance. The evaluated period is within 2025 and correlated flight counts do not justify IID confidence intervals."""),
            code("""table([r for r in seed_summary if r['subgroup'] != 'overall' and r['month'] == 'pooled'],
      ['arm', 'subgroup', 'seed_count', 'macro_f1', 'log_loss', 'severe_precision', 'severe_recall'])
coverage = csv_rows(phase4 / 'weather_coverage.csv')
if coverage:
    columns = [key for key in coverage[0] if key.startswith('g4_') and key.endswith('_nulls')]
    coverage_summary = []
    for key in columns:
        scope = 'dest_flights' if key.startswith('g4_dest') else 'origin_flights'
        available = sum(int(row[scope]) for row in coverage)
        missing = sum(int(row[key]) for row in coverage)
        coverage_summary.append(dict(feature=key.removesuffix('_nulls'), targets=available,
                                     missing=missing, fraction=round(missing / available, 6)))
    table(coverage_summary, ['feature', 'targets', 'missing', 'fraction'])
for name in ('research_question_summary.md', 'PHASE4_VALIDATION.md'):
    path = phase4 / name
    if path.exists():
        display(Markdown(path.read_text(encoding='utf-8')))"""),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", type=Path, help="Write the executed notebook to this path")
    parser.add_argument("--aggregate-root", type=Path, default=ROOT)
    args = parser.parse_args()
    notebook = build()
    if args.execute:
        notebook = nbf.read(NOTEBOOK, as_version=4)
        notebook = NotebookClient(notebook, timeout=120, kernel_name="python3",
                                  resources={"metadata": {"path": str(args.aggregate_root.resolve())}}).execute()
        args.execute.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(notebook, args.execute)
    else:
        nbf.write(notebook, NOTEBOOK)


if __name__ == "__main__":
    main()
