"""Build and optionally execute Phase IV analysis using only published aggregates."""

import argparse
import csv
import json
import re
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "04_weather_mlp_analysis.ipynb"


def csv_records(relative):
    path = ROOT / relative
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def saved_table(rows, columns):
    if not rows:
        return "Aggregate evidence is unavailable.\n"

    def value(row, key):
        item = row.get(key, "")
        if key not in {
            "month",
            "seed",
            "seeds",
            "n",
            "support",
            "candidate",
            "candidate_id",
            "epochs",
            "seed_count",
        }:
            try:
                return f"{float(item):.4f}"
            except (TypeError, ValueError):
                pass
        return str(item).replace("|", "/").replace("\n", " ")

    return (
        "\n".join(
            [
                "| " + " | ".join(columns) + " |",
                "| " + " | ".join(["---"] * len(columns)) + " |",
                *(
                    "| " + " | ".join(value(row, key) for key in columns) + " |"
                    for row in rows
                ),
            ]
        )
        + "\n"
    )


def class_records(details, source_input=False):
    rows = []
    for key, values in sorted(details.items()):
        parts = key.split("/")
        if source_input:
            view, regime, arm, seed, month = parts
        else:
            arm, seed, month, subgroup = parts
            if subgroup != "overall":
                continue
            view, regime = "Phase IV one-hot", "original"
        for class_id, name in enumerate(
            ["Minimal (<15)", "Moderate (15-<60)", "Severe (>=60)"]
        ):
            rows.append(
                dict(
                    view=view,
                    regime=regime,
                    arm=arm,
                    seed=seed,
                    month=month,
                    severity=name,
                    precision=values["precision"][class_id],
                    recall=values["recall"][class_id],
                    f1=values["f1"][class_id],
                    support=values["support"][class_id],
                )
            )
    return rows


def complete_saved_results():
    sections = [
        """## Coverage and saved results

The saved tables below are generated from the published aggregate reports and remain readable without running a kernel. All scores/rates are on a 0-1 scale (log loss is unbounded); support is an observation count. Seed summaries average metric values across three fitted models and are not ensemble predictions. The original weather/MLP study and the subsequent source-input representation study are distinct comparisons.

| Study | Completed evidence | Notebook |
|---|---|---|
| Phase III classical models | 367 development fits, 13 frozen final fits; CDF baseline amendment; 780 overall/airport metric rows | [03 classical analysis](03_classical_models_analysis.ipynb) |
| Original Phase IV | 26 candidate-fold evaluations, 2 XGBoost and 12 MLP final fits, 42 overall seed/month records and 546 subgroup records | This notebook |
| Source-input MLP follow-up | 52 candidate-fold evaluations, 24 final fits, 72 seed/month records and 24 seed-mean records; fixed and tuned regimes | This notebook |

All final models fit August-October and score both November and December without refitting. Weather, weighting and representation follow-ups use previously examined holdouts and remain exploratory. Development selects settings independently of holdout scores. No models are trained in this notebook.

### Original Phase IV seed means: every arm and period

"""
    ]
    summary = [
        row
        for row in csv_records("reports/phase4/seed_summary.csv")
        if row["subgroup"] == "overall"
    ]
    columns = [
        "arm",
        "month",
        "seed_count",
        "macro_f1",
        "log_loss",
        "severe_precision",
        "severe_recall",
        "severe_pr_auc_ap",
    ]
    sections.append(saved_table(summary, columns))
    sections.append(
        "\n### Original Phase IV: every individual fit and holdout period\n\n"
    )
    rows = [
        row
        for row in csv_records("reports/phase4/metrics_summary.csv")
        if row["subgroup"] == "overall"
    ]
    sections.append(
        saved_table(
            rows,
            [
                "arm",
                "seed",
                "month",
                "n",
                "accuracy",
                "macro_f1",
                "log_loss",
                "severe_precision",
                "severe_recall",
                "severe_pr_auc_ap",
            ],
        )
    )
    for title, name in [
        (
            "Original Phase IV development evaluations",
            "reports/phase4/mlp_tuning_summary.csv",
        ),
        ("Source-input development: G1-G3", "reports/mlp_source/tuning_G1_G2_G3.csv"),
        (
            "Source-input development: G1-G4",
            "reports/mlp_source/tuning_G1_G2_G3_G4.csv",
        ),
    ]:
        rows = csv_records(name)
        sections.append(f"\n### {title}\n\n")
        sections.append(
            saved_table(
                rows,
                [
                    "candidate_id"
                    if name.startswith("reports/phase4")
                    else "candidate",
                    "month",
                    "epochs",
                    "log_loss",
                    "macro_f1",
                    "seconds",
                ],
            )
        )
    sections.append("""\n## Source-input MLP follow-up

The follow-up retains the approved 20/25 causal source features, cohort, forecast instant and frozen fit window. Its adapter replaces one-hot encoding with eight-dimensional category embeddings, replaces train-fitted numeric medians/scaling with fixed unit conversions, and uses zero padding plus missingness indicators. Category vocabularies are still fitted on training rows only, with separate missing/unseen levels. IDs, timestamps and target outcomes remain excluded from predictors.

Fixed fits retain [256,128] widths, dropout 0.3, learning rate 0.0003 and one epoch. Tuned fits repeat development-only screening/finalist selection separately by view. Unweighted selection is shared by the standard and weighted loss arms for that view. Both regimes use seeds 6740/6741/6742. Seed variation is descriptive, not an IID confidence interval. This representation change combines categorical and numerical changes, so their individual effects cannot be isolated.

### Source-input selection locks

""")
    choices = []
    for view in ["G1_G2_G3", "G1_G2_G3_G4"]:
        path = ROOT / f"reports/mlp_source/selection_{view}.json"
        if path.exists():
            choice = json.loads(path.read_text(encoding="utf-8"))
            choices.append(
                dict(
                    view=view,
                    candidate=choice["candidate_id"],
                    **choice["candidate"],
                    epochs=choice["epochs"],
                    development_log_loss=choice["mean_log_loss"],
                )
            )
    sections.append(
        saved_table(
            choices,
            [
                "view",
                "candidate",
                "widths",
                "dropout",
                "learning_rate",
                "epochs",
                "development_log_loss",
            ],
        )
    )
    sections.append("\n### Source-input seed means: every regime, arm and period\n\n")
    sections.append(
        saved_table(
            csv_records("reports/mlp_source/metrics_summary.csv"),
            [
                "regime",
                "arm",
                "month",
                "seeds",
                "n",
                "accuracy",
                "macro_f1",
                "log_loss",
                "severe_precision",
                "severe_recall",
                "severe_pr_auc_ap",
                "delta_macro_f1",
                "macro_f1_seed_sd",
            ],
        )
    )
    sections.append(
        "\n### Source-input MLP: every individual fit and holdout period\n\n"
    )
    sections.append(
        saved_table(
            csv_records("reports/mlp_source/metrics_by_seed.csv"),
            [
                "regime",
                "arm",
                "seed",
                "month",
                "n",
                "accuracy",
                "macro_f1",
                "log_loss",
                "severe_precision",
                "severe_recall",
                "severe_pr_auc_ap",
                "delta_macro_f1",
            ],
        )
    )
    sections.append(
        "\n## Complete classwise results for both Phase IV studies\n\nMinimal includes early/on-time departures. Precision, recall and F1 below are fractions, not percentages. The two monthly populations contain 519,971 and 532,053 flights, with 1,052,024 pooled flights per fit. Counts are per model, not summed across models or seeds.\n\n"
    )
    for title, relative, source in [
        (
            "Original Phase IV class metrics",
            "reports/phase4/metric_details.json",
            False,
        ),
        ("Source-input class metrics", "reports/mlp_source/metric_details.json", True),
    ]:
        sections.append(f"### {title}\n\n")
        path = ROOT / relative
        rows = (
            class_records(json.loads(path.read_text(encoding="utf-8")), source)
            if path.exists()
            else []
        )
        sections.append(
            saved_table(
                rows,
                [
                    "regime",
                    "arm",
                    "seed",
                    "month",
                    "severity",
                    "precision",
                    "recall",
                    "f1",
                    "support",
                ],
            )
        )
    path = ROOT / "reports/mlp_source/results.md"
    if path.exists():
        interpretation = re.sub(
            r"\]\(([^/:#][^)]*)\)",
            r"](../reports/mlp_source/\1)",
            path.read_text(encoding="utf-8"),
        )
        sections.append(
            "\n## Interpretation of the source-input follow-up\n\n" + interpretation
        )
    sections.append(
        "\nThe executable sections below regenerate the analyses and expose complete aggregate CSVs, subgroups, contrasts, calibration and verification evidence. Original Phase IV subgroup analyses are retained; the source-input follow-up reports overall populations and does not claim new subgroup analyses.\n"
    )
    return "".join(sections)


def complete(notebook):
    notebook.cells[0].source += "\n\n" + complete_saved_results()
    notebook.cells[1].source += "\nsource_reports = repo / 'reports/mlp_source'\n"
    notebook.cells[3].source += """
display(Markdown('### Complete original Phase IV per-seed and subgroup metrics'))
all_phase4 = csv_rows(phase4 / 'metrics_summary.csv')
table(all_phase4, list(all_phase4[0]) if all_phase4 else [])
display(Markdown('### Source-input fixed/tuned results and matched deltas'))
source_means = csv_rows(source_reports / 'metrics_summary.csv')
source_seeds = csv_rows(source_reports / 'metrics_by_seed.csv')
table(source_means, list(source_means[0]) if source_means else [])
table(source_seeds, list(source_seeds[0]) if source_seeds else [])
for view in ['G1_G2_G3', 'G1_G2_G3_G4']:
    display(Markdown('### Complete source-input development evidence: ' + view))
    development = csv_rows(source_reports / ('tuning_' + view + '.csv'))
    table(development, list(development[0]) if development else [])
    selection_path = source_reports / ('selection_' + view + '.json')
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding='utf-8'))
        table([dict(view=view, candidate=selection['candidate_id'], **selection['candidate'],
                    epochs=selection['epochs'], mean_log_loss=selection['mean_log_loss'])],
              ['view','candidate','widths','dropout','learning_rate','epochs','mean_log_loss'])
if source_means:
    arms = ['MLP-A', 'MLP-B', 'MLP-C', 'MLP-D']
    fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
    for offset, regime, label in [(-.25, 'original', 'Original one-hot'),
                                  (0, 'fixed', 'Source inputs: fixed'),
                                  (.25, 'tuned', 'Source inputs: tuned')]:
        values = []
        for arm in arms:
            collection = seed_summary if regime == 'original' else source_means
            row = next(r for r in collection if r['arm'] == arm and r['month'] == 'pooled'
                       and (r.get('subgroup') == 'overall' if regime == 'original' else r['regime'] == regime))
            values.append(float(row['macro_f1']))
        ax.bar([i + offset for i in range(4)], values, width=.25, label=label)
    ax.set(xticks=range(4), xticklabels=arms, ylabel='Seed-mean macro-F1',
           title='Matched holdout: representation and development tuning')
    ax.legend(); plt.show(); plt.close(fig)
"""
    notebook.cells[5].source += """
source_detail_path = source_reports / 'metric_details.json'
source_details = json.loads(source_detail_path.read_text(encoding='utf-8')) if source_detail_path.exists() else {}
class_rows = []
for study, collection in [('Original Phase IV', details), ('Source-input follow-up', source_details)]:
    for key, values in sorted(collection.items()):
        parts = key.split('/')
        if study == 'Original Phase IV':
            arm, seed, month, subgroup = parts
            if subgroup != 'overall': continue
            regime = 'original'
        else:
            view, regime, arm, seed, month = parts
        for index, label in enumerate(['Minimal', 'Moderate', 'Severe']):
            class_rows.append(dict(study=study, regime=regime, arm=arm, seed=seed, month=month,
                severity=label, precision=values['precision'][index], recall=values['recall'][index],
                f1=values['f1'][index], support=values['support'][index]))
display(Markdown('### All classes, seeds and holdout periods'))
table(class_rows, ['study','regime','arm','seed','month','severity','precision','recall','f1','support'])
if source_details:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    for key, values in sorted(source_details.items()):
        view, regime, arm, seed, month = key.split('/')
        if seed != '6740' or month != 'pooled': continue
        label = arm + '/' + regime
        bins = [r for r in values['calibration'] if r['class_id'] == 2 and r['n']]
        axes[0].plot([r['mean_probability'] for r in bins], [r['observed_rate'] for r in bins], marker='.', label=label)
        curve = values.get('severe_pr_curve', [])
        if curve:
            recall, precision = zip(*curve)
            axes[1].plot(recall, precision, label=label)
    axes[0].plot([0, 1], [0, 1], color='gray', linestyle=':')
    axes[0].set(xlabel='Mean severe probability', ylabel='Observed severe fraction')
    axes[1].set(xlabel='Severe recall', ylabel='Severe precision')
    axes[1].legend(fontsize='small')
    fig.suptitle('Source-input comparisons: prespecified seed 6740 (all seed scores above)')
    plt.show(); plt.close(fig)
"""
    notebook.cells[7].source += """
display(Markdown('### Original Phase IV verification evidence'))
for name in ['upstream_verification.json','weather_source_verification.json','weather_g4_verification.json',
             'g4_audit_only_verification.json','frozen_replay.json','full_refit_reproducibility.json']:
    path = phase4 / name
    if path.exists():
        evidence = json.loads(path.read_text(encoding='utf-8'))
        display(Markdown('**' + name + '**'))
        display(evidence)
display(Markdown('### Source-input verification, fit counts and learning curves'))
for name in ['development_verification.json','holdout_verification.json','detail_export_verification.json']:
    path = source_reports / name
    if path.exists():
        display(Markdown('**' + name + '**'))
        display(json.loads(path.read_text(encoding='utf-8')))
manifest_path = source_reports / 'manifest.json'
if manifest_path.exists():
    source_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    compute_rows, curve_rows = [], []
    for unit in source_manifest.get('fits', []):
        ctx, fit = unit['context'], unit['fit']
        identity = dict(view=ctx['view'], regime=ctx['regime'], arm=ctx['arm'], seed=ctx['seed'])
        compute_rows.append(dict(**identity, train_rows=fit['rows'], epochs=fit['epochs'], fit_seconds=fit['seconds']))
        curve_rows.extend(dict(**identity, **point) for point in fit['curves'])
    table(compute_rows, ['view','regime','arm','seed','train_rows','epochs','fit_seconds'])
    table(curve_rows, ['regime','arm','seed','epoch','train_loss'])
    print('Source-input fit status:', source_manifest['status'], source_manifest['completed'], '/', source_manifest['expected'])
"""
    for i, cell in enumerate(notebook.cells):
        cell.id = f"phase4-{i:02}"
    return notebook


def build():
    markdown = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    notebook = nbf.v4.new_notebook(
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
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
    return complete(notebook)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", type=Path, help="Write the executed notebook to this path"
    )
    parser.add_argument("--aggregate-root", type=Path, default=ROOT)
    args = parser.parse_args()
    notebook = build()
    if args.execute:
        notebook = nbf.read(NOTEBOOK, as_version=4)
        notebook = NotebookClient(
            notebook,
            timeout=120,
            kernel_name="python3",
            resources={"metadata": {"path": str(args.aggregate_root.resolve())}},
        ).execute()
        args.execute.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(notebook, args.execute)
    else:
        NOTEBOOK.write_bytes(nbf.writes(notebook).encode("utf-8"))


if __name__ == "__main__":
    main()
