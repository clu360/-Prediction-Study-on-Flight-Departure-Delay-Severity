"""Recreate the lightweight aggregate-only analysis notebook; never train models."""

from pathlib import Path
import csv
import json


def saved_result_tables(root):
    """Keep the complete aggregate comparison readable without running a kernel."""
    reports = root / "reports/phase3"
    required = [reports / "metrics_summary.csv", reports / "selected_configs.json"]
    if not all(path.exists() for path in required):
        return (
            "\n## Saved result tables\n\nOfficial aggregate inputs are unavailable.\n"
        )
    selected = json.loads(required[1].read_text(encoding="utf-8"))
    if not selected.get("complete"):
        return (
            "\n## Saved result tables\n\nOfficial parameter selection is incomplete.\n"
        )
    with required[0].open(encoding="utf-8", newline="") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["airport"] == "overall"
            and row["evaluation_stage"] == "frozen_holdout"
        ]
    if not rows:
        return "\n## Saved result tables\n\nOfficial holdout results are unavailable.\n"
    assert len(rows) == 39
    assert {row["run_id"] for row in rows} == {selected["run_id"]}
    sections = [
        "\n## Saved results for every prespecified model\n\n"
        "These tables are generated from the persisted aggregate reports when "
        "the notebook source is built. They remain visible without executing "
        "cells. The analysis cells below recreate the tables and figures from "
        "the same reports; no model is retrained. Scores are rounded to four "
        "decimal places here; CSV/JSON reports retain full exported precision.\n"
    ]
    for period, label in [
        ("2025-11_12_pooled", "Pooled November and December"),
        ("2025-11", "November"),
        ("2025-12", "December"),
    ]:
        current = sorted(
            (row for row in rows if row["scored_month"] == period),
            key=lambda row: (row["model"], row["feature_view"]),
        )
        assert len(current) == 13
        sections.append(
            f"\n### {label}\n\n"
            "| Model | View | Flights | Accuracy | Macro-F1 | Severe recall | Severe precision | Log loss | Brier |\n"
            "|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        for row in current:
            values = [row["model"], row["feature_view"], f"{int(row['n']):,}"]
            values.extend(
                f"{float(row[key]):.4f}"
                for key in [
                    "accuracy",
                    "macro_f1",
                    "severe_recall",
                    "severe_precision",
                    "log_loss",
                    "brier",
                ]
            )
            sections.append("| " + " | ".join(values) + " |\n")
    sections.append(
        "\n### Frozen settings selected using development months only\n\n"
        "The baseline uses training-class proportions. Other rows show the "
        "full selected parameter dictionary. Fixed factory settings include "
        "seed 6740 and at most four estimator/BLAS threads. Development means "
        "give each of the seven months equal weight and retain selection optimism.\n\n"
        "| Model | View | Development mean macro-F1 | Month standard deviation | Selected parameters |\n"
        "|---|---|---:|---:|---|\n"
    )
    for choice in sorted(
        selected["selected"].values(),
        key=lambda row: (row["model"], row["feature_view"]),
    ):
        sections.append(
            f"| {choice['model']} | {choice['feature_view']} | "
            f"{choice['mean_macro_f1']:.4f} | {choice['macro_f1_std']:.4f} | "
            f"`{json.dumps(choice['params'], sort_keys=True)}` |\n"
        )
    coverage_path = reports / "feature_label_coverage.json"
    if coverage_path.exists():
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        assert coverage["passed"] and coverage["run_id"] == selected["run_id"]
        sections.append(
            "\n### Full feature-label coverage audit\n\n"
            f"All **{coverage['rows']:,} feature rows** have exactly one matching "
            "valid label. Labels are intentionally stored in the separate "
            "targets/cohort tables, not among the twenty predictors. The audit "
            "checked the complete CSV feature export, Parquet feature export "
            "and Phase III working feature partitions against the original "
            "cohort and stored targets. No missing, extra or duplicate IDs, "
            "missing labels, month disagreements, class disagreements or signed "
            "departure-delay disagreements were found. Every label satisfies "
            "the signed 15/60-minute thresholds; all three classes and twelve "
            "months are present. Original export checksums still match.\n\n"
            "| Class | Departure delay | Full-cohort support |\n"
            "|---|---|---:|\n"
        )
        descriptions = {0: "<15 minutes", 1: "15 to <60 minutes", 2: ">=60 minutes"}
        for row in coverage["class_counts"]:
            sections.append(
                f"| {row['target_class']} | {descriptions[row['target_class']]} | "
                f"{row['len']:,} |\n"
            )
        sections.append(
            "\nThese are full-year cohort counts, distinct from the smaller "
            "November/December supports above. The check reads existing files "
            "without fitting models or adding labels to predictors. Detailed "
            "aggregate evidence is saved in "
            "[feature_label_coverage.json](../reports/phase3/feature_label_coverage.json).\n"
        )
    return "".join(sections)


def build():
    cells = []

    def md(text):
        cells.append(
            dict(
                cell_type="markdown", metadata={}, source=text.splitlines(keepends=True)
            )
        )

    def code(text):
        cells.append(
            dict(
                cell_type="code",
                metadata={},
                source=text.splitlines(keepends=True),
                execution_count=None,
                outputs=[],
            )
        )

    md("""# Phase III: classical flight-delay models

This notebook reads persisted **aggregate** reports only. It does not fit models,
tune hyperparameters, construct features, or access flight-level records.

Development: seven rolling three-month train / next-month validation folds,
April–October 2025. Screen on April/July/October, then compare at most two
finalists on all seven months using equal-month macro-F1 and declared tie breakers.
Selected-config development scores have selection optimism.

Primary holdout: one fresh August–October model and transformer, frozen for both
November (one month ahead) and December (two months ahead). Causal G3 inputs may
update from earlier observed test-month departures; learned state never updates.
Pooled metrics are recomputed from all held-out predictions. Classes: 0 minimal
(<15 min), 1 moderate (15–<60), 2 severe (>=60).

Recreate the panels from the committed aggregates with `python scripts/execute_phase3_notebook.py --check-name official_phase3_final`. Rebuilding aggregates requires the original private checkpoints: run `phase3-report --config configs/phase3_pace.yaml`, then `python scripts/summarize_phase3_execution.py --config configs/phase3_pace.yaml`. Rebuild this source with `python scripts/build_phase3_notebook.py`. Install the `notebook` extra. Missing experiments are marked unavailable. Synthetic smoke scores remain private and do not enter the study.

## Methodology

The eligible cohort contains 6,461,609 flights from the validated 2025 BTS inputs. Phase I aircraft links, target membership and Phase II engineered features are reused unchanged. Prediction time is the preceding aircraft leg's actual wheels-off time plus one minute, before the target's scheduled departure. The prior flight's departure delay is available; prior arrival outcomes, target outcomes, delay causes, tail IDs and record IDs are excluded from predictors. Cancelled/diverted flights remain available when constructing the upstream aircraft chain, while eligible prediction targets follow the validated Phase I exclusions.

### Features and causal availability

| Group | Source features | Information available at prediction time |
|---|---:|---|
| G1 | 9 | Carrier, origin, destination, cyclic departure hour, weekday, month, scheduled elapsed minutes and distance |
| G2 | 5 | Prior origin, signed prior departure delay, prior scheduled elapsed minutes, scheduled turnaround and minutes until target scheduled departure |
| G3 | 6 | Airport moderate/severe rates and mean departure delay over 28 days, airport scheduled volume over 7 days, and airline-at-airport severe rate and observation count over 28 days |

The four views are G1 (9 columns), G1_G2 (14), G1_G3 (15), and G1_G2_G3 (20). History windows include their left boundary and exclude prediction time: [t-28 days,t) or [t-7 days,t). A target's own outcome is excluded. Rates use the Phase II smoothing/fallback rules (alpha=20); no histories or links are rebuilt here. The full formulas and source schema are in [the feature dictionary](../docs/phase2_feature_dictionary.md).

### Chronological fitting and parameter selection

| Training months in 2025 | Scored month | Use |
|---|---|---|
| January-March | April | Development; screening |
| February-April | May | Development; finalists |
| March-May | June | Development; finalists |
| April-June | July | Development; screening |
| May-July | August | Development; finalists |
| June-August | September | Development; finalists |
| July-September | October | Development; screening |
| August-October | November and December | One fresh frozen fit; both final holdouts |

There is no random train/test split. Every window fits numeric medians/scaling and categorical vocabularies on its training rows only, then freezes them. Unknown categories map to all-zero one-hot blocks; the following month's month category is necessarily unseen. Matrices are sparse CSR float32. IDs and labels are checked one-to-one, and all compared models score identical populations.

The models are the class-prior baseline, L2 logistic regression with SAGA, bootstrap random forest, and CPU-histogram XGBoost 3.1.2 with three-class soft probabilities. Logistic/forest/XGBoost grids contain 6/8/8 candidates per view. The three screening months require 264 learned fits plus seven baseline fits; two finalists per family/view add 96 fits across the remaining four months. Selected backtests reuse verified candidate predictions. Thus development has 367 fits; the 13 fresh final fits bring the scientific total to 380. Four additional independent seeded family repeats are verification only.

Rank candidates by equal-month mean macro-F1. Within the declared 1e-6 tolerance, use lower mean log loss, lower population standard deviation of monthly macro-F1, lower median fit time, then lexical parameter JSON. Finalists use all seven development months. Freeze the 13 configurations and selection checksum before accessing either holdout. August-October is the only final training population; December does not refit on September-November. Earlier observed holdout outcomes may enter causal G3 inputs, but never update weights, preprocessing, thresholds or calibration.

### Evaluation and independent checks

Probabilities are ordered [minimal, moderate, severe], finite, within [0,1], and sum to one within 1e-6 before numerical normalization of roundoff. Predictions use argmax. Metrics include accuracy, macro/weighted F1, per-class precision/recall/F1, log loss, multiclass Brier (mean summed squared error over three classes), one-vs-rest AUC, confusion matrices and ten equal-width reliability bins with counts. Undefined AUC is explicitly null. Pooled metrics use concatenated held-out predictions. ATL, DFW, DEN, ORD and LAX are target-origin evaluation slices of network-wide training.

Verification rereads every stored prediction, checks original IDs/labels, class order/simplex/argmax and consolidated-file equality, then recomputes monthly, pooled and airport metrics. Each saved model/transformer is reloaded for a deterministic 1,024-row replay per month; state hashes must stay unchanged. The release suite passed 112 tests on Linux (111 plus one symlink-privilege skip on Windows). Four independent PACE family repeats matched within 1e-7; this does not establish that all 380 fits were repeated. The separate Windows/PACE diagnostic records XGBoost numerical differences explicitly. See [the acceptance checklist](../reports/phase3/METHODOLOGY_CHECKS.md) for all 14 requirements and [validation](../reports/phase3/PHASE3_VALIDATION.md) for limitations and recovery history.

## Interpretation of the verified classical-model study

These findings describe run `phase3-709172c896241ac3a355`. All 13 configurations were frozen using April-October development scores before final evaluation. The 367 development fits and 13 fresh August-October fits completed; each final model and transformer stayed identical for November and December. Independent checks covered all 13,676,312 saved final prediction rows and replayed 26,624 rows from reloaded models. These holdout comparisons describe the prespecified models and do not change the selected configurations or decision rule.

The two-month population contains **1,052,024 flights**: 519,971 in November and 532,053 in December. Minimal, moderate and severe supports are 830,013, 154,436 and 67,575. The prior baseline attains 78.9% accuracy but macro-F1 0.2940 and zero severe recall. The G1 and G1_G3 forests also predict the minimal class for every held-out row under argmax, illustrating why accuracy alone is insufficient.

XGBoost with incoming-aircraft information produces the largest observed pooled macro-F1 among the prespecified configurations. The complete 13-model/view tables and three separate evaluation periods remain below.

| XGBoost view | November macro-F1 | December macro-F1 | Pooled macro-F1 | Pooled severe recall | Pooled severe precision | Pooled log loss |
|---|---:|---:|---:|---:|---:|---:|
| G1_G2 | 0.6407 | 0.6511 | 0.6475 | 0.3792 | 0.8831 | 0.4409 |
| G1_G2_G3 | 0.6400 | 0.6510 | 0.6471 | 0.3829 | 0.8773 | 0.4398 |

For XGBoost, adding G2 to G1 raises pooled macro-F1 from 0.2975 to 0.6475. Adding G3 to G1_G2 changes macro-F1 by -0.00036, improves log loss slightly, and raises severe recall from 37.9% to 38.3%. G3 has no consistent macro-F1 gain across views and airports in this study. All four XGBoost views selected the same hyperparameters; other families have separately tuned settings. Small observed differences carry no statistical-significance claim, because correlated flights were not given IID confidence intervals.

The G1_G2 XGBoost model identifies 37.9% of severe delays with 88.3% precision. Its mean severe probability is 5.643%, the observed severe share is 6.423%, and the severe argmax share is 2.758%. These three quantities describe different properties. Logistic G1_G2_G3 attains higher severe recall (45.9%) with lower precision (33.3%) and macro-F1 (0.4997). No threshold or calibration model was fitted after examining holdout outcomes.

The five airports are evaluation slices; training uses the complete eligible network. For pooled XGBoost G1_G2:

| Target origin | Flights | Macro-F1 | Severe recall |
|---|---:|---:|---:|
| ATL | 46,653 | 0.5927 | 0.2980 |
| DFW | 45,044 | 0.5686 | 0.2723 |
| DEN | 47,463 | 0.6481 | 0.3945 |
| ORD | 47,798 | 0.6089 | 0.3730 |
| LAX | 28,484 | 0.6102 | 0.3178 |

Airport one-hot columns absorb substantial summed XGBoost gain. Category count, correlated predictors and encoding affect these associative importances; the prespecified feature-group performance ablations provide complementary evidence. Importance values are not causal effects.

Across the 380 scientific fits, 8 development logistic G1_G3 fits reached the fixed 400-iteration cap; no final fit recorded a warning. Summed unique fit elapsed time is 35.90 hours across parallel tasks, with a maximum sampled process RSS of 4.88 GiB. These fit hours exclude preprocessing, inference and independent repeats and are not billed CPU hours or end-to-end elapsed time. Four independent full-population PACE repeats passed at 1e-7; the separate 53-fit Windows/PACE diagnostic records 16 XGBoost mismatches, with maximum probability difference 0.17829. Official results use the verified PACE environment. Linkage selection, retrospective tail assignments, left-censored January histories and the single-year cohort limit operational generalization.
""")
    cells[-1]["source"].extend(
        saved_result_tables(Path(__file__).resolve().parents[1]).splitlines(
            keepends=True
        )
    )
    code("""from pathlib import Path
import json
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'pyproject.toml').exists())
REPORTS = ROOT / 'reports' / 'phase3'
def load_json(name, fallback):
    path = REPORTS / name
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else fallback
manifest = load_json('experiment_manifest.json', {'completion_status': 'unavailable'})
selected = load_json('selected_configs.json', {'complete': False, 'selected': {}})
details = load_json('metric_details.json', [])
path = REPORTS / 'metrics_summary.csv'
metrics = pl.read_csv(path) if path.exists() else pl.DataFrame()
overall = metrics.filter(pl.col('airport') == 'overall') if metrics.height else metrics
display(Markdown('**Run status:** ' + manifest['completion_status']))
if not metrics.height:
    display(Markdown('Official full-study model scores are unavailable. CSV readiness diagnostics, if present, are shown separately below.'))
""")
    md("""## Completed CSV readiness diagnostic — not official study results

This fixed chronological subset checks whether the real CSVs can train models and
produce valid outputs. It uses the earliest 2,000 targets per month: 6,000
January–March training rows and 2,000 April scoring rows. Model settings are the
first predeclared candidates, not development-tuned selections. These scores do
not establish a model ranking or replace the seven-fold and frozen-holdout study.

Recreate these aggregate inputs with `python scripts/export_csv_readiness.py`.
The notebook reads no flight-level diagnostic predictions.
""")
    code("""csv_summary = load_json('csv_readiness_summary.json', {})
csv_path = REPORTS / 'csv_readiness_metrics.csv'
csv_metrics = pl.read_csv(csv_path) if csv_path.exists() else pl.DataFrame()
if csv_metrics.height:
    assert csv_summary.get('study_results') is False
    display(Markdown('**CSV records verified:** {:,}. **Diagnostic model/view combinations:** {}. **Prediction vectors saved:** {:,}.'.format(
        csv_summary['csv_verification']['rows'], csv_summary['model_view_combinations'], csv_summary['prediction_rows'])))
    display(csv_metrics.select('model','feature_view','train_rows','test_rows','macro_f1','severe_recall','log_loss','brier','fit_seconds'))
    labels = [m+'/'+v for m,v in zip(csv_metrics['model'],csv_metrics['feature_view'])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(labels, csv_metrics['macro_f1'].to_list())
    axes[0].set(ylabel='Macro-F1', ylim=(0,1))
    axes[1].bar(labels, csv_metrics['log_loss'].to_list())
    axes[1].set(ylabel='Multiclass log loss')
    for ax in axes:
        ax.tick_params(axis='x', rotation=90)
    fig.suptitle('CSV readiness diagnostic only: fixed small subset, untuned settings')
    plt.tight_layout(); plt.show()
else:
    display(Markdown('CSV readiness aggregates unavailable. Run the exporter after completing the diagnostic.'))
""")
    md(
        "## Development-only parameter selection\nThe selection table is persisted before opening holdout targets. Full candidate screening evidence is in `candidate_scores.json`."
    )
    code("""choices = [dict(model=x['model'], view=x['feature_view'], parameters=json.dumps(x['params'], sort_keys=True),
                mean_macro_f1=x['mean_macro_f1'], month_std=x['macro_f1_std'],
                mean_log_loss=x['mean_log_loss']) for x in selected.get('selected', {}).values()]
display(pl.DataFrame(choices)) if choices else display(Markdown('Parameter selection unavailable.'))
candidate_scores = load_json('candidate_scores.json', [])
if candidate_scores:
    display(pl.DataFrame(candidate_scores).select('model','feature_view','config_id','scored_month','macro_f1','log_loss'))
""")
    md(
        "## Rolling development trends\nThese are selected-configuration backtests, not independent test results. Each month receives equal weight in selection."
    )
    code("""dev = overall.filter(pl.col('evaluation_stage') == 'development') if overall.height else overall
if dev.height:
    fig, ax = plt.subplots(figsize=(11, 5))
    for key, rows in dev.sort('scored_month').partition_by(['model','feature_view'], as_dict=True).items():
        ax.plot(rows['scored_month'].to_list(), rows['macro_f1'].to_list(), marker='o', label='/'.join(key))
    ax.set(ylabel='Macro-F1', title='Development: selected configurations (selection optimism)')
    ax.legend(bbox_to_anchor=(1.02, 1), fontsize=8)
    plt.show()
else:
    display(Markdown('Development trends unavailable.'))
""")
    md(
        "## Frozen November / December primary evaluation\nBaseline appears once per period. Pooled scores are evaluated on concatenated predictions, rather than averaging monthly metrics."
    )
    code("""final = overall.filter(pl.col('evaluation_stage') == 'frozen_holdout') if overall.height else overall
if final.height:
    display(final.select('model','feature_view','scored_month','n','macro_f1','severe_recall','severe_precision','log_loss','brier','macro_auc'))
    for period in final['scored_month'].unique().sort():
        frame = final.filter(pl.col('scored_month') == period)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar([m+'/'+v for m,v in zip(frame['model'],frame['feature_view'])], frame['macro_f1'].to_list())
        ax.tick_params(axis='x', rotation=80)
        ax.set(title=period + ': frozen model', ylabel='Macro-F1')
        plt.show()
else:
    display(Markdown('Frozen holdout metrics unavailable.'))
""")
    md(
        "## Feature-group ablations\nCompare identical target IDs within model/month. Configurations are independently development-tuned, so changes reflect information plus the selected hyperparameters. Associations do not establish causation."
    )
    code("""ablation=[]
if final.height:
    lookup={(r['model'],r['feature_view'],r['scored_month']):r for r in final.to_dicts()}
    for (model,view,month), row in lookup.items():
        for base, added in [('G1','G1_G2'),('G1','G1_G3'),('G1_G2','G1_G2_G3'),('G1_G3','G1_G2_G3')]:
            ref=lookup.get((model,base,month))
            if view==added and ref:
                ablation.append(dict(model=model, scored_month=month, comparison=added+' minus '+base,
                    delta_macro_f1=row['macro_f1']-ref['macro_f1'], delta_log_loss=row['log_loss']-ref['log_loss']))
display(pl.DataFrame(ablation)) if ablation else display(Markdown('Ablations unavailable.'))
""")
    md(
        "## Confusion matrices, calibration and probability distributions\nEach model/view has separate November, December and pooled panels. Reliability bins are ten equal-width intervals with counts. Brier score is mean summed squared error over three classes. Mean soft probabilities differ from argmax class proportions."
    )
    code("""holdout_details = [r for r in details if r['evaluation_stage']=='frozen_holdout']
for r in holdout_details:
    m=r['metrics']
    fig, axes=plt.subplots(1,3,figsize=(14,3.5))
    axes[0].imshow(m['confusion_matrix'], cmap='Blues')
    axes[0].set(xlabel='Predicted class',ylabel='Observed class',title='Confusion matrix',xticks=range(3),yticks=range(3))
    counts=np.asarray(m['confusion_matrix'])
    for observed in range(3):
        for predicted in range(3):
            axes[0].text(predicted,observed,format(counts[observed,predicted],','),ha='center',va='center',
                color='white' if counts[observed,predicted] > counts.max()/2 else 'black')
    for c in range(3):
        points=[b for b in m['calibration'] if b['class_id']==c and b['n']>0]
        axes[1].plot([b['mean_probability'] for b in points],[b['observed_rate'] for b in points],marker='o',label=str(c))
    axes[1].plot([0,1],[0,1],'k--'); axes[1].legend(title='Class')
    axes[1].set(xlabel='Mean predicted probability',ylabel='Observed rate',title='Reliability (counts in report)')
    x=np.arange(3)
    for offset,key,label in [(-.25,'observed_proportions','Observed'),(0,'mean_probabilities','Mean probability'),(.25,'argmax_proportions','Argmax share')]:
        axes[2].bar(x+offset,m[key],width=.25,label=label)
    axes[2].set(xticks=x,title='Class distributions'); axes[2].legend(fontsize=8)
    fig.suptitle(r['model']+'/'+r['feature_view']+' '+r['scored_month']+' frozen holdout'); plt.show(); plt.close(fig)
if not holdout_details:
    display(Markdown('Confusion/calibration/distribution panels unavailable.'))
""")
    md(
        "## Five origin-airport comparisons\nModels train on the entire eligible network. ATL, DFW, DEN, ORD and LAX are evaluation slices. Counts and missing-class AUC limitations remain explicit."
    )
    code("""airport_rows=[]
for r in holdout_details:
    for airport,m in r.get('airport_metrics',{}).items():
        airport_rows.append(dict(model=r['model'],view=r['feature_view'],period=r['scored_month'],airport=airport,
            n=m['n'],macro_f1=m['macro_f1'],severe_recall=m['severe_recall'],log_loss=m['log_loss'],
            observed_minimal=m['observed_proportions'][0],observed_moderate=m['observed_proportions'][1],observed_severe=m['observed_proportions'][2],
            mean_p_minimal=m['mean_probabilities'][0],mean_p_moderate=m['mean_probabilities'][1],mean_p_severe=m['mean_probabilities'][2],
            argmax_severe_share=m['argmax_proportions'][2]))
display(pl.DataFrame(airport_rows)) if airport_rows else display(Markdown('Airport comparisons unavailable.'))
""")
    md(
        "## Feature interpretation\nLogistic coefficients use summed absolute values across classes. Tree importances are impurity/gain summaries and can favor certain encodings; these are associative summaries, not causal effects. No attribution feeds back into selection."
    )
    code("""importance=load_json('feature_importance.json',[])
if importance:
    frame=pl.DataFrame(importance)
    source=frame.group_by('model_view','source_feature','group').agg(pl.col('importance').sum()).sort('model_view','importance',descending=[False,True])
    groups=source.group_by('model_view','group').agg(pl.col('importance').sum()).sort('model_view','group')
    display(source); display(groups)
else:
    display(Markdown('Feature importance unavailable. Optional permutation analyses are not enabled.'))
""")
    md(
        "## Compute, uncertainty and limitations\nRuntime/peak RSS are measured per process; shared matrix preprocessing is reported for each candidate but must not be summed as exclusive candidate cost. Budget checks are cooperative between native operations. Counts are reported without an IID confidence interval because flights are correlated. January histories are left-censored; linkage changes severe-class prevalence; retrospective tail assignments and rotation contradictions limit operational interpretation. Other-year generalization has not been tested."
    )
    code("""compute=[dict(model=r['model'],view=r['feature_view'],period=r['scored_month'],**r.get('resources',{})) for r in details]
display(pl.DataFrame(compute,infer_schema_length=None)) if compute else display(Markdown('Experiment resource measurements unavailable.'))
execution=load_json('execution_summary.json',{})
if execution and execution.get('run_id') == manifest.get('run_id'):
    display(Markdown('**Unique fitted units:** ' + json.dumps(execution.get('fit_counts',{}))))
    display(pl.DataFrame(execution.get('compute',[]),infer_schema_length=None))
    display(Markdown(execution.get('resource_interpretation','')))
    display(Markdown('**Warnings across all candidate and final fits:** ' + json.dumps(execution.get('warning_counts',{}))))
    if execution.get('fit_warnings'): display(pl.DataFrame(execution['fit_warnings'],infer_schema_length=None))
else:
    display(Markdown('The complete unique-fit compute/warnings summary is not yet available.'))
display(Markdown('**Incomplete items / limits:** ' + json.dumps(manifest.get('limits',[]))))
display(Markdown('**Input validation:** ' + json.dumps(manifest.get('upstream_validation',{}))))
verification=load_json('result_verification.json',{})
if verification and verification.get('run_id') == manifest.get('run_id'):
    display(Markdown('**Independent saved-results verification:** ' + json.dumps({k:verification.get(k) for k in ['passed','models_verified','prediction_rows','checks']})))
else:
    display(Markdown('Independent saved-results verification is not yet available for the current run.'))
reproducibility=load_json('reproducibility_check.json',{})
if reproducibility:
    label='Current run' if reproducibility.get('corrected_run') == manifest.get('run_id') else 'Historical implementation check for '+str(reproducibility.get('corrected_run'))
    display(Markdown('**Repeated development predictions (verification only; '+label+'):** ' + json.dumps({k:reproducibility.get(k) for k in ['passed','complete','compared_fits','available_previous_fits','compared_prediction_rows','absolute_tolerance','scope']})))
else:
    display(Markdown('Repeated full-population development predictions are unavailable for this run; seeded synthetic checks are recorded in the test evidence.'))
cross_platform=load_json('cross_platform_reproducibility.json',{})
if cross_platform and cross_platform.get('corrected_run') == manifest.get('run_id'):
    display(Markdown('**Cross-platform diagnostic:** ' + json.dumps({k:cross_platform.get(k) for k in ['passed','complete','compared_fits','fits_exceeding_tolerance','absolute_tolerance','scope']})))
    display(pl.DataFrame(cross_platform.get('comparisons',[]),infer_schema_length=None))
""")
    notebook = dict(
        cells=cells,
        metadata=dict(
            kernelspec=dict(
                display_name="Flight Delay (Phase III)",
                language="python",
                name="flight-delay-phase3",
            ),
            language_info=dict(name="python", version="3.12"),
        ),
        nbformat=4,
        nbformat_minor=5,
    )
    for i, cell in enumerate(cells):
        cell["id"] = f"phase3-{i:02}"
    path = (
        Path(__file__).resolve().parents[1]
        / "notebooks/03_classical_models_analysis.ipynb"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(notebook, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    build()
