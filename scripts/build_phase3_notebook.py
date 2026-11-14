"""Recreate the lightweight aggregate-only analysis notebook; never train models."""

from pathlib import Path
import json


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

Recreate inputs with `phase3-report --config configs/phase3.yaml`; rebuild this
notebook with `python scripts/build_phase3_notebook.py`. Install the `notebook`
extra. Missing experiments produce explicit unavailable messages, never fabricated
scores. Synthetic smoke outputs are kept private and are not used here.
""")
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
