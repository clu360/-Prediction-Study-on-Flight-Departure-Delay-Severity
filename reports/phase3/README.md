# Phase III artifact guide

Start with [the classical-model analysis notebook](../../notebooks/03_classical_models_analysis.ipynb). It contains the methodology, interpretation, saved tables for all 13 configurations in each holdout period, and selected parameter settings. Its analysis cells regenerate figures from these aggregate reports without training. The saved Markdown tables remain readable without running a kernel.

| Files | Purpose |
|---|---|
| `metrics_summary.csv`, `metric_details.json` | Official development and frozen holdout metrics, class/calibration details and airport slices |
| `candidate_scores.json`, `selected_configs.json` | Development-only screening/ranking evidence and frozen choices |
| `feature_importance.json` | Encoded-column importances mapped to source predictors and groups |
| `experiment_manifest.json`, `batch_execution.json`, `runtime_environment.json` | Scientific provenance, scheduler workflow and runtime versions |
| `execution_summary.json` | Unique fit counts, timings, memory and all candidate/final warnings |
| `result_verification.json`, `reproducibility_check.json` | Full saved-prediction verification and four independent same-platform family repeats |
| `cross_platform_reproducibility.json` | Separate Windows/PACE diagnostic, including XGBoost differences; not a passed strict repeat |
| `notebook_validation.json` | Aggregate-only notebook execution, source/output hashes and input checksums |
| `feature_label_coverage.json` | Full-population CSV/Parquet/working-feature ID alignment, label validity, class supports and original export checksums |
| `PHASE3_VALIDATION.md`, `METHODOLOGY_CHECKS.md`, `test_results.txt` | Completion status, limitations and named acceptance/test evidence |
| `CSV_READINESS.md`, `csv_readiness_*.csv`, `csv_readiness_*.json` | Historical fixed-subset diagnostic, explicitly excluded from official scores |

Private models, flight-level predictions and operational logs remain under ignored `data/model_outputs/phase3_pace/`. Executed analysis copies are under ignored `data/model_outputs/phase3/notebook_checks/`; `official_phase3_final.executed.ipynb` is the current rendered copy. Historical runs and transfer/verification provenance are retained separately under ignored `data/model_outputs/phase3/`. Disposable caches moved during cleanup are recorded in that directory's `cleanup_archive/` manifest. No source data or fitted checkpoint was removed.

Rebuild the saved methodology/results source with `python scripts/build_phase3_notebook.py`. Recreate the rendered figures with `python scripts/execute_phase3_notebook.py --check-name official_phase3_final`. Both commands use persisted aggregate results; neither retrains or selects models. After changing notebook prose, refresh `notebook_validation.json` to describe that source and executed copy. Aggregate CSV/JSON bytes are preserved by `.gitattributes` so their provenance hashes survive cross-platform checkout.
