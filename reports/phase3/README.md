# Phase III artifact guide

Start with [the classical-model analysis notebook](../../notebooks/03_classical_models_analysis.ipynb). It contains the methodology, interpretation, saved tables for all 13 configurations in each holdout period, and selected parameter settings. Per-class tables include minimal/moderate/severe precision, recall, F1 and support for every configuration and period, plus a focused Random Forest G1_G2 table with predicted counts. Its analysis cells regenerate figures from these aggregate reports without training. The saved Markdown tables remain readable without running a kernel.

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
| `baseline_revision.json`, `baseline_revision_verification.json` | Current post-holdout CDF baseline overlay, original/revised scores, source hashes and independent full-population replay; learned results unchanged |
| `feature_label_coverage.json` | Full-population CSV/Parquet/working-feature ID alignment, label validity, class supports and original export checksums |
| `PHASE3_VALIDATION.md`, `METHODOLOGY_CHECKS.md`, `test_results.txt` | Completion status, limitations and named acceptance/test evidence |
| `CSV_READINESS.md`, `csv_readiness_*.csv`, `csv_readiness_*.json` | Historical fixed-subset diagnostic, explicitly excluded from official scores |

Private models, flight-level predictions and operational logs remain under ignored `data/model_outputs/phase3_pace/`. Executed analysis copies are under ignored `data/model_outputs/phase3/notebook_checks/`; `official_phase3_cdf_baseline.executed.ipynb` is the current amended rendered copy; `official_phase3_final.executed.ipynb` retains the original argmax-baseline analysis. Historical runs and transfer/verification provenance are retained separately under ignored `data/model_outputs/phase3/`. Disposable caches moved during cleanup are recorded in that directory's `cleanup_archive/` manifest. No source data or fitted checkpoint was removed.

Rebuild the saved methodology/results source with `python scripts/build_phase3_notebook.py`. Recreate the rendered figures with `python scripts/execute_phase3_notebook.py --check-name official_phase3_cdf_baseline`. Both commands use persisted aggregate results; neither retrains or selects models. After changing notebook prose, refresh `notebook_validation.json` to describe that source and executed copy. Aggregate CSV/JSON bytes are preserved by `.gitattributes` so their provenance hashes survive cross-platform checkout.

The current baseline samples its class from the historical-probability CDF; learned models still use argmax. Original baseline classifications and signed store are retained privately and through Git history. Use the baseline-revision script described in the methodology to rebuild amended comparisons; the original report exporter alone recreates the historical argmax baseline. Original scientific manifest hashes are historical; report revision hashes describe the current implementation and notebook.

Current baseline prediction rows are saved separately under `data/model_outputs/phase3_pace/baseline_revisions/prior_cdf_sha256_v1/baseline-cdf-d16c277eb09b46119ffa/predictions/`. The original run?s combined prediction file retains its original baseline; current comparisons combine the twelve unchanged learned-model results with this separately verified baseline overlay.
