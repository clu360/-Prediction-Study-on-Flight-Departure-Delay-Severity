# CSV readability and real-data model-output check

Both complete CSV exports are now readable outside the sandbox. A sandboxed read still denied access to the engineered-feature CSV, so the verification and fitting command used the approved outside-sandbox execution path.

## Complete-file verification

- Both CSVs contain **6,461,609** records with unique, nonnull target IDs.
- Feature and cohort IDs match one-to-one across the complete files; feature months match cohort local source months.
- The feature CSV has ID plus exactly the approved 20 predictors. Its SHA-256 matches the checksum recorded when the validated Phase II table was exported.
- Cohort size, byte size and class supports match its export validation metadata. Every signed departure-delay label satisfies the original 15/60-minute thresholds.
- Class supports: minimal **5,199,881**, moderate **883,015**, severe **378,713**.
- Current input SHA-256 digests are saved with the private readiness report.

## Actual model outputs produced

Command: `python scripts/check_phase3_csv.py --rows-per-month 2000`, using the repository environment, four Polars/estimator threads and the fresh bytecode prefix.

This is an explicitly bounded **readiness diagnostic**, not official hyperparameter selection or a study leaderboard. It selects the earliest scheduled 2,000 targets in each of January, February, March and April. Models fit on **6,000 January–March** records and score the same **2,000 April** records. Each learned model uses the lexical-first predeclared candidate; no setting is selected from the resulting scores. No November/December evaluation is performed.

The baseline runs once; logistic regression, random forest and XGBoost each run on all four approved feature views: **13 fitted model/view combinations**, **26,000 probability vectors** total.

Private artifacts are under `data/model_outputs/phase3/csv_readiness/b4671a18ca771639/`:

- `readiness.json`: validation evidence, input hashes, diagnostics, parameters, timings and diagnostic metrics.
- `model_MODEL_VIEW.joblib`: estimator and its train-fitted transformer.
- `predictions_MODEL_VIEW.csv` and `.parquet`: stable IDs, observed class, ordered probabilities, predicted class and fit ID.

All 13 serialized models were reloaded successfully. All prediction files were reread: row counts, identical target IDs across models/views, finite probabilities, row sums, argmax labels and CSV/Parquet probability agreement passed.

Ten targeted tests passed, including three new CSV tests for complete ID/label verification, checksum corruption rejection and invalid-label rejection. The preceding full Phase III regression suite passed 80 tests.

## Generated aggregate reports and notebook

`python scripts/export_csv_readiness.py` generates `csv_readiness_summary.json`, `csv_readiness_metrics.csv` (13 model/view rows), and `csv_readiness_details.json` from persisted diagnostic outputs. These files contain aggregates only, with explicit `study_results=false` labeling. They remain separate from the completed official study metrics/selection files.

The analysis notebook includes a dedicated CSV readiness table and comparison chart, with a prominent fixed-subset/untuned-settings caveat. Its updated 20 cells execute in an aggregate-only checkout. The source notebook has cleared outputs; the executed copy is private at `data/model_outputs/phase3/notebook_checks/aggregate_only.executed.ipynb`. The user's earlier executed notebook was backed up as `before_csv_panel.ipynb` in that directory.

## Historical limitation and completed official study

The diagnostic originally ran while Windows cloud error 389 blocked the separate full upstream gate. That storage problem was subsequently resolved using checksum-equivalent private working copies. The fresh Phase I/II audit passed all 19 checks on 6,461,609 rows. The official Phase III run has now completed all 380 scientific fits and independently verified frozen holdout results. See [final validation](PHASE3_VALIDATION.md). Diagnostic scores remain explicitly excluded from study results; original records, features, folds and holdout protocols are unchanged.
