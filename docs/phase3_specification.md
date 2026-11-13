# Codex Implementation Plan: Phase III Classical Machine Learning

Project: ISyE 6740 | Prediction Study on Flight Departure Delay Severity

Repository: https://github.com/clu360/-Prediction-Study-on-Flight-Departure-Delay-Severity

Team: Caleb Lu, Ho Young Lee, Christopher Yi
Date: November 13, 2026

## 0. Task authorization and strict scope

Implement and execute Phase III only: probability baseline, multinomial logistic regression, random forest, XGBoost, rolling-window hyperparameter selection, strict final holdout evaluation, saved prediction results, and one reproducible analysis notebook. This brief authorizes classifiers and their evaluation but **not** the Phase IV MLP, rewriting Phase I/II data or features, or drafting the final course report.

Before any implementation, read the repository's AGENTS.md; docs/phase1_specification.md; docs/phase2_specification.md; docs/phase2_feature_dictionary.md; configs/feature_groups.yaml; src/flight_delay/features/{views,transforms}.py; and reports/PHASE2_VALIDATION.md. Verify their current versions against this brief. Update AGENTS.md to explicitly authorize Phase III while retaining every applicable Phase I/II safety, causal, data-protection, and allowlist restriction. Do not overwrite user modifications.

**Explicit methodology decision:** the final primary holdout protocol below supersedes the earlier Phase I document's permission to refit a separate September–November model for December. Our primary experiment deliberately uses **one frozen August–October-fitted model for both November and December**. Treat any monthly refit in the holdout as a separately labeled secondary experiment requiring authorization; it is not part of this task.

Work on a named feature branch if appropriate, keep code small and modular, test before claiming success, and prepare a reviewable change. Never commit flight-level data, prediction tables, serialized fitted models, raw inputs, paths, secrets, or large model checkpoints to the public repository. Small aggregate summaries, source code, YAML, tests, and a lightweight notebook are permitted.

## 1. Verified upstream contract (do not rebuild)

- Dataset: January–December 2025 BTS Reporting Carrier On-Time Performance, already preprocessed by Phases I and II.
- Valid linked forecasting cohort: **6,461,609** target flights. Confirm from live local artifacts, not a hard-coded expected count.
- For each target, predict **after the prior aircraft has taken off**, before its arrival and before the target scheduled departure. Prediction instant is predecessor wheels-off UTC plus one minute.
- Target labels: 0 when signed target departure delay < 15 minutes; 1 when delay is at least 15 but < 60; 2 when delay is at least 60. The three output classes must always be ordered [0,1,2].
- Existing feature source: data/features/full_features, data/features/targets, data/features/metadata (month/airport-partitioned Parquet); existing fold assignments: data/processed/folds.
- Keep all original target_record_id values and the Phase I split manifest. Never split randomly or derive a replacement cohort. Load labels by a checked one-to-one ID join, never by relying on position or sorting coincidence.
- Exactly 20 pre-encoding predictors: G1 (9 scheduled-flight features), G2 (5 incoming-aircraft features), G3 (6 causal airport/airline-history features). Do not add the redundant minimal-delay rate, IDs, actual outcomes, arrival outcomes, tail numbers, timestamps, audit flags, or unknown raw fields as predictors.
- Exactly four views, with the existing order: G1 (9), G1_G2 (14), G1_G3 (15), G1_G2_G3 (20). The feature groups remain **separate columns**, not three composite meta-features; group names are metadata for ablation and interpretation.
- Phase II features are point-in-time causal: historical observations contributing to a target's G3 feature have event_time strictly earlier than prediction_time. Within a test month, later flights may have updated G3 values from earlier *observed* departures; this is allowed **as input information only**, not fitted model updates.
- The existing build_fold(fold_id, feature_set, encoding="onehot", ...) in src/flight_delay/features/transforms.py returns X_train/X_test, y_train/y_test, train_ids/test_ids, fitted_transformer. It fits categorical vocabulary, imputation and scaling using that fold's training rows only; CSR float32 is required.
- An April full-size smoke test used roughly 1.51 million training flights, 0.55 million validation flights and 694–1037 encoded columns, depending on view. Resource estimates must account for millions of samples, sparse matrices, and expensive tree models.

**Pre-run gate:** run the existing Phase I/II validation and verify local input files, schemas, counts, 19 Phase II acceptance checks, and checksums. If full data are unavailable, create synthetic fixtures and run tests only. Never claim real model performance from synthetic fixtures.

## 2. Locked learning/evaluation protocol

Three separate layers must be clear in code, naming and reporting:

1. **Estimator parameters:** learned separately from the labeled training flights in each development fold (e.g., logistic regression coefficients or tree splits). Optimization of these parameters is the model's fit procedure, not the cross-fold hyperparameter search.
2. **Hyperparameters:** e.g., logistic C, forest depth, XGBoost tree depth/learning rate. The *same candidate configuration* is tried in multiple causally ordered development folds; scores across April–October select one configuration for each model/view.
3. **Strict final holdout:** after hyperparameters are chosen, instantiate and fit a **fresh estimator** and **fresh fitted transformer once** using August–October, and predict on both November and December without refitting or updating the estimator, encoders, imputers, scaling, calibrators, or other learned preprocessing state.

### 2.1 Development / hyperparameter selection folds

| Fold ID / scored month | Train months | Scored month | Purpose |
|---|---|---|---|
| 2025-04 | Jan–Mar | Apr | Development |
| 2025-05 | Feb–Apr | May | Development |
| 2025-06 | Mar–May | Jun | Development |
| 2025-07 | Apr–Jun | Jul | Development |
| 2025-08 | May–Jul | Aug | Development |
| 2025-09 | Jun–Aug | Sep | Development |
| 2025-10 | Jul–Sep | Oct | Development |

For **each candidate hyperparameter configuration and feature view**, fit a new estimator to each fold's three-month X_train/y_train. Predict probabilities for the subsequent validation month. Never carry fitted estimator weights or a fitted preprocessor across these folds. April–October labels are permissible for development scoring and parameter selection. Each fold uses the frozen estimator and frozen preprocessing throughout its scored month.

### 2.2 Final held-out months (primary results)

1. Finish and lock all model/view hyperparameters, feature encoding policy, scoring rules, and plotting/aggregation specifications based on April–October only.
2. Construct final train set by taking the **August, September and October** training source rows, with the identical eligibility, approved feature view, and exact IDs from Phase I/II. Reuse the existing August–October training membership from the **November** fold, but do **not** use its generic build_fold output as the *sole* holdout loader.
3. Fit one new FoldTransformer (or compatible Phase III train-only wrapper) on August–October data. Transform August–October once; transform November **and** December with the same fitted object, including its fixed categorical vocabulary, medians and scalers. Validate returned data dimensions, class ordering and ID alignment.
4. Fit one new estimator on August–October labeled data using the chosen hyperparameters; store its immutable fit signature and training cutoff.
5. Evaluate that identical fitted estimator first on November and then December (the ordering does not affect fitted state). The current December Phase I fold is September–November→December and **must not** be used for this primary frozen holdout; it would train on November labels.
6. Export **November**, **December** and **pooled November–December** scores. November is a 1-month-ahead month and December a 2-month-ahead month relative to the training cutoff. The pooled metrics must be recomputed from all predictions; they are not the arithmetic mean of monthly scores.
7. November and December targets are unavailable to estimator fitting, tuning, imputation, encoding, scaling, calibration, early stopping, feature selection, or any choice of the chosen model. Later flight rows' precomputed as-of G3 histories may legitimately contain *earlier observed November outcomes* as operational inputs. Distinguish this from label leakage and call it out in the manifest and report.

**Important:** April–October results for the ultimately selected hyperparameters are useful *development/backtest* results. Since they were used to select the configuration, do not describe them as independent or unbiased test results. **November/December are the decisive held-out model evaluation.** Do not tune further after examining holdout scores; a follow-up experiment would need a fresh held-out period.

## 3. Repository buildout

Implement the user's proposed layout (reuse existing package conventions and avoid unnecessary new layers):

    src/flight_delay/
        models/
            __init__.py
            baseline.py
            logistic.py
            random_forest.py
            xgboost.py
            registry.py
        experiments/
            __init__.py
            runner.py
            evaluation.py
            tuning.py
            results.py
        cli.py                   # extend the existing CLI only
        features/              # retain validated Phase II code and contracts
    configs/
        models/
            baseline.yaml
            logistic.yaml
            random_forest.yaml
            xgboost.yaml
        phase3.yaml             # shared protocol, compute and artifact settings
    notebooks/
        03_classical_models_analysis.ipynb
    data/
        model_outputs/         # ALL content ignored by Git
    reports/
        phase3/
            metrics_summary.csv
            experiment_manifest.json
            selected_configs.json
            PHASE3_VALIDATION.md
    tests/
        test_phase3_contracts.py
        test_phase3_tuning.py
        test_phase3_holdout.py
        test_phase3_evaluation.py

Dependencies: keep project Python >=3.11 and existing pinned SciPy/NumPy/scikit-learn versions. Add and **pin** a compatible tested XGBoost version as a declared optional modeling dependency or a dedicated Phase III extra. Add notebook plotting dependencies only if needed, also version-pinned. Use sparse CSR for scikit-learn estimators and compatible XGBoost input. Ensure fresh-install instructions in README. Do not accidentally turn each model into a heavyweight framework.

### 3.1 Models / registry

- baseline.py: train-window empirical **three-class priors**, equivalent to sklearn DummyClassifier(strategy="prior"); always emit probabilities in [0,1,2] order. Apply additive smoothing only if separately specified and consistently reported. Default is unsmoothed training priors. This is feature-independent and requires **one** fitted baseline per development fold, not four redundant fits. For final, fit August–October priors once and use them for both months.
- logistic.py: sklearn LogisticRegression as three-class softmax classification; use an appropriate solver compatible with sparse input (default saga), fixed random_state, and inspect convergence warnings explicitly. Use train-fold-only imputation/scaling. Default penalty L2; do not use undocumented class balancing.
- random_forest.py: sklearn RandomForestClassifier, bounded depth/leaves and max_features, deterministic random_state; handle sparse CSR without unnecessary materialization. Limit parallel workers.
- xgboost.py: XGBClassifier(objective="multi:softprob", num_class=3, tree_method="hist", eval_metric="mlogloss"), fixed random_state and thread count; validate predict_proba shape and class mapping. Do not use test-month early stopping.
- registry.py: a typed name→factory mapping. Each factory takes a validated YAML config, creates an **unfitted** estimator, and exposes a uniform fit/predict_proba contract. Validate accepted hyperparameter keys with descriptive errors. Never rely on numeric ordering returned implicitly by a model if a training fold lacks a class; assert/handle classes explicitly.

Every estimator must produce a calibrated-or-uncalibrated probability *vector* (not just a hard label); output prediction is argmax over ordered probabilities. Do not calibrate on Nov/Dec. Any optional calibration must use only development-train-contained historical data and be clearly reported as another fitted step.

### 3.2 Experiment modules / responsibilities

- experiments/runner.py: one configurable runner that loops over model, view, fold, and candidate, calling the existing Phase II fold loader for development. Add a **separate audited final_holdout_loader** or helper in this module or features/transforms.py that fits on Aug–Oct once and transforms Nov and Dec without changing the fit. Guard against arbitrary train/test input month combinations and unexpected duplicate IDs. Avoid reimplementing Phase I joins.
- experiments/tuning.py: enumerate the bounded YAML candidate space; perform causally ordered Apr–Oct evaluation; aggregate fold metrics; select best configuration per model/view using the deterministic rules below. Save all candidate scores and selected configs. No November/December file/label is loaded during this stage.
- experiments/evaluation.py: accept y_true, ordered (N,3) float probabilities and optional metadata. Perform strict probability checks and calculate all prescribed metrics plus uncertainty summaries. Avoid fitting anything here.
- experiments/results.py: append results atomically and idempotently with unique run IDs; save per-flight private predictions, fit metadata, all candidate/selected aggregate scores, final model artifacts and input/provenance digests. Support resumption after interruption without mixing stale or incompatible runs.
- cli.py: wire existing flight-delay entry point to explicit commands for validating inputs, development tuning, frozen holdout, reporting and smoke tests. Do not disrupt existing Phase I/II CLI functions.

### 3.3 Configuration contract

Define all of the following in YAML with a strict schema and no hidden defaults: dataset roots; fixed year 2025; 7 allowed development fold IDs; final training months [8,9,10]; final evaluation months [11,12]; ordered views; label order [0,1,2]; primary metric macro_f1; tie breakers; seed 6740; candidate sets; compute budget/threads; artifact paths; memory guard; version tag; scaling/encoding choices. Record an exact resolved-config copy and hash per run.

Recommended **initial bounded** search spaces (can be smaller only when compute budget requires, with a logged reason):

| Model | Candidate values | Total combinations |
|---|---|---:|
| Prior baseline | none | 1 |
| Multinomial logistic | C in {0.1, 1, 10}; class_weight in {null, balanced}; solver=saga; penalty=L2 | 6 |
| Random forest | n_estimators in {100,200}; max_depth in {12,24}; min_samples_leaf in {2,10}; max_features=sqrt | 8 |
| XGBoost | max_depth in {4,6}; learning_rate in {0.05,0.1}; n_estimators in {200,400}; subsample=0.8; colsample_bytree=0.8 | 8 |

Additional fixed parameters: RandomForest bootstrap=True; logistic max_iter=400 and tol=0.001 (record any nonconvergence); XGBoost tree_method=hist, objective=multi:softprob, num_class=3, eval_metric=mlogloss. Pin and test exact library semantics. If extremely imbalanced severe cases motivate class weighting or sample weighting, treat this as a predeclared candidate, inspect probability calibration, and record weights. Do not use gradient information to change the *hyperparameter* search during fitting.

**Computationally bounded selection:**

1. Stage A: verify pipeline with tiny deterministic synthetic fixtures, then a small chronological subset smoke run; those outputs are *not* study results.
2. Stage B: screen every predefined candidate per model/view on 3 prespecified development folds, April, July and October, using the full eligible train/test data in those folds. Save all screening results. Prefer these folds for spread across seasons; never sample final data to screen configurations.
3. Stage C: take at most the **best 2 candidates per model/view** by mean screening macro-F1 (tie-break mean log loss), and evaluate each on **all seven development folds**. Scores for Apr/Jul/Oct may be reused if provenance/hashes match. Select the best based on 7-fold mean macro-F1, not screening scores alone. Store the seven per-month values and variance.
4. Stage D: create one final fitted estimator per chosen model/view and score it on Nov and Dec as specified. Baseline needs no search; run it once per dev fold and one final fit.

This may still entail substantial CPU runtime on 1.5M-row folds. Use a per-model/view streaming/cache policy; reuse each fold/view sparse matrix across candidates where memory allows; otherwise regenerate on demand, never densify annual one-hot data. Limit outer experiment parallelism to 1 by default, with estimator threads <=4; avoid 4 simultaneous large forests or nested oversubscription. Time operations and log peak RSS. Do not silently downsample the official fitting population; if resources prevent full runs, finish code/tests and report the incomplete runs explicitly. Expose a --resume flag and deterministic checkpointing.

### 3.4 Selection objective

For configuration h on view v/model m, calculate each development month k's macro-F1 across the three classes. Rank by **unweighted arithmetic mean across seven months** (each month counts once). In order of ties within numerical tolerance 1e-6: (1) lower mean multiclass log loss, (2) lower standard deviation of monthly macro-F1, (3) lower median runtime, (4) deterministic lexical config key. Record selected hyperparameters **per model and view**. The held-out Nov/Dec data must never affect this selection.

If one wants to compare feature-group effects fairly, show within-model view comparisons under their individually development-tuned parameters and also optionally a same-hyperparameter sensitivity on the full view. Do not claim a feature group itself is universally causal based on ablation correlations.

## 4. Representation and train-only transformations

Keep exactly the validated Phase II source features and view definitions. Use the train-only FoldTransformer and sparse float32 CSR output. Consistency of preprocessing across models matters more than squeezing incremental gains via test-aware encoding.

**Known issue to document and test:** g1_month is currently one-hot categorical. A three-month training window never contains the following validation month's month category; thus the validation month's g1_month one-hot vector is all-zero under handle_unknown="ignore". This is a deterministic limitation, not label leakage. For the initial locked Phase III run, retain existing Phase II encoding and log unknown-category counts. If implementing a *predeclared* cyclical month encoding alternative, it must derive only sin(2πmonth/12), cos(2πmonth/12) from the existing g1_month source, preserve the 20-source-feature allowlist, be selected using development folds alone, have its own provenance, and **not** replace or mutate the validated Phase II tables. Avoid adding this experiment unless its separate dev-only implementation and compute budget can be validated.

For both development and final: fit one-hot vocab, numeric imputer, scaler and any class weight logic from that training window only. Log unknown category rates, missing-value handling, output dimension, sparse format/dtype and feature names. Never fit encoders globally on all 2025 rows. Never use g3 histories to train weights outside the explicit training date range. Retain row order/ID mapping in all saved predictions.

## 5. Exact evaluation and research outputs

### 5.1 Per-fold and per-holdout metrics

Every model/view/month must produce:

- Cohort size and class 0/1/2 supports; class probabilities in columns p_minimal, p_moderate, p_severe with sum approximately 1 (tolerance 1e-6), all finite and in [0,1].
- Accuracy; macro-F1 (**selection metric**), weighted-F1; per-class precision, recall and F1, including **class-2/severe recall** and precision.
- Multiclass log loss (labels explicitly [0,1,2]); multiclass Brier score as mean of sum of squared errors across the three class probability components (state this normalization); per-class one-vs-rest ROC AUC and macro AUC when each class is present.
- Confusion matrix with exact class ordering [0,1,2]; observed class proportions vs mean predicted probabilities; predicted argmax class proportions (these are different objects); calibration/reliability curves using prespecified bins (e.g. ten equal-width bins, with counts).
- Training time, preprocessing time, inference time, peak RSS estimate, estimator convergence/warnings; number of rows and columns, matrix memory and unknown-category diagnostics.

Metrics on a month with a missing class must be explicitly NA for undefined AUC and consistent zero_division=0 policy for per-class precision/recall. Never hide severe-class performance inside overall accuracy. Compute pooled holdout scores from concatenated November+December predictions, and report both months separately.

### 5.2 Comparative analysis

- Model × feature-view table: baseline and 3 learned models × four feature views; baseline appears once since views cannot affect a constant prior.
- Rolling development trend: Apr–Oct monthly metric lines for each final selected hyperparameter configuration, with selection-optimism caveat.
- Held-out primary: Nov, Dec and pooled metrics, with confusion matrices, macro-F1, severe recall, log loss and calibration plots. Show run-to-run / day-block uncertainty where practical; at minimum counts and caution about correlated flights.
- Group ablation: compare G1 vs G1_G2 (incremental incoming-aircraft information), G1 vs G1_G3 (historical operations), and G1_G2/G1_G3 vs G1_G2_G3 (joint use). Compare on identical target IDs and fixed evaluation months.
- Airport slices: five target-origin airports ATL, DFW, DEN, ORD, LAX, plus overall; give n, class proportions, metrics, mean p by class, empirical severe rate and predicted severe rate. Do not accidentally filter the *training* sample to those airports.
- Feature interpretation: logistic absolute coefficients grouped by encoded source feature and aggregate RF/XGB importances. Report limitations of biased impurity/gain importances; permutation importance only on a deterministic bounded post-holdout analysis subset, never fed back into model selection. Preserve encoded-to-source-column and source-column-to-G1/G2/G3 mapping.
- Optional qualitative examples only as deidentified, aggregate evidence in public artifacts. Do not embed flight-level identifiers/metadata in the committed notebook.

### 5.3 Saved artifact schema

**Private / ignored** under data/model_outputs/phase3/RUN_ID/:

    development/candidate_metrics.parquet
    development/selected_fold_metrics.parquet
    development/selected_predictions/...
    final/frozen_model_M_VIEW.joblib or equivalent
    final/frozen_transformer_VIEW.joblib or equivalent
    final/predictions.parquet
    final/holdout_metrics.parquet
    logs/...

Private prediction rows must contain at least run_id, model, feature_view, evaluation_stage (development or frozen_holdout), scored_month, target_record_id, y_true, p_minimal, p_moderate, p_severe, y_pred, and model_fit_id. A metadata join can later append target origin and prediction timestamps by ID **for analysis only**. Deduplicate on run/model/view/month/ID; fail on missing or extra targets. Predictions are never written to public reports.

**Commit-safe aggregate reports** under reports/phase3/:

- metrics_summary.csv: one row per model/view/stage/scored_month (and pooled holdout rows), with counts and headline metrics, selected hyperparameter ID, fit ID and provenance; avoid storing a fabricated baseline for every view.
- experiment_manifest.json: git commit SHA, source schema/dataset hashes, preprocessing hashes, class order, fold IDs, candidate grids, chosen configs, feature views, exact months, selection rule, random seeds, dependency versions, resource settings, completion status, warnings and explicit as-of-history policy.
- selected_configs.json: selected hyperparameter dictionary per model/view, fold-level development evidence, tie-break decisions and hashes. No holdout scores influence this file.
- PHASE3_VALIDATION.md: test checklist, actual commands run, artifacts found, class/ID/causality checks, resource use, incomplete items and limitations. If full-data runs did not finish, accurately say so.

Write CSVs and JSONs atomically via temp paths and rename; stable column ordering, numeric precision and deterministic sorting. Public summary must never include IDs or raw records.

## 6. Notebook: analysis only

Create notebooks/03_classical_models_analysis.ipynb with a brief protocol description followed by parameter-selection tables, monthly development scores, frozen Nov/Dec primary scores, model comparisons, ablations, class probability calibration, five-airport slices, feature importance and compute summary. All displays must be produced from persisted results; **no estimator training, hyperparameter optimization, feature construction, or raw data mutation in notebook cells**. An optional noncommitted local execution can read private predictions to build *aggregate* plots; clear any private embedded outputs before committing the notebook. Its default execution should run on the repository's aggregate reports or gracefully mark private-only analyses as unavailable. Document how to recreate every panel.

## 7. Acceptance tests / methodology guardrails

Implement unit, integration and synthetic boundary tests. The following tests should be named and individually auditable:

1. Fixed three-class threshold labels and [0,1,2] prediction probability order, including class absent from a tiny synthetic training fixture.
2. Exactly the four approved ordered views and source column counts; no label, target delay, raw timestamp, tail ID, target_record_id, prior arrival outcome or redundant minimal rate in X.
3. One-to-one joins with unique stable IDs; y and X aligned across all views; all compared models predict on identical holdout IDs.
4. Development folds exactly Jan–Mar→Apr through Jul–Sep→Oct; train/test months do not overlap within a fold; no random split.
5. For every candidate, seven development fits (or only screening fits until shortlisted) with **new estimator and transformer instances**; candidate hyperparameters remain constant within its score aggregation.
6. Tuner imports or loads **no November/December target rows**. Use synthetic sentinel values or monkeypatch open/read functions to prove holdout-only labels cannot affect selected config or summaries.
7. Final training IDs exactly August–October; no November or December label/row in the estimator fit or transformer.fit. Same final fitted model object/fit checksum and transformer checksum used for *both* months. Assert final Dec does not call the standard 2025-12 build_fold path.
8. Prediction-time causal G3 audit is preserved; history outcome observation times strictly precede each target's prediction time, regardless of the target month. Document observed earlier test-month outcomes are permitted features, never training labels.
9. Every probability finite, within [0,1], sums to 1 (1e-6), with shape (n,3); argmax corresponds to saved y_pred; cross-check evaluation functions against tiny hand-calculated cases.
10. Unknown categorical levels and missing numeric data in validation/holdout do not refit vocabulary/imputer/scaler; no data-dependent preprocessing state changes after training.
11. Development metric selection/ties deterministic under fixed seed; changing November/December y must not change selected configs; duplicated/stale checkpoint cannot be silently reused.
12. Exact November, December and combined held-out counts; pooled scores recomputed from row-level predictions; correct airport origin grouping and sufficient sample support.
13. Dependency installation, synthetic CLI end-to-end smoke test, pytest, Phase I/II regression tests, saved-results reload, notebook execution on aggregate-only repository checkout, and Git ignore checks all pass.
14. Run manifest hashes match actual source artifact inputs and selected hyperparameters; repeated seeded rerun gives matching predictions within declared floating-point tolerance.

## 8. Incremental execution sequence and Codex completion gate

1. **Inspect and validate** repository and local upstream files; capture existing git status. Confirm Phase II output readiness and ignored-data handling. Stop on missing real data rather than inventing it.
2. **Implement tests and model factories**, plus pinned modeling dependencies. Verify the training/probability API on synthetic three-class data, including class imbalance.
3. **Implement shared fold/final loaders**, especially frozen August–October→Nov/Dec; write leakage and identity tests before launching full model fitting.
4. **Implement tuning, metrics, artifact and CLI plumbing**. Run synthetic pipeline and a small chronological smoke test, instrument time and memory. Confirm public report redaction and checkpoint resume.
5. **Run baseline plus bounded development screening** on Apr/Jul/Oct for all model/view candidates, then complete all 7 development folds for at most two finalists per model/view; select best settings without accessing held-out labels.
6. **Freeze and record all chosen configurations.** Confirm by logs/assertions that no holdout scores were consulted. Generate complete selected-candidate Apr–Oct backtest metrics.
7. **Run final August–October fit once per model/view; score November and December using that same fit.** Save full private predictions and public aggregate summaries. Do not update any hyperparameters or fit the model between months.
8. **Build and execute notebook on aggregate reports** and optionally enrich local private-analysis plots without committing private output. Review metric interpretations and group ablations.
9. **Re-run pytest and methodological audit**, inspect train/holdout ID hashes and fit signatures, verify report file schemas, run a reproducibility check, and update README/PHASE3_VALIDATION.md. Report actual completion status and any resource limitations.

### Suggested invocations after adding CLI subcommands

    python -m pip install -e '.[dev,ml]'
    pytest -q
    flight-delay features-audit --features-dir data/features
    flight-delay phase3-validate --config configs/phase3.yaml
    flight-delay phase3-tune --config configs/phase3.yaml --resume
    flight-delay phase3-final --config configs/phase3.yaml --selected reports/phase3/selected_configs.json
    flight-delay phase3-report --config configs/phase3.yaml

These are **planned** CLI command names, not claims that they already exist; Codex must add/document them and confirm --help.

### Definition of done

- The four models have reproducible, configuration-driven implementations; there is one execution path per model, view and fold.
- All required development and strict final splits execute with checked row IDs, frozen preprocessing and verified lack of holdout leakage.
- Selected configurations come solely from April–October aggregated rolling scores.
- One fit on August–October produces both November and December forecasts for each model/view; no December refit occurs.
- All model results, severe-class performance, class probabilities, group-ablation effects and airport comparisons are available for analysis; the notebook reads results without retraining.
- The requested repository layout, two core public reports, selected configs, tests, dependency pins and Phase III validation report exist and pass review.
- If any full-data run is incomplete, the handoff distinguishes completed coding/tests from missing experimental evidence. Never invent numbers.
