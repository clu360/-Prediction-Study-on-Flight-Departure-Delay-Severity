# Codex Phase II: Feature Engineering and Modeling Matrix

Project: ISyE 6740 — A Prediction Study on Flight Departure Delay Severity

Repository: https://github.com/clu360/-Prediction-Study-on-Flight-Departure-Delay-Severity.git

Team: Caleb Lu, Ho Young Lee, Christopher Yi

## Authorization and deliverable

The team **authorizes Phase II only**. Implement three feature groups and their sample-by-feature table on top of the validated Phase I artifacts. Do not retrain the aircraft-linkage pipeline, implement classifiers, train the MLP, tune models, or write final results. Update the existing AGENTS.md from Phase-I-only to Phase-II-permitted without weakening safety and leakage guardrails.

Output a reproducible **6,461,609-row by 20-source-feature table**, plus labels, group manifests, and a fold-aware numerical matrix loader. The 20 features are grouped as 9 G1, 5 G2, and 6 G3. Keep one row per existing target_record_id and no extra or missing target IDs. The Phase I number is a reconciliation target, not a quantity to hard-code.

Locked methodology:

1. Prediction time is one minute after the linked prior aircraft takes off, before its arrival and before the target flight's scheduled departure.
2. G1 and G2 inherit precisely the eligible cohort, source values, and aircraft linkages already built by Phase I.
3. G3 historical statistics update causally as earlier departures are observed. For a target at time t, every historical outcome included in its features must be observed at a time strictly less than t.
4. For the initial specification, use 28-day lookbacks for historical delay rates and mean delay, 7-day lookbacks for prior scheduled departure volume, alpha=20 additive smoothing, and no minimal-delay-rate predictor (it is redundant with moderate and severe rates).
5. All four feature configurations operate on identical samples: G1, G1+G2, G1+G3, and G1+G2+G3.
6. The model fitted on the previous three months remains fixed during the following test month. Its training-fitted encoders/scalers also remain fixed. G3 feature values may change during the test month as earlier observed departures accrue.
7. November and December remain the final holdout months. No model training or test-month model selection is authorized by this Phase II task.

## 1. Read the current Phase I contracts and artifacts

Review the current repository's AGENTS.md, README.md, src/flight_delay/outputs.py, docs/data_dictionary.md, docs/phase1_specification.md, reports/schema.json and reports/VALIDATION.md. The last verified Phase I outputs were:

- data/processed/modeling_cohort — 6,461,609 linked eligible target rows; contains target schedule and prior flight data.
- data/processed/flights_clean — 6,879,485 completed labeled target rows.
- data/processed/history_departure_events — 6,898,735 observed departures with origin, reporting_airline, observed_departure_time_utc, dep_delay, departure_severity and valid_observation.
- data/processed/folds — nine rolling three-month-training, next-month-testing manifests from April to December.
- data/interim/[run-signature] — normalized master scheduled flight records including cancellations and diversions.

All raw/processed flight records are local and excluded from public Git. If these artifacts are missing in the Codex environment, implement and test with synthetic fixtures but do not claim the full output was built. If a local Phase I schema disagrees with the repository's current schema.json, stop and report the discrepancy; do not silently fabricate columns or recalculate the predecessor links.

## 2. Exact group definitions: 20 pre-encoding feature columns

### G1: Scheduled flight characteristics — 9 columns

| New feature column | Existing Phase I source | Meaning or transformation |
|---|---|---|
| g1_carrier | target_carrier | Reporting airline, categorical |
| g1_origin | target_origin | Origin IATA code, categorical |
| g1_dest | target_dest | Destination IATA code, categorical |
| g1_dep_hour_sin | target_crs_dep_local_hour | sin(2*pi*hour/24), float |
| g1_dep_hour_cos | target_crs_dep_local_hour | cos(2*pi*hour/24), float |
| g1_day_of_week | target_day_of_week | Weekday, categorical integer; verify existing Polars convention |
| g1_month | target_month | Month, categorical integer 1–12 |
| g1_crs_elapsed_min | target_crs_elapsed_min | Scheduled duration in minutes |
| g1_distance_miles | target_distance_miles | Scheduled distance in miles |

These use eight existing Phase I source fields, with hour expanded to two cyclical coordinates. Keep the original integer hour in metadata if desired but not as an extra default model feature. Do not one-hot encode these categories globally across 2025.

### G2: Incoming aircraft information — 5 columns

| New feature column | Existing Phase I source | Meaning |
|---|---|---|
| g2_prior_origin | prior_origin | Previous leg's origin, categorical |
| g2_prior_dep_delay_min | prior_dep_delay_min | Previous leg's **signed** departure delay |
| g2_prior_crs_elapsed_min | prior_crs_elapsed_min | Previous leg's scheduled duration |
| g2_scheduled_turnaround_min | scheduled_turnaround_min | Target scheduled departure UTC minus previous scheduled arrival UTC |
| g2_minutes_until_crs_dep_at_prediction | minutes_until_target_crs_dep_at_prediction | Target scheduled departure UTC minus prediction instant |

Reuse these exact Phase I values. Do not substitute prior actual flight duration, prior arrival delay, or prior WheelsOn. Do not include Tail_Number, prior_record_id or takeoff timestamp as direct predictors.

### G3: Historical airport/airline operations — 6 newly constructed columns

For each target row, define:

- t = its prediction_time_utc.
- a = its target_origin.
- c = its target_carrier.
- W28(t) = historical event times in [t minus 28 days, t).
- W7(t) = historical schedule times in [t minus 7 days, t).
- Eligible historical outcome events are from Phase I history_departure_events, valid_observation=True, using observed_departure_time_utc strictly less than t. These include valid departure observations of flights that later diverted; do not select events based on a future diversion outcome.
- N_a = number of eligible events in W28(t) from airport a; M_a = number among them of class 1; S_a = number among them of class 2.
- N_ac = number of eligible events in W28(t) from airport a and airline c; S_ac = severe events in this cell.
- Global probabilities p_global_mod(t) and p_global_sev(t) use all earlier eligible W28 events nationwide, not the whole training or test month.

With alpha=20, construct:

    airport_moderate = (M_a + 20*p_global_mod) / (N_a + 20)
    airport_severe   = (S_a + 20*p_global_sev) / (N_a + 20)
    airline_severe   = (S_ac + 20*airport_severe) / (N_ac + 20)

If global W28 has no events, return null rates with an explicit no_global_history audit flag, not rates estimated from future records. If N_a is zero but global history exists, smoothed airport rates equal the global rates. If N_ac is zero, the airline-at-origin rate equals the as-of airport severe rate.

| New feature column | Exact definition | Window |
|---|---|---|
| g3_airport_moderate_rate_28d | Smoothed airport_moderate above | 28 days |
| g3_airport_severe_rate_28d | Smoothed airport_severe above | 28 days |
| g3_airport_mean_dep_delay_28d | Mean signed historical dep_delay at origin a; null if no observations | 28 days |
| g3_airport_scheduled_volume_7d | Count of scheduled records in normalized master table with Origin=a and scheduled_departure_utc in W7(t), **including eventually canceled/diverted flights** | 7 days |
| g3_airline_airport_severe_rate_28d | Smoothed airline_severe above | 28 days |
| g3_airline_airport_count_28d | N_ac, nonnegative integer | 28 days |

The scheduled-volume feature MUST use master **schedule** records, not simply the observed departure-event table. The latter would measure completed/observed volume, a different construct. If the master schedule is unavailable, stop for a choice instead of changing the feature silently.

Minimal historical delay rate is **not** an X column. By definition minimal = 1 - moderate - severe. A minimal rate may be computed only for temporary auditing or explanatory plots and must not enter the explicit feature-group manifest.

There are no other initial G3 features: no weather, maintenance, route target encoding, current departure outcomes, or actual arrival information.

## 3. Causal history computation and expected performance

Build G3 by streaming chronological events, not by iterating all historical rows independently for each of the 6.46 million targets:

1. Sort target queries by UTC prediction time with target_record_id tie-breaking.
2. Sort valid historical outcome events by observed_departure_time_utc. Maintain rolling 28-day counts and signed-delay sums for global, per-origin, and per-(origin, airline) keys.
3. Independently sort normalized master scheduled departure times and maintain a rolling 7-day per-origin schedule count.
4. Before each target query at t, add only events with event time < t, evict events older than the left edge, compute counts/rates/sums, and apply smoothing. At an identical timestamp, calculate query features **before** adding the equal-time event.
5. Preserve features for training rows using each training flight's actual prediction-time cutoff. Never group by a full calendar month or leak later outcomes backward.
6. Earlier April outcomes may update April 20's G3 features, but they cannot alter April 10's. This is the agreed causal online updating protocol; model parameters are NOT updated within April.
7. No December 2024 source is available. January history is left-censored; do not backfill using later 2025 events. For no origin history, preserve count=0 and the stated smoothing/fallback. For no mean-delay observations, output null rather than fake zero.
8. Retain all target rows, including those with missing early history. Training-fold-local imputation is handled later.

Use Polars lazy queries, Arrow, streaming keyed deques, or correctly implemented range/as-of windows. Benchmark throughput and memory. Avoid a huge dense annual one-hot matrix. Preserve deterministic sorting, schema and output hashes.

Audit-only counts (global N28, airport N28, history available days, no-history flag) can be stored in metadata/quality reports but must **not** silently expand the 20-feature experiment.

## 4. Sample × feature matrix design

Output four distinct concepts and keep them separate:

1. **Features:** One pre-encoded mixed-type table with one row per target_record_id and exactly 20 model features, visibly prefixed g1_, g2_, g3_.
2. **Labels:** target_record_id + target_class (0/1/2). Target_dep_delay_min may be retained for analysis but never enters X.
3. **Metadata:** target_record_id, flight_date, source_month, target_origin, target_carrier, prediction_time_utc, target_crs_dep_utc, quality flags needed for joining/temporal audits. IDs, timestamps and audit outcomes are NOT predictors.
4. **Fold-aware numeric matrix builder:** A callable function or CLI that fits encoders/scalers on the three training months and then transforms the following test month, returning X_train, X_test, y_train, y_test, IDs and the fitted transformer. These are generated on demand, not fitted once on all 2025 data.

Suggested local output tree:

    data/features/full_features/
    data/features/targets/
    data/features/metadata/
    data/features/audit/phase2_summary.json
    data/features/audit/feature_schema.json
    data/features/audit/history_coverage_by_month.csv
    data/features/audit/history_coverage_by_airport.csv
    data/features/audit/feature_missingness.csv
    data/features/audit/feature_samples.csv
    docs/phase2_feature_dictionary.md
    docs/phase2_methodology.md
    reports/PHASE2_VALIDATION.md
    configs/feature_groups.yaml
    configs/phase2.yaml

The features, labels and metadata tables must join one-to-one by target_record_id. Partition the full-data Parquet outputs by target local year/month in a manner compatible with Phase I, and keep the full flight-level data outside Git. Public reports must contain aggregates only.

**Exactly four default feature views**, whose pre-encoding column counts are:

| View | Include | Column count |
|---|---|---:|
| G1 | All g1_ columns | 9 |
| G1_G2 | All g1_ and g2_ columns | 14 |
| G1_G3 | All g1_ and g3_ columns | 15 |
| G1_G2_G3 | All three prefixes | 20 |

Use a checked explicit ordered allowlist in configs/feature_groups.yaml; do not use all non-label columns automatically. The feature matrix must not contain raw tail number, target outcome, previous arrival outcome, record ID, prediction timestamp, source month, or audit flags.

### Per-fold numerical transformations

The full_features table is a **mixed-type** sample × feature matrix, not yet fully numerical; this is deliberate so each model can use appropriate categorical encoding while seeing identical source information.

Provide an API of the form:

    build_fold(fold_id, feature_set, encoding='onehot')
      -> X_train, X_test, y_train, y_test, train_ids, test_ids, fitted_transformer

Use data/processed/folds as the source of training/testing membership; never random-split. For the April fold fit all transformations using Jan–Mar rows only:

- Treat g1_carrier, g1_origin, g1_dest, g1_day_of_week, g1_month, g2_prior_origin (when included) as categorical; one-hot with unknown-category support fit on train.
- Numeric columns include cyclical hour, scheduled duration/distance, prior numeric features, historical rates, means, counts and volume. Fit imputation and StandardScaler on train for logistic/MLP. For tree models, scaling can be optional but no new information may be introduced.
- Use sparse CSR for one-hot matrices; never densify the entire year's airport/carrier dummy expansion.
- Optionally expose a second MLP-ready interface with train-only categorical embedding vocabularies and scaled numeric arrays, including explicit unknown-category indices.
- Retain class order [0,1,2] in all labels and later probability output conventions.
- Training-fitted encoders, imputers and scalers are frozen during the following test month, just like model parameters.
- Do not fit any preprocessing transform on all of 2025 in a way that learns test-month distributions.

Smoke-test at least one development fold (2025-04) on all four views and show shapes, one-to-one target ID alignment, missingness and sparse memory use. No classifier training is required.

## 5. Implementation modules and recommended collaboration

Create a Python package with modules analogous to:

    src/flight_delay/features/schedule.py
    src/flight_delay/features/incoming.py
    src/flight_delay/features/history.py
    src/flight_delay/features/builder.py
    src/flight_delay/features/views.py
    src/flight_delay/features/transforms.py
    src/flight_delay/features/validation.py
    tests/test_features_schedule.py
    tests/test_features_incoming.py
    tests/test_features_history.py
    tests/test_feature_views.py
    tests/test_feature_folds.py

Member workstreams may be one owner each for (G1 + integration), (G2 + predecessor audit), and (G3 + causal aggregation). Each group module should produce one table keyed on target_record_id, allowing builder.py to join one-to-one while preventing duplicate/missing rows.

Keep the existing Phase I code, tests, raw inputs and outputs unchanged. Update README, pyproject dependencies and the project-specific AGENTS.md to reflect explicit Phase II authorization while keeping all original safety rules. Work in a new branch such as feat/phase2-feature-engineering and prepare a PR; do not merge automatically.

## 6. Mandatory validation and anti-leakage tests

### G1 and G2

- Hour 0 maps to sin=0/cos=1; hour 6 to sin=1/cos=0; hour 12 to sin=0/cos=-1; hour 23 neighbors midnight on the unit circle.
- All G1/G2 values other than the documented hour transformation exactly match Phase I's allowlisted source columns, including signed negative prior delays.
- Exactly the specified G1/G2 columns are included; no arrival outcomes, realized target features or raw tail IDs.
- G2 does not change the existing linked cohort; no fresh predecessor search.

### G3 causal behavior

- Construct a tiny synthetic history with manually known rates/counts, verifying the alpha=20 formulas and fallback hierarchy.
- A historical event at t-28 days counts, one at exactly t does not, and one after t does not. Verify the analogous t-7 days boundaries for volume.
- Inject a severe departure event in the future: no earlier flight's G3 features change.
- Inject the target flight's eventual delay: its own features cannot change.
- If a target is predicted later the same day, earlier events may affect it; later events never affect an earlier target.
- Equal-time queries/events are deterministic and obey strict event_time < prediction_time.
- Validate correct grouping by airport and by (airport, airline), nonnegative counts, rates in [0,1], and moderate+severe <=1 plus floating tolerance.
- A flight that later diverted may contribute its earlier valid departure observation, but no retrospective diversion outcome can be used as a feature.
- Count past **scheduled** flights from the master schedule, including cancellations, and verify no future schedule event enters past-volume lookbacks.
- Training feature histories are causal at each historical training observation, rather than retrospectively computed using its entire future month.
- No fabricated December 2024 history; zero-history cases documented.

### Matrix/split integrity

- Same 6,461,609 target IDs in every feature view, no duplicates and no row losses due to missing G3 values.
- Four views have exactly 9/14/15/20 source columns.
- All labels match Phase I's target_class; no target outcome or identity field is in X.
- The existing rolling folds are unchanged. For April, Jan–Mar training transformations never fit on April rows.
- Missing/unknown categories in April are handled with train-fitted rules; no new test-only category vocabulary leaks back into training.
- November and December flags are preserved, and no model evaluation or tuning is performed during this task.
- The output Parquet is readable with schema stability and deterministic sorted logical hashes on rerun.

### Required full-data audit

Publish a table of every feature's dtype, source, formula, availability time, missing count/fraction, distribution or categorical cardinality, and model group. Provide missingness/history coverage overall, by month, and by ATL/DFW/DEN/ORD/LAX. Report mean/median/min/max history counts, fallback frequency, extreme signed delays, rate distribution, numerical-finiteness checks, runtime and peak memory. Explicitly document that early January suffers from absent 2024 history.

Phase I's small set of retrospective rotation contradictions must not silently change the primary sample. Keep those rows and flags in metadata; optionally recommend a later sensitivity analysis excluding them without using future outcomes as features.

## 7. Execution entry points and acceptance report

Implement documented functional CLI commands equivalent to:

    python -m pip install -e '.[dev]'
    pytest -q
    python -m flight_delay.cli features --processed-dir data/processed --out-dir data/features --config configs/phase2.yaml
    python -m flight_delay.cli features-audit --features-dir data/features
    python -m flight_delay.cli feature-fold --fold 2025-04 --feature-set G1_G2_G3 --encoding onehot

CLI names may be adapted to fit the existing package, but list exact usable commands in README. Make builds idempotent, use source/config fingerprints, cache intermediate history safely, and do not overwrite unknown existing output folders destructively.

**Return this Codex handoff when finished:**

1. Git branch/commit/PR link and changed files.
2. Verified source input paths, row/target counts and Phase I schema reconciliation.
3. Exact table schemas and one-to-one ID/label checks.
4. Group manifest with 9+5+6 column names, four ablation configurations and per-fold smoke matrix dimensions.
5. Implementation explanation for 28-day history, 7-day scheduled volume, strict as-of events, alpha=20 smoothing, no-minimal-rate policy and missing history.
6. Feature quality/coverage results, missingness by month/five airports, no-leakage test evidence, tests passed, reproducibility, runtime and memory.
7. Scientific/technical limitations requiring team review before Phase III.

Do not claim full success unless the 2025 Phase I artifacts were available, the full Phase II feature table was constructed, test suites passed, and output audits verified the result. If blocked by missing master schedule or another essential artifact, report the blocker rather than silently substituting a different feature definition.

## 8. Out-of-scope future phases

Phase III implements the historical-probability baseline, multinomial logistic regression, random forest and XGBoost with four ablations and their prediction/feature-importance analysis. Phase IV implements the MLP with the same splits, labels and features, then conducts its analysis. Phase V compares calibrated class probabilities, macro-F1/log loss, severe recall, ablations and five-airport distributions. Phase VI writes the report based on locked evidence. These phases are **not authorized for implementation by this document**.
