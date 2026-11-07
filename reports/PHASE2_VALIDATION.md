# Phase II validation handoff

The complete feature pipeline was executed against the validated 2025 Phase I outputs. No aircraft linkage was recomputed and no classifier or MLP was trained.

## Sample and schema reconciliation

Phase I cohort: **6,461,609** targets. Full master schedule: **7,001,619** events, including cancellations/diversions. History event store: **6,898,735** observations. All Phase I partition schemas match the repository declaration and all counts reconcile.

| Phase II artifact | Rows | Stored columns |
|---|---:|---:|
| full_features | 6,461,609 | 21 |
| metadata | 6,461,609 | 21 |
| targets | 6,461,609 | 3 |

The mixed-type sample-by-feature matrix is N×20; target_record_id is an additional join column. All three artifacts have exactly the same unique IDs, labels match Phase I, and no sample is dropped for missing history. Local output: `data/features/{full_features,targets,metadata}/year=2025/month=MM/part-AIRPORT.parquet`.

## Four feature configurations

| View | Source columns |
|---|---:|
| G1 | 9 |
| G1_G2 | 14 |
| G1_G3 | 15 |
| G1_G2_G3 | 20 |

The exact ordered column lists are `configs/feature_groups.yaml`. G1 has nine columns, G2 five and G3 six. No minimal-delay-rate predictor, raw ID, timestamp, target outcome or prior arrival outcome enters X. Source/formula/availability/dtype details appear in [the dictionary](../docs/phase2_feature_dictionary.md).

## Full April matrix smoke tests

All rows in each three-month training window and following test month were used. Every view has identical train IDs and identical test IDs. Encoders, median imputers and optional scalers fit only January–March and remain frozen for April. Matrices are sparse Float32 CSR; labels remain 0/1/2.

| View | X_train | X_test | Train CSR MiB | Test CSR MiB |
|---|---|---|---:|---:|
| G1 | [1508151, 694] | [547836, 694] | 109.31 | 35.53 |
| G1_G2 | [1508151, 1031] | [547836, 1031] | 166.84 | 56.43 |
| G1_G3 | [1508151, 700] | [547836, 700] | 178.35 | 60.61 |
| G1_G2_G3 | [1508151, 1037] | [547836, 1037] | 235.88 | 81.50 |

Test-only categories, including April's unseen month category, are handled by train-fitted unknown-category rules; counts are recorded per view in `phase2/matrix_smoke.json`. No test distributions were used to fit transformations.

## History policy and coverage

Outcomes use strict 28-day as-of histories, independently of 7-day master scheduled volume. Global-to-airport-to-airline/airport smoothing uses alpha=20. Earlier test-month observations may update later feature values, while transformations and later model parameters remain frozen. Null/zero history fallbacks are retained and audited; no December 2024 data are invented.

- audit_prior_arrival_after_target_departure: **14**.
- audit_prior_arrived_before_prediction: **0**.
- audit_target_departed_before_prediction: **1**.
- no_airline_airport_history: **1,859**.
- no_airport_history: **248**.
- no_global_history: **0**.
- own_outcome_excluded: **1**.

| 28-day history support | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| global_history_count_28d | 512530.1999 | 537797 | 2 | 563,349 |
| airport_history_count_28d | 9541.0492 | 7987 | 0 | 28,313 |
| airline_airport_count_28d | 2935.2507 | 1326 | 0 | 19,841 |

All fourteen retrospective rotation contradictions are retained as metadata. The one already-departed target is explicitly protected against self-outcome leakage; its event is subtracted from its own global/airport/cell history without changing the sample. An independent direct oracle verifies this safeguard. History availability is left-censored in early January.

## Feature distributions and grouped missingness

`phase2/feature_missingness.csv` gives dtype, group, null count/fraction and every numeric feature's mean/median/min/max/std/1st/99th percentiles or categorical cardinality. Grouped missingness is in feature_missingness_by_month.csv and feature_missingness_by_airport.csv. History support/fallback counts and mean/median/min/max airport counts are in history_coverage_by_month.csv and history_coverage_by_airport.csv, including ATL, DFW, DEN, ORD and LAX.

| Feature | Missing | Min / cardinality | Median | Max |
|---|---:|---|---|---|
| g1_carrier | 0 | 14 | — | — |
| g1_origin | 0 | 352 | — | — |
| g1_dest | 0 | 352 | — | — |
| g1_dep_hour_sin | 0 | -1.0 | -0.25881904510252035 | 1.0 |
| g1_dep_hour_cos | 0 | -1.0 | -0.4999999999999998 | 1.0 |
| g1_day_of_week | 0 | 7 | — | — |
| g1_month | 0 | 12 | — | — |
| g1_crs_elapsed_min | 0 | 20.0 | 132.0 | 739.0 |
| g1_distance_miles | 0 | 31.0 | 692.0 | 5095.0 |
| g2_prior_origin | 0 | 352 | — | — |
| g2_prior_dep_delay_min | 0 | -75.0 | -2.0 | 1496.0 |
| g2_prior_crs_elapsed_min | 0 | 20.0 | 132.0 | 717.0 |
| g2_scheduled_turnaround_min | 0 | 0.0 | 65.0 | 1440.0 |
| g2_minutes_until_crs_dep_at_prediction | 0 | 1.0 | 200.0 | 2101.0 |
| g3_airport_moderate_rate_28d | 0 | 0.0 | 0.13234437130629892 | 0.3887810132342145 |
| g3_airport_severe_rate_28d | 0 | 0.0 | 0.07271940999334439 | 0.4977645379244056 |
| g3_airport_mean_dep_delay_28d | 248 | -36.25 | 12.107747187414251 | 431.5 |
| g3_airport_scheduled_volume_7d | 0 | 0 | 2089.0 | 7181 |
| g3_airline_airport_severe_rate_28d | 0 | 0.0 | 0.07018836906823382 | 0.5441960597693785 |
| g3_airline_airport_count_28d | 0 | 0 | 1326.0 | 19841 |

## Complete single-file exports

Alongside the partitioned tables, the requested complete annual feature table is exported as two separate files under `data/features/exports/`: `full_features.csv` and `full_features.parquet`. Each has the same target IDs and exactly twenty predictors plus the ID column. Labels and metadata remain separate to preserve the X/y boundary. Every column/row was checked against the validated feature table, and the Parquet file was independently decoded with Arrow.

- csv: 6,461,609 rows; 1,386,931,893 bytes; SHA-256 `542670df1e33776a8e21c51a549ea41f60e89333bbf8e6cc5170ed631cca5f98`.
- parquet: 6,461,609 rows; 285,994,872 bytes; SHA-256 `dcba1f79f4039e96e6350c3b302722ab6bb12a0ab398c9ceb0b00711c43f5379`.

## Verification and reproducibility

All **19** full-data acceptance checks pass. Every output shard was reread and its schema/UTC/logical hash checked. G1/G2 values and all target IDs/labels match Phase I. Numeric finiteness, rate/simplex bounds, count bounds, fallback hierarchy and strict history cutoffs pass.

The independent boolean-window oracle checked 60 real samples / 360 scalar G3 values. The own-outcome oracle checked 1 additional anomalous samples. Synthetic test results are in `phase2/test_results.txt`.

A complete rerun produced identical counts and all **12,219** canonical sorted logical hashes. Combined logical SHA-256: `509c8ecc9937bdd6f62b543a1885081548581619715ec5c8abb3d0f2a44539c9`. Source byte/configuration/code/dependency fingerprints are recorded. Phase I source fingerprints and fold bytes remain unchanged.

Initial validated source reconciliation/build/write stage: 230.77 seconds; source cache imported: True; peak process working set before the independent audit: 1189.01 MiB. Cached complete rerun including its audit: 469.03 seconds. Matrix summaries record cumulative process peak working set across validation and successive views, not isolated allocations; exact individual CSR sizes appear in the table above. The maximum observed cumulative peak was 5.472 GiB. Full-year dense one-hot expansion was never allocated.

## Review before Phase III

Review a later sensitivity policy for retained retrospective contradictions; acknowledge January left censoring and Phase I tail/takeoff limitations; review unknown-category/all-null-column transformation rules. History regime, windows, smoothing and sample are locked here and were not selected using final holdout model scores. November/December remain final holdouts. No classifier fitting, predictive metrics, tuning, feature-importance analysis or final report conclusions were produced.
