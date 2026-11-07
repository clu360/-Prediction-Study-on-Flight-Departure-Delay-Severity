# Phase II feature engineering

This phase constructs the exact twenty source features authorized in [the specification](phase2_specification.md). It does not train classifiers, tune models, evaluate predictive performance or train an MLP. The existing Phase I cohort, links, labels, folds, raw files and processed artifacts remain unchanged.

## Input reconciliation and storage

Every Phase I Parquet partition is checked against both the repository and local Phase I schema declarations; row counts must reconcile with the successful Phase I audit. The full normalized master schedule is required and its row count must equal the raw ingestion count. Byte-level SHA-256 hashes of every source partition and master shard contribute to the feature input fingerprint. Input and output directories are separate.

Arrow creates hash-verified airport-partitioned caches of the unchanged cohort, valid departure history and full master scheduled departures. This is a one-pass repartition, not a new aircraft-linkage computation. Only required schedule/history columns are cached. Processing each airport bounds the working set. The three public-contract tables are partitioned by target local year/month, with one shard per airport; hive year/month directory fields are not extra predictors.

The writer keeps the 352 airport partitions open with bounded small row groups, avoiding repeated tiny-file creation. `features --reuse-cache <prior-feature-directory>/cache` can import a completed cache into a fresh output directory only when its source-byte fingerprint matches; all imported file hashes are rechecked. This cache contains source repartitions and prefix counts, not fitted encoders/scalers. Native Arrow buffers are released before feature computation. Superseded local development attempts are retained under clearly named ignored `data/phase2-*` folders; `data/features/` is the final handoff path.

The execution flag source_cache_reused records explicit import via --reuse-cache; ordinary verified cache hits on a same-directory rerun are also supported and are reported in progress logs. Build-stage peak memory is measured before the independent audit. Matrix smoke summaries record cumulative process peak working set across validation and successive views; their CSR byte counts measure the actual individual returned matrices.

`full_features` contains target_record_id plus exactly twenty g1_/g2_/g3_ columns. `targets` contains target_record_id, target_class and signed target_dep_delay_min. `metadata` preserves dates, identity, UTC prediction and scheduled departure times, Phase I quality/retrospective flags and history support diagnostics. Never infer X by dropping labels from a joined table.

## Scheduled and incoming information

G1 reuses carrier, origin, destination, weekday, month, scheduled duration and distance, while expanding the Phase I local departure hour into sin/cos coordinates. Polars weekday is ISO Monday=1 through Sunday=7. No annual categorical vocabulary is fitted.

G2 reuses the five incoming-aircraft values exactly, including signed early-departure delays. No predecessor search, actual elapsed duration, arrival outcome or target outcome is used. The incoming origin is categorical; the other four G2 fields are numeric.

## Causally updated G3

The global history and each airport/cell history are sorted by timezone-aware UTC observed gate-departure time. Prefix class counts and signed delay sums support binary searches at the two window boundaries. `searchsorted(..., side='left')` at both t−window and t implements the inclusive-left, strict-right interval [t−window,t). This is equivalent to maintaining chronological rolling counters and evaluating queries before equal-time events; it avoids scanning historical records separately for each target. [NumPy boundary semantics](https://numpy.org/doc/2.2/reference/generated/numpy.searchsorted.html).

The 28-day global counts provide p_mod=M_global/N_global and p_sev=S_global/N_global. Airport rates are (M_a+20*p_mod)/(N_a+20) and (S_a+20*p_sev)/(N_a+20). The airline-at-origin severe rate is (S_ac+20*airport_severe)/(N_ac+20). The mean is the signed airport delay sum divided by N_a. No global events means null rates; no airport events means global-rate fallback and null mean; no carrier/airport events means airport-rate fallback and count=0. Missing-history rows remain in every view. Minimal rate is not a feature.

Scheduled volume independently counts master schedule records from the target origin in [t−7 days,t). No cancellation, diversion or completion-status filter is applied. Observed departure volume is never substituted for scheduled volume. History uses valid observations including departures of flights that later diverted; future diversion status is not a predictor or eligibility criterion added by Phase II.

All training histories use their own historical prediction times. April's earlier observed departures can influence later April rows, with model parameters and training-fitted transformations frozen. The same causal rule applies to November/December holdouts. January is left-censored by the absence of December 2024 local-source records; no future data or fabricated buffer is used. The diagnostic history_available_days measures elapsed time since the earliest available nationwide observation, capped at 28 days; it is not a predictor and does not claim continuous airport-specific data coverage.

One retained Phase I anomaly has its own outcome observed before its nominal forecast. Strict time filtering alone would expose that target's label. Phase II therefore explicitly subtracts the target's own event from global, airport and airline/airport counts and sums whenever it falls in the window. This reconciles the mandatory own-label leakage test with the requirement to retain all Phase I samples. The own_outcome_excluded metadata flag quantifies this safeguard. The other retrospective flags remain metadata; the cohort is not silently filtered.

## Fold-aware numerical interfaces

`build_fold(fold_id, feature_set, encoding='onehot')` returns X_train, X_test, y_train, y_test, train_ids, test_ids and fitted_transformer. Membership comes directly from the existing Phase I rolling-fold manifest; there is no random splitting. All IDs/labels are joined one-to-one and sorted deterministically. Class order remains [0,1,2].

Categorical training vocabularies are fitted separately for each view and fold. Null categories become an explicit __MISSING__ token; a token absent from training is handled like any other unknown category. Test-only categories map to all-zero blocks through OneHotEncoder(handle_unknown='ignore'). [Encoder documentation](https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.OneHotEncoder.html). In particular, April's month category is unknown in a January–March vocabulary; this expected behavior is recorded in smoke summaries rather than adding future categories to training.

Numeric features use training-only median imputation and optional training-only StandardScaler. An entirely null training numeric column is retained with the imputer's zero fallback, not dropped. [Imputer documentation](https://scikit-learn.org/1.7/modules/generated/sklearn.impute.SimpleImputer.html). Encoders/imputers/scalers are fixed while transforming test rows. Source features remain Float64 where applicable; on-demand matrices are Float32 sparse CSR. Batches limit temporary object-category arrays. There is no annual dense dummy matrix and no classifier training. Tree interfaces may disable scaling via --no-scale without changing source information.

## Validation and limitations

Synthetic tests cover manually calculated smoothing, empty histories, exact boundaries, equal-time determinism, later events, own-label exclusion, signed means, diversion observations, scheduled cancellations, feature allowlists, train-only vocabularies/scaling and all rolling-fold roles. Full artifact audits check every shard, schema, logical SHA-256, all IDs and labels against Phase I, every G1/G2 value, finite numeric values, rate bounds, support counts, history cutoffs, missing-history fallbacks and unchanged fold bytes. Direct boolean-window reductions independently verify G3 on one sample per month at each comparison airport; the retained own-outcome anomaly is checked separately. A complete repeated build must yield the same sorted logical hashes and counts.

Full-data April smoke tests construct all four sparse matrix configurations and record dimensions, ID hashes, unknown-category counts, class counts, sparse bytes and peak process memory. No model score is produced. Aggregate reports contain distributions and coverage; row-level feature samples, caches and numerical matrices remain local and ignored by Git.

Floating probability comparisons allow 1e-12 rounding tolerance, including the mathematically equal zero-cell fallback. This handles multiply/divide rounding without changing the alpha=20 formulas or any source feature values.

Operational limitations inherited from Phase I include retrospective tail assignment, unreported movements, missing TaxiOut, and fourteen retrospective rotation contradictions. Calendar truncation and sparse cells affect early histories. The all-zero representation of unseen month/categories, sparse-history smoothing, and a later sensitivity policy for flagged rotations require team review before Phase III. No tuning is performed here.
