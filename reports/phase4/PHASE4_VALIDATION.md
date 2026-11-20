# Phase IV validation

The real-data model, source, cohort and prediction checks passed for both XGBoost fits and all 12 MLP fits. The notebook execution record is published separately alongside these aggregates.

## Source and cohort

- All three locked Phase II tables contained 6,461,609 aligned target IDs. The 12 monthly populations, source fingerprints and verified partition counts are recorded in the [upstream audit](upstream_verification.json).
- All 352 station requests were verified against exact IEM response bytes for December 30, 2024 through January 2, 2026 UTC, with an exclusive end. The 3,895,897 archived rows include 36,527 undated corrections, 605 out-of-scope high-frequency rows and one malformed relay row excluded from the causal stream. Real sample checks confirmed PBI–DJT, HDN–HDN, ADK–PADK, BQN–TJBQ and SPN–PGSN. [Source inventory](weather_source_inventory.csv) · [Archive verification](weather_source_verification.json)
- G4 has exactly five numeric predictors across all 12 target months. Source availability is strictly before prediction, origin and destination match Phase I, and the prediction clock equals prior wheels-off plus one minute. All 8,145 direct-oracle comparisons passed. Missingness ranges from 0.169% to 1.526%, with seven zero-coverage airport/month combinations. All flights remained in the study. The separate private audit table contains station keys, report counts, ages and missingness indicators. [Feature verification](weather_g4_verification.json) · [Coverage](weather_coverage.csv) · [Audit-only diagnostics](g4_audit_only_verification.json)

## Training and scoring

- The matched XGBoost fits reuse the byte-verified Phase III selected G1–G3 configuration at seed 6740 and four CPU threads. MLP tuning evaluated all 26 prescribed candidate-fold combinations. Candidate 6 won with mean development log loss 0.480053. The final lock uses hidden widths 256 and 128, dropout 0.3, learning rate 0.0003, one epoch and seeds 6740, 6741 and 6742. The epoch is the median of the seven inner stopping choices [1, 2, 1, 1, 1, 1, 1]. [Selection lock](mlp_selection.json) · [Tuning](mlp_tuning_summary.csv) · [Learning curves](mlp_learning_curves.csv)
- All 14 final fits used 1,663,534 identical training IDs with class counts [1,370,826, 212,248, 80,460]. Weighted arms use training-derived weights [0.404509, 2.612563, 6.891764]. Both frozen months used identical fitted state and transformer hashes per fit. Each fit scored 519,971 November and 532,053 December flights. Verification reread canonical labels, confirmed probability class order [0, 1, 2], row sums and argmax, and recomputed pooled scores from all predictions. [Fit and prediction manifest](experiment_manifest.json) · [Metrics](metrics_summary.csv)
- Replayed 2,048 evenly spaced predictions per fit/month across all 28 scoring slices using saved models. Maximum absolute probability difference was 1.19×10⁻⁷ against a tolerance of 2×10⁻⁶. [Replay evidence](frozen_replay.json)

## Environment and integrity

- Local Windows CPU execution used PyTorch 2.9.1 and XGBoost 3.1.2. The XGBoost fits took 143.70 and 121.33 seconds. MLP final fit times were 17.98 to 34.56 seconds, excluding input preparation and scoring. PACE SSH timed out during read-only preflight, so no Slurm jobs were submitted.
- The full regression suite passed 146 tests with one skip. The later weather-only suite passed 26 tests, including the new per-target diagnostic-window assertion. Windows file-sharing errors were resolved through bounded retry and verified checkpoint reuse. [Chronological journal](phase_4_progress.md)
- Phase I–III scientific files remain as recorded at the branch base. Public Phase IV outputs contain aggregate metrics, plots derived from aggregate data, and hashes. Stored flight-level records, source CSVs and fits remain under ignored data paths.
- All new weather and weighting tests on the previously examined holdout months are exploratory. Reliability bins and severe average precision describe the saved uncalibrated probabilities. Inverse-frequency cross-entropy changes the training objective. The fixed one-epoch rule may favor some arms, and the evaluation supports generalization claims within 2025 only.

[Research-question summary](research_question_summary.md) · [Executable notebook](../../notebooks/04_weather_mlp_analysis.ipynb) · [Execution guide](../../docs/phase4_execution.md)
