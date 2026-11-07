# Phase 1 validation handoff

The complete final pipeline was executed on all twelve supplied 2025 monthly datasets. All source-byte hashes were rechecked after processing. These are Phase 1 preprocessing results, not model-performance findings.

## Record flow

| Artifact | Rows |
|---|---:|
| `raw` | 7,001,619 |
| `flights_clean` | 6,879,485 |
| `flight_connections` | 6,879,485 |
| `modeling_cohort` | 6,461,609 |
| `history_departure_events` | 6,898,735 |
| `folds` | 19,697,112 |

Valid linkage coverage: **93.925766%** of eligible labeled targets.

## Class distributions

| Class | All eligible targets | Share | Linked primary cohort | Share |
|---|---:|---:|---:|---:|
| 0 | 5,387,042 | 78.305891% | 5,199,881 | 80.473470% |
| 1 | 940,795 | 13.675370% | 883,015 | 13.665559% |
| 2 | 551,648 | 8.018740% | 378,713 | 5.860971% |

The change in severity shares after linkage filtering must be acknowledged when defining the Phase 2 study population.

## Target and link exclusions

Target-level reasons are mutually exclusive in the documented priority order.

| Target status | Count |
|---|---:|
| cancelled | 102,876 |
| diverted | 19,258 |
| eligible | 6,879,485 |

Every eligible target has one primary link reason. Reasons absent from the following table have zero occurrences.

| Link reason | Count |
|---|---:|
| airport_mismatch | 113,768 |
| ambiguous_rotation | 3,038 |
| gap_over_24h | 94,058 |
| intervening_cancelled_or_diverted | 76,619 |
| invalid_prior_schedule | 122 |
| late_prior_takeoff | 62,973 |
| negative_scheduled_turnaround | 61,162 |
| no_previous_record | 6,086 |
| predecessor_missing_takeoff | 50 |
| valid | 6,461,609 |

12-hour sensitivity cohort: **6,157,416** rows (same links, stricter turnaround maximum).

## Full source inventory

All files use the same source header list, available in `input_manifest.json`. Full SHA-256 values below refer to the original unmodified bytes. No exact duplicate raw records were found.

| File | Rows | Dates | SHA-256 |
|---|---:|---|---|
| January.csv | 539,747 | 2025-01-01 to 2025-01-31 | `bce4a1a67ec97d7e65546411d051314ef2c50548dcf9d6d350b479c53bb56df8` |
| February.csv | 504,884 | 2025-02-01 to 2025-02-28 | `5666f6fa2d5dfc24eaa654560e2b1098e6cd085cf7f2a972cbaff1b7a4f7746e` |
| March.csv | 600,872 | 2025-03-01 to 2025-03-31 | `0ca2a12730641e97827da7931f6deaab73d34a04589ec3d99e3b3e8c542a0e89` |
| April.csv | 583,950 | 2025-04-01 to 2025-04-30 | `f376457a5428fcf9d7ba3a2c559d63c041373ff880678b7394ca003e2d389dbf` |
| May.csv | 605,648 | 2025-05-01 to 2025-05-31 | `5f87ef9a5963ec0301ff6e43412c838ec8679bef1a3c7557546800997f8e0f5f` |
| June.csv | 611,575 | 2025-06-01 to 2025-06-30 | `a5f1ba0c83a5623e6f2fe51245843be423a375d0a4f0547de29f612e7a1136ae` |
| July.csv | 631,428 | 2025-07-01 to 2025-07-31 | `a56ad38adfb6c7896a36ae9916e5b23b652cf7874b6aa6946a95b0b20eef195e` |
| August.csv | 602,378 | 2025-08-01 to 2025-08-31 | `6c45f0d3e9acc4bf3defa7779a00f582b4edf40d26097d8014b467a8ce21693a` |
| September.csv | 562,439 | 2025-09-01 to 2025-09-30 | `53f49bffbbf4292c651ff5082423951632d45fc55a3770d4393f783430c35769` |
| October.csv | 605,844 | 2025-10-01 to 2025-10-31 | `7042da6c5e68b5905c9a8b10f004c3073edc65e8e5555a463d02c0c61af0ac93` |
| November.csv | 570,550 | 2025-11-01 to 2025-11-30 | `4ab02ccd510182b9f0f6ed9b102f899bbcab76bcdb23cc768e1ab5effcd76179` |
| December.csv | 582,304 | 2025-12-01 to 2025-12-31 | `489a0895c105e2ae7f26846f8ae2567a13e78af6b3dbb37b1a50ae53261bbc68` |

## Airport subsets by target local month

The following are primary linked cohort counts. Every airport/month has data. Class counts before and after filtering are in `five_airport_monthly_class_counts.csv`; airport, month and carrier coverage tables include denominators.

| Month | ATL | DFW | DEN | ORD | LAX |
|---|---:|---:|---:|---:|---:|
| 2025-01 | 20,612 | 21,190 | 22,545 | 18,893 | 13,614 |
| 2025-02 | 21,125 | 20,600 | 21,168 | 17,802 | 12,651 |
| 2025-03 | 24,727 | 23,831 | 24,898 | 21,943 | 14,947 |
| 2025-04 | 23,758 | 22,521 | 23,445 | 23,853 | 14,683 |
| 2025-05 | 25,596 | 23,672 | 24,804 | 25,728 | 15,535 |
| 2025-06 | 24,866 | 24,276 | 25,566 | 26,344 | 15,290 |
| 2025-07 | 26,090 | 24,707 | 26,250 | 26,089 | 15,503 |
| 2025-08 | 25,477 | 23,421 | 25,840 | 26,383 | 15,302 |
| 2025-09 | 24,771 | 23,030 | 25,212 | 26,387 | 14,340 |
| 2025-10 | 26,233 | 24,837 | 26,416 | 28,600 | 15,091 |
| 2025-11 | 23,529 | 21,860 | 23,427 | 24,364 | 13,928 |
| 2025-12 | 23,124 | 23,184 | 24,036 | 23,434 | 14,556 |

## Exact output schemas and location

Local official outputs are under `data/processed/`, with `year=2025/month=MM/part-BB.parquet`. All UTC timestamps are timezone-aware microsecond timestamps. `schema.json` gives every column and exact dtype. The full [data dictionary](../docs/data_dictionary.md) gives provenance, transformations, availability, allowed groups and missingness policies.

- `flight_connections`: 22 stored columns.
- `flights_clean`: 22 stored columns.
- `folds`: 7 stored columns.
- `history_departure_events`: 13 stored columns.
- `modeling_cohort`: 26 stored columns.

Raw sources remain in their original external folder. Normalized master events and hash-verified caches live under ignored `data/interim/`. Local audits, example records and timestamp anomaly details live under `data/processed/audit/`; these flight-level records are not exported to Git.

## Data quality and validation

- Observed airport codes: **352**.
- Unresolved airport timezones: **0**.
- Exact raw duplicate rows: **0**.
- Source records sharing a conflicting purported-service key: **0**.
- Invalid cancellation/diversion flags: **0**.
- Raw missing tail IDs: **12,501**.
- Fall-back ambiguous schedules resolved by independent schedule checks: **4**.
- Unresolved scheduled departure timestamps: **0**.
- Scheduled arrival clock mismatches: **201**.
- Gate departure clock mismatches: **10**.
- Missing distances: **0**.
- Missing/nonpositive scheduled durations: **7**.
- Actual takeoff date crossings: **67,741**.
- Cross-zone scheduled routes: **3,729,537**.
- Taxi-out intervals above 180 minutes (flag only): **21**.
- Scheduled durations over 720 minutes (flag only): **4**, maximum **1510** minutes (3 canceled and 1 operated records). These values are retained with documented schedule/timestamp flags, not removed by the duration cutoff.
- Maximum resolved taxi-out interval: **379** minutes; signed departure delays range from **-115** to **4352** minutes.
- Primary takeoff source `parsed`: 6,461,609 (100.000000%).

`TaxiOut` is absent in every source file. Takeoff clocks use the documented conservative WheelsOff calendar resolution. Gate-clock inconsistencies invalidate takeoff and exclude history observations; no arbitrary one-hour DST repair is applied. All eligible target labels remain retained. Detailed per-column invalid/null counts and per-month diagnostics are in `normalization_quality.json`.

- audit_prior_arrival_after_target_departure: **14** valid links flagged.
- audit_prior_arrived_before_prediction: **0** valid links flagged.
- audit_target_departed_before_prediction: **1** valid links flagged.

Targeted review found 14 flagged connections with prior arrival overlapping target actual departure by 1–150 minutes. The destination-local arrival clock independently agrees with the elapsed-time-derived arrival in 13 cases and conflicts in 1; every flagged target's gate clock agrees. These are unresolved record/rotation contradictions, not silently repaired timestamps. Aggregate evidence is `retrospective_anomaly_review_summary.json`.

Retrospective flags do not silently filter the primary schedule-selected cohort. Local `retrospective_anomalies.csv` retains every flagged connection for review. The balanced 100-example sample covers all twelve months, cross-zone, day-crossing and unusual taxi timings; examples are available locally, not in the public repository.

Automated artifact acceptance: **True** (21 checks). Every Parquet partition was reread, schema/UTC/logical hashes checked, and immediate same-tail predecessor relationships reconstructed from master events. All 38 synthetic tests passed; the transcript is `test_results.txt`. The independently repeated full-data run is recorded in `reproducibility.json`. A second engine, PyArrow, fully decoded and validated every record batch in all five tables; `arrow_validation.json` records matching counts and schemas.

The full uncached preprocessing/write stage took 436.84 seconds. Its measured process peak working set was 2221.96 MiB. This stage measurement excludes the subsequent independent read-back audit. The cached full rerun, including its complete artifact audit, took 439.62 seconds. All 1,920 logical partition hashes and all record counts were identical on rerun.

## Decisions before Phase 2

1. Review the remaining retrospective forecast-window/physical-rotation contradictions. Retain audit flags and choose an explicitly documented sensitivity policy; do not silently use future outcomes to redefine feature availability.
2. Accept the takeoff date-resolution limitation without TaxiOut, or obtain an approved official export containing TaxiOut for an independently tracked enhancement.
3. Acknowledge retrospective tail assignments, unreported ferry movements, conservative rotation exclusions, absent December 2024 predecessors, and the linkage-dependent shift in severity shares.
4. Choose and freeze Phase 2's online-updated versus month-boundary-frozen history policy and sparse-history smoothing using development data. Preserve Nov–Dec as final holdouts and use explicit predictor allowlists.

No models, model feature matrices, MLP training, final project report, or performance claims were produced. This handoff documents computational validation and targeted review; it does not claim certification by an external human reviewer.
