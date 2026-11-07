# Codex Implementation Brief
## ISyE 6740 | Prediction Study on Flight Departure Delay Severity

**Repository:** `https://github.com/clu360/-Prediction-Study-on-Flight-Departure-Delay-Severity.git`  
**Repository status when this spec was drafted (November 7, 2026):** public, empty, default branch `main`. Confirm this again at the start of work.  
**Team:** Caleb Lu, Ho Young Lee, Christopher Yi  
**Upstream documents:** Team's `ISYE_6740_Project_Proposal.pdf`; `ISYE_6740_Flight_Delay_Project_Workflow.md` (prior project workflow).  
**Data:** User-supplied January–December 2025 BTS Reporting Carrier On-Time Performance ZIP/CSV monthly raw files. The user will make them available in the Codex workspace.

> **Execute Phase 1 only in this task.** Set up the repository and build, test, and run the preprocessing pipeline. Do not implement feature-group training matrices, classical classifiers, or the MLP yet. The latter phases below define the contract that Phase 1 must support. Do not report success unless the real twelve-month data were actually processed and their outputs validated. If the raw files are absent, implement and test the pipeline with synthetic fixtures, describe the missing inputs, and stop short of claiming a full dataset.

## A. Project goal and the locked research protocol

Predict the **upcoming flight's departure-delay severity** using: (1) scheduled flight information, (2) information from the incoming aircraft's preceding flight, and (3) historical airport/airline operations. Compare probability baselines, classical machine learning, and an MLP on temporally out-of-sample data, with special analysis for **ATL, DFW, DEN, ORD, LAX**.

### A1. Three-class target

```text
target_dep_delay_min = upcoming flight's BTS DepDelay, signed minutes
target_class = 0 if DepDelay < 15
             = 1 if 15 <= DepDelay < 60
             = 2 if DepDelay >= 60
```

Include early/negative departures in class 0. Exclude target flights with `Cancelled=1`, `Diverted=1`, or missing/invalid target `DepDelay`. BTS `DepDelay` is **departure** delay despite a typo in §2 of the proposal calling it arrival delay.

### A2. Forecast-time contract

We predict flight `i` **after the linked preceding aircraft flight `j` has actually taken off, but before `j` arrives and `i` departs**. Use the reproducible proxy `prediction_time_utc = prior_wheels_off_utc + 1 minute`. For the primary operational cohort, require `prediction_time_utc < target_crs_dep_utc` (and, for retrospective sanity checking only, verify the target had not already departed).

The prediction feature table may know the prior flight's **actual departure delay** and its scheduled flight details. It may **not** know prior actual arrival, prior arrival delay, actual elapsed flight duration, target departure outcome, post-departure delay causes, future airport delay rates, or future scheduled-flight completion status. Aircraft tail assignment in BTS is retrospective; acknowledge that real-time assignment may differ.

### A3. Unit of observation and modeling population

One row represents a scheduled upcoming flight with its outcome label and either a **valid immediate predecessor** or an explicit non-linkage reason. **Phase 1 produces both all eligible completed target flights and the subset that qualifies for the incoming-aircraft forecasting cohort.** Do not restrict raw data to the five comparison airports before joining aircraft legs. Those five are **evaluation subgroups by target origin**, not the entire ingestion scope.

### A4. Temporal evaluation design (later phases)

Train on three calendar months; score the following calendar month:

| Train | Score | Role |
|---|---|---|
| Jan–Mar 2025 | Apr | Development fold 1 |
| Feb–Apr | May | Development fold 2 |
| Mar–May | Jun | Development fold 3 |
| Apr–Jun | Jul | Development fold 4 |
| May–Jul | Aug | Development fold 5 |
| Jun–Aug | Sep | Development fold 6 |
| Jul–Sep | Oct | Development fold 7 |
| Aug–Oct | Nov | Final out-of-sample month |
| Sep–Nov | Dec | Final out-of-sample month |

Use Apr–Oct folds for model and hyperparameter selection. Freeze choices before Nov–Dec evaluation. Sequential retraining for Dec using historical November labels is permitted and must be described explicitly; no tuning based on either final month's results. No random train/test split. In Phase 1, build only a split **manifest**, not trained models.

## B. Phase 1: Build a deterministic, auditable preprocessing pipeline

### B0. Repository and input discovery (must happen first)

1. Clone/open the exact repo. Check branch and contents. If still empty, initialize `README.md`, `.gitignore`, `pyproject.toml`, package directory, tests, and `docs/`. Work on a named feature branch such as `feat/bts-preprocessing`; preserve unrelated files if the repo changed since the spec.
2. Locate the user-supplied 12 monthly source files under the Codex workspace; accept ZIP containing CSV or standalone CSV. **Never assume names**. Parse candidate month/year from filenames *and* the `Year`/`Month`/`FlightDate` columns. A month file may have an internal CSV name different from the archive name.
3. Produce a discovery table: file path, ZIP member name, size, detected month, row count, CSV header list, duplicates, missing required columns, read errors, SHA-256 of source bytes. Confirm exactly one authoritative input per 2025 month; do not silently concatenate duplicate month copies.
4. Fail clearly if any month 1–12 is missing, unreadable, or incorrectly dated. If multiple candidates represent the same month, stop for explicit resolution. Do **not** download or substitute an unofficial dataset without approval.
5. Maintain raw input files byte-for-byte unchanged. The repository is **public**; exclude all source ZIP/CSV, processed Parquet, scratch outputs, local secrets, and large checkpoints from Git. Only commit code, documentation, small synthetic fixtures, and aggregate quality summaries.

**Canonical source:** BTS Reporting Carrier On-Time Performance, NOT its separate Marketing Carrier table. Official selector:  
`https://www.transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=b0-gvzr&gnoyr_VQ=FGJ`  
Official field dictionary: `https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ`  
Official 2025 monthly ZIP listing: `https://transtats.bts.gov/PREZIP/`.

### B1. Required source fields and explicit allow/deny rules

Inspect actual headers; normalize accepted spelling variants without changing semantics (e.g. `Tail_Number` / `TailNum`). Do not guess the meaning of unknown columns.

| Source field | Required? | Phase 1 use |
|---|---|---|
| `FlightDate`, `Year`, `Month` | Required or reliably derivable from `FlightDate` | Target local calendar date; monthly coverage |
| `Reporting_Airline` | Required | Carrier ID; G1 source |
| `Flight_Number_Reporting_Airline` | Required if present in official selected dataset; otherwise explicitly log | Audit and duplicate detection |
| `Tail_Number` | Required for linking; may be null per row | Exact aircraft predecessor chain |
| `Origin`, `Dest` | Required | Airports, continuity validation |
| `CRSDepTime`, `CRSArrTime` | Required | Local scheduled clock fields |
| `CRSElapsedTime` | Required | Scheduled duration; scheduled arrival UTC |
| `Distance` | Required for later G1; missing values flagged | Flight distance |
| `DepDelay` | Required (some rows may be null) | Upcoming **label**, prior known departure delay |
| `Cancelled`, `Diverted` | Required | Eligibility, intervening-flight guards |
| `DepTime` | Preferred | Audit actual gate departure |
| `TaxiOut` | Preferred | Resolve/check prior takeoff timestamp |
| `WheelsOff` | Required unless `DepDelay + TaxiOut` securely reconstruct takeoff | Prior actual takeoff/forecast instant |
| `DOT_ID_Reporting_Airline`, `DayOfWeek`, `OriginAirportID`, `DestAirportID` | Optional | Cross-checks and later metadata |
| `ArrTime`, `ArrDelay`, `WheelsOn`, `ActualElapsedTime` | Optional **audit-only** | Never model predictors, never used to choose a predecessor |

Target realized `DepTime`, target `WheelsOff`, target `TaxiOut`, `DepDel15`, `DepDelayMinutes`, `DepartureDelayGroups`, `LateAircraftDelay`, `CarrierDelay`, `WeatherDelay`, `NASDelay`, `SecurityDelay`, any arrival outcome, and all target realized post-schedule outcomes are **forbidden predictor columns**. Their presence in raw data is harmless only if the downstream feature allowlist excludes them. Do not calculate label from `ArrDelay`.

If an essential column is entirely absent, **stop and report the source-column mismatch** instead of filling it with made-up data. If `WheelsOff` is absent but `TaxiOut` and `DepDelay` allow takeoff reconstruction, implement a separately flagged `takeoff_time_source=derived` path, verify it with sample rows when possible, and document the resulting limitation.

### B2. Canonical row identity and data cleaning

1. Read all twelve months with explicit dtypes and safe parsing for blank/numeric null values, padded or unpadded `hhmm` fields, and strings such as `NA` or empty. Preserve raw text of identifiers where needed; do not treat tail numbers as numeric.
2. Generate a **stable source record ID** from the source file's canonical month and 0-based CSV row index plus a content/source hash. Never assume flight number + date uniquely identifies a row. Keep `raw_source_month`, `raw_source_file`, and `raw_row_number` metadata locally.
3. Determine and report exact duplicate raw records, conflicting purported duplicates, null tail IDs, invalid airport codes, invalid dates/clocks, missing distances and elapsed times, invalid cancellation/diversion flags, and missing delays.
4. Keep a *master flight-event table* including canceled and diverted flights for conservative tail-chain auditing. For target label cohort, retain only flights with `Cancelled==0`, `Diverted==0`, and nonnull finite `DepDelay`. **Do not delete canceled/diverted records from the tail chain before determining adjacency**, since skipping an intervening movement can fabricate a link.
5. Apply target formula in A1. Save raw numeric label `target_dep_delay_min` strictly for evaluation/EDA, not feature input.
6. Convert `Origin`/`Dest` airport codes to uppercase, `Tail_Number` to a standardized nonempty string without inventing a missing value, and parse time fields as minute-precise clocks. Keep a per-column null/invalid count.

### B3. Time-zone and overnight normalization

All BTS scheduled/realized `hhmm` clocks are in **local airport time**. Do not sort flights by raw `FlightDate + hhmm` without converting to absolute timestamps.

1. Use a documented, version-pinned **IATA airport → IANA timezone** map covering every observed 2025 airport; store the mapping provenance and its unresolved-code list. `zoneinfo` (IANA) or equivalent is acceptable. Use per-date DST rules, not fixed UTC offsets.
2. Interpret `FlightDate` as the originating airport's scheduled departure local date and `CRSDepTime` as the local scheduled clock; obtain `target_crs_dep_utc`. Validate `0000` and `2400` carefully; define exact policy for `2400` as next-day midnight only where BTS values warrant it.
3. Derive `scheduled_arrival_utc = scheduled_departure_utc + CRSElapsedTime minutes`. Use `CRSArrTime` and the destination timezone as a cross-check, not as a simple same-date timestamp. This avoids wrong overnight/timezone subtraction.
4. For an **operated previous leg**, derive `actual_gate_departure_utc = scheduled_departure_utc + DepDelay minutes` when both fields are valid. Resolve `WheelsOff` to UTC by choosing the local calendar date nearest the actual gate departure consistent with a **nonnegative plausible taxi-out interval**. When `TaxiOut` exists, check `actual_wheels_off_utc ≈ actual_gate_departure_utc + TaxiOut` and verify the local `WheelsOff` clock; allow a documented small reporting tolerance, not an arbitrary hour offset.
5. For records lacking usable `WheelsOff` but with valid `TaxiOut`, derive takeoff UTC from actual gate departure + taxi-out and set `takeoff_time_source=derived`; for ambiguous/inconsistent timestamps set `takeoff_time_source=invalid` and exclude from valid predecessor matching.
6. Log daylight-saving transitions (especially the March and November changes), cross-time-zone legs, overnight arrivals, positive/negative delays and day-crossing actual takeoffs. If a fall-back hour is truly ambiguous and cannot be resolved with independent schedule/elapsed checks, **flag and exclude** rather than arbitrarily choosing a DST fold.
7. All `*_utc` columns must be timezone-aware UTC timestamps. No unlocalized or mixed-timezone values in any chronological join.

Do not hard-delete unusual long flights or taxi-out times solely via arbitrary numerical cutoffs. Flag and quantify outliers, inspect them, and only apply explicitly documented feasibility rules.

### B4. Conservative immediate-predecessor linkage

**The distinction between physical adjacency and mere tail matching is critical.** A previous flight with the same tail number is not necessarily the incoming leg for an upcoming departure.

For every target `i` in the completed-and-labelled cohort:

1. Use the **master table** of all records, grouped by nonmissing `Tail_Number`, sorted by `scheduled_departure_utc` with stable source ID as tie-breaker. Select the **immediately preceding scheduled record** `j` for that tail. Do **not** silently skip an intervening record to find a more convenient match.
2. Require `j` to be an operated non-canceled, non-diverted leg, with valid prior departure delay, valid prior takeoff UTC, and `Dest_j == Origin_i`. Compare standardized airport codes or verified airport IDs.
3. Require `scheduled_departure_utc_j < scheduled_departure_utc_i`, and `prediction_time_utc_i = actual_wheels_off_utc_j + 1 minute < scheduled_departure_utc_i`.
4. Define `scheduled_turnaround_min = scheduled_departure_utc_i - scheduled_arrival_utc_j`; require nonnegative scheduled turnaround. Limit the scheduled gap between predecessor arrival and target departure to **24 hours** for the primary cohort, with the threshold configurable and a 12-hour sensitivity check in an audit. Preserve excluded rows and reason counts. These are conservative study-design thresholds, not BTS rules.
5. If multiple records share the same tail and effectively identical schedule ordering or there is a conflicting intervening operated record, mark the match **ambiguous** and exclude from the primary linked cohort. Do not infer a match using **future** `ArrTime`, `ArrDelay`, `WheelsOn`, or target `DepDelay`.
6. Check once after selection, **only as an audit**, whether target actual gate departure appears to precede the prediction instant or the claimed predecessor's observed arrival is inconsistent. An audit failure should be flagged, investigated and quantified; it must not silently alter feature construction via target information.
7. Every completed target is assigned either `link_status=valid` or a stable `link_reason`, such as `missing_tail`, `no_previous_record`, `intervening_cancelled_or_diverted`, `predecessor_missing_takeoff`, `airport_mismatch`, `late_prior_takeoff`, `negative_scheduled_turnaround`, `gap_over_24h`, `ambiguous_rotation`, `invalid_timezone`.

**Important edge case:** a January 2025 flight may have a real predecessor in December 2024 that is outside the supplied files. Mark `no_previous_record`; do not fabricate the connection or silently use a 2026 record. A 2024 buffer can be added later as a separately tracked enhancement.

### B5. Phase 1 output schema and filesystem contract

Use partitioned **Parquet** (e.g. `year=2025/month=01`) and save schema metadata and aggregate reports. Deliver:

```text
data/
  raw/                               # user inputs; gitignored, never modify
  interim/                           # normalized events; gitignored
  processed/
    flights_clean/                   # all completed flights with valid labels
    flight_connections/              # one row per completed target, valid or invalid link
    modeling_cohort/                 # subset of VALID linked forecasting rows
    history_departure_events/        # events available for Phase 2 as-of histories
    folds/                           # training/testing membership manifest
    audit/
      input_manifest.json
      preprocessing_summary.json
      exclusion_counts.csv
      linkage_examples.csv
      schema.json
      preprocessing_audit.md
docs/
  data_dictionary.md
  preprocessing_methodology.md
configs/
  preprocessing.yaml
src/flight_delay/
  ...                                # Python package, typed functional pipeline
tests/
  ...                                # synthetic unit + integration tests
scripts/
  ...                                # repeatable CLI entry point
README.md
AGENTS.md                             # repository-specific Codex guardrails
pyproject.toml
.gitignore
```

**Minimum `flights_clean` fields**:

```text
record_id, source_month, FlightDate, Reporting_Airline,
Flight_Number_Reporting_Airline, Tail_Number, Origin, Dest,
CRSDepTime, CRSArrTime, CRSElapsedTime, Distance,
target_crs_dep_utc, target_crs_arr_utc, target_dep_delay_min,
target_class, Cancelled, Diverted
```

**Minimum `flight_connections` fields**, one row per `flights_clean` `record_id`:

```text
target_record_id, prior_record_id (nullable),
link_status, link_reason,
prediction_time_utc (nullable), prior_wheels_off_utc (nullable),
prior_actual_gate_departure_utc (nullable), takeoff_time_source,
prior_scheduled_departure_utc (nullable),
prior_scheduled_arrival_utc (nullable),
scheduled_turnaround_min (nullable),
prior_to_target_schedule_gap_min (nullable),
prior_route_continuity_ok, timestamp_quality_flag,
ambiguous_tail_rotation_flag
```

**Minimum `modeling_cohort` fields**, one row for **each** validated link:

```text
target_record_id, prior_record_id, FlightDate, target_origin, target_dest,
target_carrier, target_flight_number,
target_crs_dep_utc, target_crs_dep_local_hour,
target_day_of_week, target_month,
target_crs_elapsed_min, target_distance_miles,
prior_origin, prior_dest,
prior_dep_delay_min, prior_crs_elapsed_min,
prior_wheels_off_utc, prediction_time_utc,
scheduled_turnaround_min, minutes_until_target_crs_dep_at_prediction,
target_class, target_dep_delay_min,
source_month, timestamp_quality_flag, takeoff_time_source
```

**IMPORTANT:** `modeling_cohort` may contain `target_dep_delay_min` as a **label/audit column**, but model-building functions must use explicit feature allowlists. **No model input matrix** may include `target_dep_delay_min`, `target_class`, IDs, timestamps used only for leakage control, realized target outcomes, or the predecessor's arrival outcomes. Preserve these metadata columns for later analysis; avoid accidental `drop(target)`-style feature selection.

**Minimum `history_departure_events` fields:** `record_id`, airport of departure, carrier, `observed_departure_time_utc` (= valid actual gate-out UTC), signed `DepDelay` and derived severity, plus valid-observation flags. This is an **event store for later causal as-of calculations**, not a target-feature table. Do not aggregate over the full future month in Phase 1.

**Minimum `folds` fields:** `target_record_id`, `fold_id`, `train_months`, `test_month`, `role` (train/test), `is_final_holdout`. Rows can legitimately occur in multiple rolling folds; keep unique (`target_record_id`, `fold_id`, `role`) keys. Generate these for the primary linked cohort using the **target local FlightDate**; no random splitting.

Write `data_dictionary.md` with column name, dtype, source field, transformation formula, allowed predictor group (G1/G2/metadata/label), availability relative to prediction time, and missingness policy. Use **explicit standardized snake_case names** in processed Parquet, with a raw-to-processed mapping; the mixed-case examples above identify logical source values, not a requirement to preserve mixed-case in all files.

### B6. Data integrity tests and scientific safeguards

Implement automatically runnable **synthetic tests** first:

- Class labels at `-5, 0, 14.999, 15, 59.999, 60, 180` (and null rejection).
- Overnight takeoff/arrival; negative departure delays; local HHMM zero-padding and 2400; DST spring forward / fall back; cross-zone flight (e.g. ATL↔DFW).
- Same-tail, same-airport immediate preceding leg links; airport mismatch rejects; wrong tail rejects; intervening canceled/diverted/moved leg is not silently skipped; no predecessor for first observed tail.
- Late preceding takeoff `>=` upcoming scheduled departure rejects; scheduled turnaround negative or over 24 hours rejects; exact boundary behavior documented.
- Ambiguous duplicate tail rotation rejects; missing/invalid `WheelsOff`/time zones handled with explainable status.
- Test linkage remains unchanged if later prior `ArrDelay`, `ArrTime`, `WheelsOn`, and target delay cause fields are permuted or removed.
- Known records fall into the correct Apr–Oct and Nov–Dec folds; no train sample whose target month is the test month within that fold; no duplicates in unique key spaces.
- A reproducibility test: same input hashes/config produce identical counts and sorted logical output hashes on rerun.

**Full-data acceptance conditions:**

1. All twelve 2025 source months are present, with raw row counts, source hashes, and canonical headers documented.
2. Every retained completed target has exactly one label in {0,1,2}. Raw→clean→linked flow accounting reconciles without unexplained loss.
3. Every valid linked row has a nonempty tail, immediately preceding same-tail scheduled record, prior destination equal to target origin, predecessor valid and earlier, prediction instant before target scheduled departure, and scheduled turnaround within the configured interval.
4. Zero non-UTC datetime comparisons; no unhandled ambiguous DST or midnight parsing.
5. The primary cohort contains all five airports as **subsets if present in source**; save per-month, per-airport sample counts and class distributions. Report if a subgroup is too small or absent.
6. Explicit per-reason exclusion counts, prior-link coverage overall and by airport/carrier/month, percent of takeoff times derived vs directly parsed, timestamp anomalies, and class distributions **before and after** the linkage eligibility filter.
7. Report at least **50–100 manually reviewable linkage examples** sampled across early-year, late-year, overnight, cross-zone and unusual-time cases, with identifiers and scheduling/timing details. Avoid exposing unneeded personal information.
8. All tests pass; CLI finishes successfully on all twelve months without loading all CSV columns into a single in-memory dense matrix; Parquet files are readable and schema-stable.
9. No raw/large data in Git commits. Do not substitute a tiny sample for full processing while calling results complete. If runtime/space is a blocker, report measured resource use and a safe streaming plan.

### B7. Implementation and run logistics

**Recommended Python stack:** Python 3.11+, `polars` or chunked `pandas`, `pyarrow` for Parquet, `zoneinfo` plus a versioned airport→IANA timezone mapping, `pytest`, and a typed configuration/CLI. Reproducibility is more important than a particular library.

Provide these usable entry points (names can be adapted but document exact commands):

```bash
python -m pip install -e '.[dev]'
python -m flight_delay.cli discover --raw-dir data/raw --year 2025
python -m flight_delay.cli preprocess --raw-dir data/raw --out-dir data/processed --year 2025 --config configs/preprocessing.yaml
python -m flight_delay.cli audit --processed-dir data/processed
pytest -q
```

Use configurable paths, not hardcoded personal directories. Cache valid intermediate Parquet and run idempotently; never mutate input files. Emit progress by month/stage, elapsed wall time, peak memory when measurable, record counts and structured errors. The CLI should have clear exit codes and no silent fallbacks. Pin dependency versions or provide a lockfile. Write a concise `README.md` setup/run/troubleshooting guide.

### B8. Required Codex handoff after Phase 1

Return a concise, evidence-backed completion report covering:

1. Git branch and commit/PR link (if operations are authorized and performed).
2. Exact 12 input filenames detected, counts, schema mismatches and hashes.
3. Raw/clean/linked/modeling counts, valid link percentage, three class shares, all exclusion-reason counts and monthly five-airport counts.
4. Exact local output paths and formats, Parquet schema, audit report paths.
5. Example validity checks, DST/timezone treatment, missing tail/rotation limitations.
6. CLI and test commands run, their pass/fail results, runtime/memory where measured.
7. Any **unresolved interpretation** or data issue that must be approved before feature engineering. No invented metrics.

Stop after the audited Phase 1 deliverables. Do not train models or write report conclusions in this task.

## C. Planned Phase 2: Three engineered feature groups (do not implement yet)

Phase 1 must preserve all raw ingredients for later construction. In Phase 2, create the modeling feature table by **as-of prediction time**, not by aggregating an entire month with lookahead.

| Group | Specific planned features | Logic |
|---|---|---|
| **G1: scheduled flight** | Target carrier, origin, destination, departure local hour / cyclic sin-cos, weekday, month, scheduled duration, distance | Schedule only |
| **G2: incoming aircraft** | Prior departure delay, prior origin, prior scheduled duration, scheduled turnaround, time until upcoming scheduled departure | All known after the preceding takeoff; no prior arrival information |
| **G3: historical operations** | Origin previous 7-/28-day departure volume, mean signed delay, moderate/severe delay rates, carrier-at-origin past rates/counts | Strict as-of join against earlier observed departures, smoothed for sparse cells |

Primary ablations: `G1`, `G1+G2`, `G1+G3`, `G1+G2+G3` on exactly the same linked cohort. Fit imputers/encoders/scalers **only on each rolling fold's training window**. The proposal's airline-specific historical information belongs to G3, not an unannounced fourth group.

Recommended smoothing for rate cells:

`smoothed_rate = (cell_positive_count + alpha * global_past_rate) / (cell_past_count + alpha)`, with `alpha` fixed during development and `global_past_rate` computed from earlier events only. Use a neutral fallback when absolutely no earlier events exist. Add history cell counts to distinguish uncertain from well-supported rates. Freeze a single online-updated-history-vs-month-boundary-frozen regime before model benchmarking.

**Phase 2 deliverable:** `modeling_features.parquet` plus four feature-view allowlists, as-of leakage tests and fold-specific encoding functions.

## D. Planned Phase 3: Classical ML + dedicated analysis

Models: historical class-frequency probability baseline, multinomial logistic regression, random forest, XGBoost (`multi:softprob`). Each gives an ordered triple `[P(Y=0),P(Y=1),P(Y=2)]`.

Across four feature groups and all chronological folds, evaluate accuracy, **macro-F1**, class-specific precision/recall (especially severe class), multiclass log loss, macro one-vs-rest AUC-ROC, confusion matrices, reliability/calibration, observed vs **mean predicted** severity shares, and per-airport subgroup performance. The hard-predicted-class histogram is a distinct statistic from mean soft probabilities.

Feature interpretation: configuration ablation and held-out permutation importance, aggregated from encoded columns to raw features and groups. All comparisons use the same target IDs and prediction timestamps; do not make causal claims from importance scores.

**Phase 3 deliverable:** classical-model predictions, metrics, figures, ablation results and written verified findings.

## E. Planned Phase 4: MLP implementation + dedicated analysis

Use the same raw feature information, four configurations, chronological folds and evaluation cohort. A reproducible starting network is categorical embeddings + scaled numerics → Dense(128)/ReLU/dropout → Dense(64)/ReLU → 3 logits; cross-entropy training and softmax at inference. Development-only tuning of dimensions/dropout/learning rate/early stopping; monitor class imbalance and probability calibration.

MLP analysis replicates classical metrics and adds learning curves, early-stopping epoch, random-seed stability and runtime. Compare only after both analysis packages are frozen.

**Phase 4 deliverable:** checkpoint, transformations, model probabilities, learning curves, tests and matched analysis.

## F. Planned Phase 5: Final synthesis and Phase 6: Report

Final comparative outputs:

1. Five-model leaderboard by Nov/Dec and macro-F1/log loss with per-class diagnostics.
2. All-model feature-set ablations and full-feature vs scheduled-only gains.
3. Observed vs mean predicted minimal/moderate/severe shares at ATL, DFW, DEN, ORD, LAX, with sample sizes and severe-delay calibration.
4. Global and airport-specific permutation/ablation importance; interpret associations only.
5. Linkage-coverage limitations, time-zone robustness, seasonality, class imbalance and data shift.
6. Optional **other-year** experiment only with separately acquired and approved data; one year of 2025 data does not support a claim about year-to-year out-of-distribution generalization.

**Write the final report only after all implementation analyses and comparison tables are locked.** Suggested report: introduction/question, BTS and inclusion policy, exact prediction-time definition, preprocessing/linkage coverage, G1–G3 construction, temporal protocol, classical and MLP methods, evaluation, results by question, limitations, reproducibility appendix. The proposal's claims that MLP/RF will do best or full features always help are **hypotheses, not established results**.

## G. Global project rules for all phases

- Never claim a dataset was produced or model evaluated unless logs/artifacts confirm it.
- Keep the raw data immutable and outside public Git. Do not commit full processed flight-level data by default.
- Use explicit feature allowlists and anti-leakage checks; never infer predictors by taking every non-label column.
- No use of prior arrival outcomes or current realized outcomes in prediction features.
- Build deterministic, testable stages; fail clearly on missing months, impossible schema or ambiguous rotations.
- Do not silently revise class thresholds, model cohort, rollout chronology or holdout policy.
- If a material scientific choice is ambiguous, document the alternatives, choose the conservative auditable default specified here, and ask for confirmation before expanding scope.
- Avoid heavy dependencies or distributed compute unless demonstrated necessary from actual measured resource use.
- Keep a clean separation among (a) raw/source facts, (b) preprocessed/linked event data, (c) as-of feature matrix, (d) model outputs, and (e) analysis/report claims.

## H. Authoritative source references

1. Team-provided `ISYE_6740_Project_Proposal.pdf`, pp. 1–5.
2. Team-generated `ISYE_6740_Flight_Delay_Project_Workflow.md` (prior workflow).
3. BTS [Reporting Carrier download page](https://www.transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=b0-gvzr&gnoyr_VQ=FGJ), [field dictionary](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ), [monthly ZIP index](https://transtats.bts.gov/PREZIP/), [download help](https://www.transtats.bts.gov/showHelp2.asp).
4. October 10, 2026 project-ideas discussion: three severity classes; probability outputs; universal forecast after prior takeoff but before prior arrival; incoming prior departure delay (not arrival delay); 3-month-to-1-month rolling evaluation; five-airport comparisons.
