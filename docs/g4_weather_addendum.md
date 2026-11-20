# Addendum G4: Airport Weather Observation Features (Secondary, Development-Only)

Project: ISyE 6740 — A Prediction Study on Flight Departure Delay Severity

Team: Caleb Lu, Ho Young Lee, Christopher Yi

**Status: DRAFT for team review. Nothing in this document is authorized for implementation until the team approves it and AGENTS.md carries the authorization line in §11.**

Revision 2 (2026-11-20): the feature list was reduced from seven to five after an exploratory analysis on January–March data (§13), and the case for forecast features was weakened (§4, §10).

## 1. Purpose and authorization boundary

Phase II deliberately excluded weather from the initial feature set ([phase2_specification.md §2, G3](phase2_specification.md)). The completed Phase III study shows that the incoming-aircraft group (G2) carries almost all predictive signal. Even the best model, XGBoost G1_G2, classifies 43% of severe holdout delays as minimal. Delays that do not propagate from the incoming aircraft are the largest unexplained component, and local weather is the most plausible measurable cause.

This addendum defines one new feature group, **G4: airport weather observations**, and a predeclared development-only experiment that measures its incremental value.

Locked boundaries:

1. G4 is **additive and secondary**. G1, G2 and G3 definitions, the 20-feature Phase II table, the four locked views, Phase I artifacts and folds, and all official Phase III results remain unchanged and remain the primary study results.
2. Evaluation uses the **April–October development folds only**. November and December targets are never built, loaded, fitted, tuned or scored here. The holdout was already examined in Phase III, so a G4 holdout claim would require a fresh held-out period (for example, 2026 months processed by a new Phase I run) under a separate authorization.
3. All G4 choices in this document (sources, windows, lags, thresholds, features, models, metrics) are fixed **before** any G4 feature is joined to April–October labels. The feature list was informed by one exploratory analysis on January–March targets at five hubs (§13). Those months are training-only: no fold ever scores them. Changing any choice after seeing G4 development results requires a revised addendum. Results produced before the revision must be reported alongside the revised ones, not discarded.
4. No hyperparameters are tuned for G4 views. Models reuse the officially selected Phase III parameters of the corresponding base view.
5. The MLP and the final report remain out of scope.

## 2. Source data

**Source:** the Iowa Environmental Mesonet (IEM) ASOS/METAR archive, `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py`. It provides routine hourly METARs (issued around :51–:56 each hour) and special reports (SPECI) issued when conditions change, timestamped in UTC.

Fixed request parameters:

    station=<iem_station>  data=all  tz=Etc/UTC  report_type=3,4  missing=empty
    trace=0.0001  format=onlycomma  sts=2024-12-30T00:00:00Z  ets=2025-11-02T00:00:00Z

- `report_type=3,4` selects routine and special reports. One-minute HFMETAR (`report_type=1`) is excluded because its coverage differs by station.
- The window starts two days before 2025 so January lookbacks are complete. It ends after the last possible October prediction time. Weather observations are not labels. November-dated observations may serve only as features of October targets, and no November or December target row is ever constructed.

**Station mapping:** `configs/g4_station_map.csv`, checked into the repository.

- It has one row per airport appearing as `target_origin` or `target_dest` in the Phase I cohort (352 airports).
- Columns: `iata`, `icao`, `iem_station`, `override_reason`.
- `icao` comes from the pinned `airportsdata==20250909`. By default `iem_station = icao`.
- IEM accepts ICAO identifiers for all stations. Alaska, Hawaii and territory airports (`PANC`, `PHNL`, `TJSJ`, `PGUM`, `NSTU`, …) do not resolve by IATA code.
- Overrides must cite evidence. Known override: **PBI → `DJT`**. IEM files Palm Beach International under `DJT`, and `KPBI` returns no 2025 data. A 2026-11-20 probe confirmed that all 9,649 reports in the `DJT` 2025 file identify the station as Palm Beach: 9,648 as `KPBI`, and one formatted `METAR PBI`.

**Fetching rules:**

- Fetch sequentially, one station per request, with at least 2 seconds between requests. IEM enforces a 1-request-per-second IP throttle and caps a request at 1,000 station-years.
- Retry 429/503 responses with exponential backoff and a bounded attempt count.
- Send only station identifiers and dates, using a generic client identifier. No project, personal or flight data leaves the machine.
- Save each response byte-for-byte under `data/external/iem/metar/raw/`, which git ignores, with a `manifest.json` recording the request parameters, retrieval time (UTC), SHA-256, byte count and row count. The build never refetches silently. An existing raw file whose manifest hash verifies is reused. Refetching requires an explicit flag and writes a new version, never an overwrite.
- After fetching, report each station's days with at least one report. Any station with zero reports fails the fetch audit. This would have caught the PBI identifier problem.

Known coverage issues from the 2026-11-20 probe (verify at full fetch):

| Station | Issue |
|---|---|
| PBI | No data under `KPBI`; complete under `DJT` |
| HDN | No reports May–June 2025, partial April |
| ADK | Partial, especially October–November |
| BQN | About 17 reports per day (not 24-hour) |

## 3. Report parsing

Each IEM row is one report r from station s:

- `v_r`: the `valid` field, parsed as a UTC-aware timestamp (the request sets `tz=Etc/UTC`).
- **Availability time** `a_r = v_r + L`, with **L = 10 minutes**. This conservative publication lag reflects that a METAR is transmitted after its observation time.
- Duplicate `(station, v_r)` rows: keep a corrected report (`COR` in the METAR text) over the original. Otherwise keep the last row in file order. Log duplicate counts.
- Raw `metar` text is retained for audit only. It is never a feature.

Derived per-report fields:

**Ceiling (ft):** the height of the lowest layer among `skyc1..4` whose cover is `BKN`, `OVC` or `VV`.

- If no such layer exists (only `CLR`, `SKC`, `FEW`, `SCT`, `NSC` or `NCD`), the ceiling is unlimited.
- If the qualifying layer has no height, the ceiling is unknown.

**Visibility (statute miles):** `vsby`. Null is unknown.

**Flight category** (FAA definitions, ordinal). IEM provides no flight-category field, so it is derived from `skyc1..4`, `skyl1..4` and `vsby`:

| Category | Code | Ceiling | Visibility |
|---|---|---|---|
| VFR | 0 | > 3,000 ft | > 5 mi |
| MVFR | 1 | 1,000–3,000 ft | 3–5 mi |
| IFR | 2 | 500 to < 1,000 ft | 1 to < 3 mi |
| LIFR | 3 | < 500 ft | < 1 mi |

The category is the worse of the ceiling-based and visibility-based categories. If either is unknown, the category is null.

**Thunderstorm flag:** 1 if any space-separated `wxcodes` token contains `TS` (`TS`, `TSRA`, `-TSRA`, `+TSRA`, `VCTS`, `TSGR`, …), otherwise 0.

**Frozen/freezing precipitation flag:** 1 if any `wxcodes` token contains `SN`, `SG`, `PL`, `GS`, `FZRA` or `FZDZ`, otherwise 0. Substring matching intentionally counts `BLSN`, `DRSN`, `SHSN` and mixed codes such as `-SNPL`.

**None vs missing:** a blank `wxcodes` in an existing report means **no weather reported**, so both flags are 0, not missing.

**Wind (kt):** `max(sknt, gust)`. A blank `gust` means **no gust reported**, so wind equals `sknt`. If `sknt` is blank, wind is unknown.

## 4. Exact G4 definitions: 5 pre-encoding feature columns

For each target row, define:

- t = its `prediction_time_utc`; o = `target_origin`; d = `target_dest`; s(x) = `iem_station` of airport x.
- R_x(t) = reports of station s(x) with availability time a_r **strictly less than** t.
- latest_x(t) = the report in R_x(t) with the greatest a_r. It is **absent** if R_x(t) is empty or t − v_r > 180 minutes (staleness cap).
- W60_x(t) and W180_x(t) = reports of s(x) with a_r in [t − 60 min, t) and [t − 180 min, t), with an inclusive left boundary and a strict right boundary.

| New feature column | Exact definition | Window |
|---|---|---|
| g4_origin_worst_category_180m | Maximum non-null flight category code in W180_o(t); null if none | 180 min |
| g4_origin_frozen_precip_180m | 1 if any report in W180_o(t) has the frozen/freezing flag; 0 if W180_o(t) is non-empty and none do; null if empty | 180 min |
| g4_origin_thunderstorm_60m | As above with the thunderstorm flag over W60_o(t) | 60 min |
| g4_origin_wind_max_kt | Wind (kt) of latest_o(t); null if absent or unknown | latest, ≤ 180 min old |
| g4_dest_worst_category_180m | As the origin version, for station s(d) | 180 min |

Why these five (evidence in §13):

- **Worst category over 180 minutes** carried the most information in the exploratory analysis, more than the latest category alone. Restricted conditions reduce airport capacity, and the resulting traffic programs outlast the weather. Low ceilings and fog also occur in every season.
- **Frozen/freezing precipitation** was nearly as informative in winter (de-icing). It is expected to be near zero from May to September.
- **Thunderstorm** is retained on domain grounds. When present it more than tripled the severe-delay rate among flights whose incoming aircraft was on time, but winter has too few cases to measure its overall value. Measuring it in summer would require April–October labels, which this addendum reserves for evaluation.
- **Wind** carried moderate information overall and little among flights whose incoming aircraft was on time. It is kept as a single cheap numeric feature.
- **One destination feature** is kept because destination ceilings and visibility trigger FAA ground delay programs that hold flights at the origin. Its exploratory signal was about a third of the origin's (§13).

Revision 1 also had the latest origin category and a destination thunderstorm flag; both were dropped. Rain/drizzle was screened (§13) and not included.

**Encoding:** all five columns are numeric (ordinal codes, binary flags, knots). None is added to `CATEGORICAL`. Genuine missingness (absent, stale or unknown) is left null and filled by the existing train-fold-only median imputer in `FoldTransformer`. The "none" cases in §3 are never null.

**Ordering:** use the same strict as-of rule as G3. A report whose availability time equals t is excluded. Queries at equal timestamps are deterministic, with ties broken by target_record_id.

**Lead-time caveat:** among January–October targets, the time from prediction to scheduled departure has a median of 200 minutes (p25 137, p75 401, p90 701). In the exploratory analysis, the information carried by the origin worst category fell from 12.7 millinats for lead times up to 2 hours to 4.7 at 2–6 hours and 2.8 beyond 6 hours, so §7 stratifies by lead time.

However, conditions observed at the scheduled departure time were *not* more informative than conditions at prediction time (5.3 vs 5.6 millinats). That departure-time comparison is an oracle, an upper bound no forecast can beat. Delays appear to follow conditions that have already built up rather than the weather at the departure minute, which weakens the case for forecast features (§10).

## 5. Views

Two new views are defined in a separate G4 manifest. The locked Phase II `VIEWS` and `load_views()` are not modified.

| View | Source columns | Base view for comparison |
|---|---|---|
| G1_G2_G4 | 9 + 5 + 5 = 19 | G1_G2 (14) |
| G1_G2_G3_G4 | 9 + 5 + 6 + 5 = 25 | G1_G2_G3 (20) |

Views without G2 are omitted. G2 dominates, and the scientific question is G4's value on top of the best existing views.

## 6. Storage, privacy and modules

- **Raw downloads:** `data/external/iem/metar/raw/` plus manifest (ignored).
- **G4 table:** `data/features_g4/g4_features.parquet`, one row per target with `source_month ≤ 10`, keyed by `target_record_id`, with exactly the five G4 columns. Audit and logical hashes go under `data/features_g4/audit/` (ignored). Phase II tables are never mutated.
- **Private fits and predictions:** `data/model_outputs/g4_eval/` (ignored).
- **Public outputs:** `reports/g4/` holds aggregates only:
  - per-station coverage
  - a feature dictionary with missingness overall, by month and by ATL/DFW/DEN/ORD/LAX
  - a metrics summary
  - a validation report

  Station-level weather coverage is public data and may be published. No flight-level rows are published.
- **Modules:** a new package `src/flight_delay/weather/` containing `stations.py`, `fetch.py`, `metar.py` (parsing and per-report fields), `features.py` (as-of windows) and `views.py` (G4 view lists).
  - Reuse `FoldTransformer`, `models.registry.create`, `experiments.evaluation.evaluate` and `airport_metrics` unchanged.
  - Do not modify the Phase III config locks or runner. G4 evaluation is a separate secondary runner.
- **CLI (names adaptable):** `weather-fetch`, `weather-features`, `weather-audit`, `g4-evaluate`.

## 7. Predeclared evaluation

**Folds:**

- The seven development folds 2025-04 … 2025-10. Each trains on the three preceding months and tests on one month, with membership identical to the Phase I fold manifest.
- When the Phase I manifest is unavailable and folds are derived from source months, the August–October training IDs must reproduce the official frozen-fit `train_ids_hash` (`cf1f3815…7fcc`).

**Models:**

- **Primary: XGBoost** with the officially selected parameters for both base views: `learning_rate=0.1`, `max_depth=6`, `n_estimators=400`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method=hist`, `objective=multi:softprob`, seed 6740, 4 threads.
- **Secondary: multinomial logistic regression** with the base view's official SAGA parameters (`C=1.0`, L2, `tol=0.001`, `max_iter=400`). It uses the predeclared cyclical month encoding of [phase3_specification.md §4](phase3_specification.md) (sin/cos of 2π·month/12 in place of the `g1_month` one-hot) in **both** base and G4 views.
  - Rationale: a development-only secondary diagnostic (2026-11-20) found that the official one-hot month encoding leaves logistic probabilities systematically miscalibrated, with log loss worse than the constant baseline in every month. The cyclical encoding restores calibration (log loss below baseline in 7/7 months).
- **Same-environment refits:** base and G4 views are refit in the same run and environment. Official base-view metrics are used only as a reproduction diagnostic, never as the comparison arm. XGBoost predictions differ across platforms.

**Metrics:** every metric returned by `evaluate()` (accuracy, macro-F1, weighted F1, log loss, Brier, macro AUC, per-class precision/recall, confusion matrix, mean probabilities, calibration bins), plus `airport_metrics` for ATL/DFW/DEN/ORD/LAX.

**Primary comparison** (proposed; see §12):

- Δ = equal-month mean **log loss** of the G4 view minus that of its base view, with the number of months (of 7) in which the G4 view's log loss is lower.
- Log loss is proposed over macro-F1 because the secondary logistic diagnostic showed that argmax macro-F1 rewards miscalibrated over-prediction of minority classes. A proper scoring rule better isolates added information.
- Macro-F1, macro AUC and severe recall/precision are reported alongside as secondary comparisons.

**Predeclared strata** (metrics per stratum for base and G4 views):

1. **Adverse weather:**
   - adverse: `g4_origin_worst_category_180m ≥ 2` (IFR or LIFR), or `g4_origin_thunderstorm_60m = 1`, or `g4_origin_frozen_precip_180m = 1`
   - benign: all other targets with those three inputs non-null
   - unknown: targets with any of those three inputs null
2. **Lead time:** `g2_minutes_until_crs_dep_at_prediction` ≤ 120, 120–360, and > 360 minutes.

**Claim language:** results are descriptive. Flights are correlated within airports and days, so no IID confidence intervals or significance tests are claimed. Report the effect sizes, the k/7 month consistency and the strata. G4 results are labeled "secondary, development-only" wherever they appear.

**Compute budget (estimate):**

| Stage | Estimate |
|---|---|
| Fetch | 352 requests, ≤ 0.8 GB, 30–80 min |
| Feature build | Minutes |
| XGBoost | 4 views × 7 folds = 28 fits, about 65–85 s each on PACE |
| Logistic | 28 SAGA fits, about 2–6 min each |

## 8. Mandatory validation and anti-leakage tests

### Source and mapping

1. The station map covers 100% of cohort origin and destination airports. Every override has a documented reason. The PBI override is verified by the `KPBI` identifier inside the `DJT` METARs.
2. Raw files are byte-identical to their manifest SHA-256. A rebuild with existing raw files performs no network request.
3. The fetch audit fails on any station with zero reports. It lists per-station coverage days and the known gaps in §2.
4. All report timestamps are UTC-aware. No naive datetimes enter the feature build.

### Parsing

5. Flight-category boundaries are unit-tested:
   - ceilings of 3,000/3,001, 1,000/999 and 500/499 ft
   - visibilities of 5/5.01, 3/2.99 and 1/0.99 mi
   - the worse-of rule, unlimited ceiling, `VV` as ceiling, and an unknown layer height giving a null category
6. None vs missing: a report with blank `wxcodes` yields flags = 0 and a blank `gust` yields wind = `sknt`. A window with no reports yields null.
7. Thunderstorm and frozen token matching is tested on `VCTS`, `+TSRA`, `BLSN`, `-SNPL`, `FZFG` (must be 0 for frozen precipitation), `VCSH` and `RA` (both 0 for both flags).
8. Duplicate/COR resolution is deterministic and logged.

### Causal behavior

9. A report with availability time exactly t is excluded; one at t − 1 µs is included; one with valid time in (t − 10 min, t) is excluded.
10. Window boundaries: a report available at exactly t − 60 min is in W60 (inclusive left), and one at t − 60 min − 1 µs is not. The same holds for 180 minutes.
11. Staleness: a latest report with valid time older than t − 180 min yields null latest-report features.
12. Future injection: adding a report after t changes no feature of the target. Adding a report for another station changes nothing for unrelated airports.
13. No target, arrival or realized outcome field is read by the G4 build. The only cohort inputs are `target_record_id`, `prediction_time_utc`, `target_origin`, `target_dest` and `source_month`.

### Table and split integrity

14. The G4 table has exactly one row per cohort target with `source_month ≤ 10`, IDs match the Phase II feature table one-to-one, and no November/December IDs are present.
15. Phase I artifacts, the Phase II table, the four locked views and their logical hashes are unchanged. The existing test suite still passes.
16. G4 imputers and scalers are fitted on training rows only (existing FoldTransformer guarantees, re-tested on G4 views).
17. Rebuilding G4 from the same raw files reproduces identical logical hashes.

### Required audit

Publish, as aggregates:

- per-feature dtype, formula, availability rule, missing fraction (overall, by month, by the five airports) and distribution
- per-station coverage and staleness rates
- the share of targets in each adverse/benign/unknown stratum and lead-time stratum
- runtime and peak memory

## 9. Handoff when finished

1. Changed files and a commit or PR link; no data, raw downloads or local paths committed.
2. The station map with overrides and evidence; the fetch manifest summary and coverage table.
3. The G4 table schema, row/ID reconciliation and a missingness summary.
4. Test results for every check in §8.
5. The evaluation table (§7) for all four view-model pairs, with strata, the reproduction diagnostic and runtime.
6. Limitations requiring team review.

## 10. Out of scope (separate addenda required)

- **Terminal forecasts (TAFs) valid at the scheduled departure hour.** IEM archives TAF text, about 12.6 issuances per day at ATL in 2025 including amendments. The 2026-11-20 probe found coverage of 340/352 cohort airports, 99.2% of departures. Missing are PBI (unresolved) and several joint-use military fields (VPS, GRK, YUM, BLV) plus small airports. The oracle comparison in §4 and §13 suggests that even a perfect forecast of departure-time conditions would add little beyond observations, at least in winter. TAFs are therefore a lower priority than measuring G4 itself.
- Radar or gridded precipitation, and FAA ground delay programs, ground stops or other traffic-management advisories.
- Same-day airport departure outcomes ("current departure outcomes", also excluded by Phase II). These would be an informative comparator for G4 but need their own definition.
- Any November/December evaluation, hyperparameter tuning, MLP training or the final report.

## 11. Proposed AGENTS.md authorization line

> Secondary G4 weather features are authorized by docs/g4_weather_addendum.md for April–October development evaluation only. Do not modify G1–G3, the four locked views, Phase I/II artifacts or official Phase III results, and never build or score November/December G4 rows.

## 12. Decisions requiring team approval

1. **Primary comparison metric:** log loss (proposed) or macro-F1 (the Phase III convention).
2. **Timing constants:** publication lag L = 10 minutes and staleness cap = 180 minutes.
3. **Feature list:** the five features. In particular: whether the single destination feature and the winter-only frozen-precipitation feature stay in, and whether to keep thunderstorm without seasonal evidence.
4. **Secondary model:** whether to include the secondary logistic arm with cyclical month encoding.
5. **Follow-up:** whether to pursue a TAF addendum and a fresh 2026 holdout.

## 13. Exploratory evidence behind the feature list (January–March only)

Run on 2026-11-20, before any G4 feature was joined to April–October labels. January–March targets are training-only: no fold scores them. November and December were never loaded.

- **Population:** 300,546 January–March 2025 departures from ATL, DFW, DEN, ORD and LAX (severe rate 5.0%). Of these, 249,034 had an incoming-aircraft departure delay under 15 minutes ("aircraft on time").
- **Weather:** IEM METARs for the five stations, parsed per §3. Features were joined at prediction time with the 10-minute lag and 180-minute staleness cap of §4. Window features were approximated by trailing windows at each report time.
- **Oracle:** the same features from the latest report at or before the scheduled departure time. This is not a usable feature; it bounds what a perfect forecast could add.
- **Measure:** in-sample information gain about the three-class label, H(Y) − H(Y | feature), in millinats. Null is its own level, and wind uses four bins (<10, 10–19, 20–29, ≥30 kt). For scale, the G2 prior departure delay in four bins scored 54.1 on all flights and 3.1 on aircraft-on-time flights.

| Candidate | All flights | Aircraft on time | Oracle at departure (all flights) |
|---|---|---|---|
| Worst category, 180 min | 5.6 | 3.8 | 5.3 |
| Frozen/freezing precipitation, 180 min | 4.8 | 3.2 | 5.4 |
| Latest category | 4.6 | 3.4 | 3.9 |
| Wind (4 bins) | 2.3 | 0.6 | 2.2 |
| Rain/drizzle, latest | 0.8 | 0.8 | 0.4 |
| Thunderstorm, 60 min | 0.5 | 0.4 | 0.8 |

Severe-delay rates among aircraft-on-time flights:

- **By worst category at prediction time:** VFR 2.2%, MVFR 3.8%, IFR 4.3%, LIFR 5.2%.
- **Frozen precipitation:** 5.8% when present vs 2.5% when absent.
- **Thunderstorm:** 9.0% when present (only 992 flights) vs 2.6% when absent.

On hub-to-hub flights (27,153), the destination worst category scored 1.1 against 3.1 for the origin.

**Limitations:**

- Each score is for a single feature, computed in-sample on the same data it describes.
- Winter months and five hubs only.
- The scores overlap and are not additive.
- Weather days also have more incoming-aircraft delay, which the aircraft-on-time subset only partly controls.

These numbers rank candidates. They are not model results and are not the G4 evaluation.
