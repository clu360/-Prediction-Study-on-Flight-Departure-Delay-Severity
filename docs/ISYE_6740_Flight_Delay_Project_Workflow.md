# ISyE 6740: Flight Departure Delay Severity Prediction
## Implementation and analysis workflow (working plan, 7 November 2026)

**Team:** Caleb Lu, Ho Young Lee, Christopher Yi  
**Source of scope:** *ISYE_6740_Project_Proposal.pdf*, Sections 1–6, and the team's October 10 project-ideas discussion.  
**Goal:** Compare a historical-probability baseline, multinomial logistic regression, random forest, XGBoost, and an MLP on three-class departure-delay prediction, then examine the effects of additional operational information and differences across ATL, DFW, DEN, ORD, and LAX.

This document is a **plan**, not a claim that data have already been downloaded or models trained. Phase gates should be signed off before subsequent phases begin. All experiments must use the same eligible target flights, chronological splits, class definitions, and feature-availability rules when compared directly.

## 0. Lock the research protocol (before coding)

### Prediction task and timing

- Observation: an upcoming flight `i` associated retrospectively with the incoming aircraft's earlier flight `j` using `Tail_Number`.
- Operational prediction instant: just after `j`'s actual takeoff (`WheelsOff_j`) and **before** `j` arrives and `i` departs. No prior arrival delay, actual arrival time, actual flight duration, or target-flight realized outcomes may enter predictors. The assignment of the eventual tail number is known retrospectively in BTS and should be acknowledged as a deployment limitation.
- Define departure severity from the upcoming flight's signed `DepDelay` (minutes): `Y=0` if `<15`, `Y=1` if `15 <= DepDelay < 60`, `Y=2` if `>=60`. Early departures (negative delay) belong in class 0.
- Each model predicts probabilities `(p0,p1,p2)`, summing to one. Its hard prediction is `argmax_k pk`.
- Scope: reporting/operating-carrier domestic nonstop flights in **calendar year 2025**, subject to valid flight labels, completed service, and the specified incoming-leg availability. Keep the entire domestic network while building prior-leg joins; restrict **analysis**, not ingestion, to the five named airports.

### Corrections and clarifications for the final report

1. Proposal §2 calls `DepDelay` an *arrival* delay. This is a typo: BTS defines it as **departure** delay.
2. Proposal §3.1 describes November–December as held out “for hyperparameter tuning.” A genuinely untouched final test cannot also be used for tuning. **Recommended protocol:** conduct hyperparameter selection using April–October rolling development folds; treat November and December as out-of-sample evaluation months. Freeze hyperparameters and feature rules before scoring November.
3. Sequential rolling evaluation permits new ground-truth labels to join later training windows: train **Aug–Oct → predict Nov**, then train **Sep–Nov → predict Dec**. The latter is not November-to-December target leakage: November outcomes are historical by December. Explicitly disclose this rolling retraining; do not quietly claim a single frozen two-month holdout. If a strict frozen test is desired, run an additional Aug–Oct model on both Nov and Dec and label it separately.
4. Proposal §5.6 asks for out-of-distribution evaluation in other years; 2025 alone cannot test that claim. Treat a 2024/2026 experiment as an **optional extension**, with new downloads and a separately specified temporal transfer protocol. Do not claim year-to-year generalization from 2025 rolling tests.
5. Class probabilities alone are not an estimate of the expected delay **in minutes**. For an optional approximate-minute output, estimate `mu_k = E[DepDelay | Y=k]` **only from the training window**, then use `sum_k p_k * mu_k`. Clearly separate this approximation from the primary three-class objective; the severe-delay class is open ended.

## 1. Verified data access and acquisition

**Primary source:** U.S. DOT/Bureau of Transportation Statistics (BTS), [Reporting Carrier On-Time Performance download selector](https://www.transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=b0-gvzr&gnoyr_VQ=FGJ). BTS [table profile](https://www.transtats.bts.gov/TableInfo.asp?QO_fu146_anzr=b0-gvzr&gnoyr_VQ=FGJ) identifies domestic nonstop reporting-carrier flight records with **monthly** frequency. The [official field dictionary](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ) confirms the candidate columns below.

**Verified download route:** BTS [PREZIP directory](https://transtats.bts.gov/PREZIP/) lists all twelve monthly `On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2025_M.zip` archives for `M=1,...,12`. Each listed 2025 archive is roughly 25–32 MB compressed (total about 355 MB); extracted CSVs will be substantially larger. The download UI also allows choosing **Year = 2025**, **Period = selected month**, and fields. BTS help notes that whole-year downloads may be unavailable for large tables, so monthly downloads are the robust default. Verify the ZIPs and actual CSV headers upon acquisition, because no data files were provided with the proposal.

**Required data fields** (BTS export spellings may need normalization):

| Purpose | Columns / groups | How used |
|---|---|---|
| IDs and time | `FlightDate`, `Year`, `Month`, `DayOfWeek`, `Reporting_Airline`, `DOT_ID_Reporting_Airline`, `Flight_Number_Reporting_Airline`, `Tail_Number` | Deduplication, aircraft pairing, temporal fold assignment |
| Route | `Origin`, `Dest`, optionally `OriginAirportID`, `DestAirportID` | Current schedule and predecessor destination-to-origin continuity |
| Schedule | `CRSDepTime`, `CRSArrTime`, `CRSElapsedTime`, `Distance` | Allowed predictors, scheduled turnaround and time feasibility |
| Realized departure of **prior** leg | `DepTime`, `DepDelay`, `WheelsOff` | Prior aircraft's known departure delay and prediction instant |
| Target and eligibility | `DepDelay`, `Cancelled`, `Diverted` | **Target** label and exclusion of cancelled/diverted flights |
| Audit-only (optional) | `ArrTime`, `ArrDelay`, `WheelsOn`, `ActualElapsedTime` | Linkage diagnostics only; **never** predictors at the selected prediction instant |

Do **not** substitute `DepDelayMinutes` for signed `DepDelay` when defining labels unless the choice is explicitly recorded (classification thresholds happen to be compatible for 0/15/60 but numerical analyses differ for early departures). Do not feed `DepDel15`, `DepartureDelayGroups`, target `DepTime`, or any target delay-cause columns to models.

**Acquisition steps:**

1. Download January–December 2025 from the same **Reporting Carrier** table, not the distinct **Marketing Carrier** table. Prefer the same selected-column schema for all twelve months, or all-fields ZIPs with column projection after read.
2. Record source URL, retrieval date, archive name, compressed file size, SHA-256, row count, column list, and any download failures in a manifest. Keep monthly raw inputs immutable.
3. Ingest CSVs with typed schemas to partitioned Parquet (`year/month`), using Polars or chunked pandas/pyarrow for memory. Keep a stable `record_id` rather than assuming a composite flight key is unique.
4. Reconcile monthly totals, duplicate counts, cancellation/diversion counts, null fractions, and delay-class proportions. Record any missing 2025 month as a **stop condition**.

**Data-volume expectation:** the PREZIP listing supports the compressed sizes, not a guaranteed extracted size or number of eligible examples. Measure these during Phase 1 rather than claiming predetermined totals. Commodity laptop memory may be tight for full-data one-hot encoding; prefer Parquet, sparse matrices, categorical embeddings, and/or a shared sample for early smoke tests.

## 2. Phase I: data preprocessing and aircraft linkage

**Inputs:** 2025 monthly raw records; authoritative airport-to-IANA-time-zone mapping (separate small reference, with version and provenance).  
**Output:** one consistent target-flight table with identifiers, target class, predecessor ID or missingness flag, and fully audited prediction timestamp.

### 2.1 Standardize, validate, and filter

1. Normalize columns and data types; validate `FlightDate`, `hhmm` values, categorical airport/carrier codes, numeric delays, and Boolean `Cancelled`/`Diverted`.
2. Remove flights with `Cancelled=1`, `Diverted=1`, or missing/invalid **target** `DepDelay` for the primary completed-flight task. Keep a table of excluded rows for transparent population accounting. For predecessor matching, retain an appropriately audited pool of eligible prior legs; no future outcome becomes a feature.
3. Convert scheduled departure and arrival **local** times into timezone-aware timestamps, accounting for origin/destination time zones, overnight dates, and daylight-saving transitions. The BTS clock fields are local `hhmm`, not global UTC timestamps. For scheduled arrival, reconcile dates with `CRSElapsedTime`; for prior takeoff, reconcile `WheelsOff` against actual gate departure and expected flight sequence. Handle `2400`, ambiguous DST local times, missing clocks, and invalid transitions explicitly.
4. Preserve `scheduled_departure_utc`, `scheduled_arrival_utc`, and `prior_wheels_off_utc`, plus UTC parsing flags. Check representative cross-time-zone, overnight, and DST examples manually before bulk processing.

### 2.2 Construct incoming-aircraft pairs

For each target flight `i`, search earlier flight `j` on the same `Tail_Number` subject to:

- `Dest_j == Origin_i` (physical continuity of the aircraft's planned route).
- A valid `WheelsOff_j`, with takeoff before the prediction instant and before the target's relevant upcoming departure. The fixed prediction instant can be `WheelsOff_j + 1 minute`, assuming no prior-arrival information is used at that instant.
- Plausible schedule/rotation ordering; choose the most recent qualifying earlier leg by actual known departure/takeoff, with deterministic tie-breaking. Set a documented maximum gap (pilot 24 hours; sensitivity check at 12 hours) to avoid arbitrary aircraft rotations or multi-day mismatches.
- Restrict primary **pre-scheduled-departure** predictions to cases where the prediction instant precedes `CRSDepTime_i`. This intentionally removes cases where the incoming aircraft takes off after the next leg is already scheduled; report exclusion counts and severity distribution so selection bias is visible. If the team wants to allow post-scheduled-time risk updates, define and evaluate a separate scenario rather than mixing tasks.

Use retrospective arrival fields only to **audit** whether the chosen predecessor had already landed, not as a selection criterion or model feature. Drop or flag ambiguous matches instead of resolving them using the later observed arrival delay. `Tail_Number` can be missing or change with equipment swaps; characterize linkage coverage by month, airline, and target airport. The BTS tail identifier is retrospectively known and may not be known reliably to a live predictor, which is an important limitation.

**Acceptance checks:**

- All retained linked examples satisfy same-tail, arrival/departure airport continuity, valid chronological order, and allowed information timing.
- A reviewed sample of at least 50–100 pairs includes overnight and cross-time-zone cases; document anomalies and chosen rules.
- Publish flow counts: raw → completed → labelled → valid timestamp → linked → forecast-eligible, by month and for each of the five airports.
- Calculate target-class prevalence overall and conditional on linkage eligibility. No major systematic data drop is left unexplained.

## 3. Phase II: build and freeze the three feature groups

**Rule:** for target flight `i`, feature values must be known at or before its recorded prediction instant `t_i`. No hindsight from future 2025 records, the target outcome, or the preceding aircraft's arrival.

| Group | Features in first version | Specific construction / guardrail |
|---|---|---|
| **G1: Scheduled flight characteristics** | `Reporting_Airline`, `Origin`, `Dest`, `CRSDepTime` hour, day of week, month, `CRSElapsedTime`, `Distance` | Use schedule only. Encode hour cyclically (`sin/cos`); carrier, origin, destination as categoricals. Optional route ID / holiday only after scope approval. |
| **G2: Incoming aircraft** | `prior_DepDelay`, predecessor `Origin`, predecessor `CRSElapsedTime`, scheduled turnaround minutes, optionally `minutes_since_prior_takeoff` and link-quality flag | The **prior** gate departure delay is observable after `WheelsOff`; never use `prior_ArrDelay` or actual prior flight duration. Scheduled turnaround = target scheduled departure minus prior scheduled arrival, normalized to UTC. Negative or implausible values need audit/flag. |
| **G3: Historical airport/airline operations** | Origin airport past severe/moderate delay rate, mean signed departure delay, prior 7-/28-day departures, carrier-at-origin past delay rates and counts | Calculate from **past completed departures only** relative to `t_i`; use 7/28-day lookbacks and smoothing/shrinkage to global/carrier levels for small cells. Planned volume is valid only from schedules available by `t_i`; do not count future completed flights. |

The older conversation mentioned a separate airline-history concept; proposal §3.2 intentionally consolidates it into **G3**, leaving three final groups. Additional route/airline rates can be secondary ablations but should not quietly alter the promised G3 definition.

### Causal history features and leakage audits

- Compute history using rolling as-of joins / cumulative lagged aggregates. For each target row, a contributing flight's outcome must have been observed before `t_i`, including in the test month if the design simulates continuously updated public operational data. Choose one regime and document it: **recommended** is causal online feature updates with a fixed fitted model during each test month. An alternative is freezing historical rates at the training-window boundary. Do not mix regimes across models.
- For training rows, construct history as of **their own** prediction time, never by calculating full-month target averages in advance. Add unit tests with deliberately shuffled rows and injected future severe delays that must not change earlier features.
- Handle sparse airport/airline combinations with additive smoothing, a `history_count` feature, and a deterministic fallback to the global prior. Fit encoders/imputation/scalers on each fold's training set; reuse them unchanged for that fold's test set.
- Expose exactly four feature-set configurations across the experiment: **G1**, **G1+G2**, **G1+G3**, **G1+G2+G3**. For the primary ablation, score every model on the **same linked eligible cohort**, even when G2 is not used, to avoid confounding cohort selection with feature value.

**Phase gate:** create a feature dictionary with sources, formulas, availability times, missingness handling, and dimensionality. Verify no forbidden columns in any training matrix. Save the feature configurations as versioned artifacts.

## 4. Time-series validation and experiment contract

Treat every split by the **target flight's local flight date** and build features at the row's actual prediction time. An incoming predecessor may be in the preceding calendar month, so linkage must be created from the full chronological table **before** splitting, while respecting feature availability.

| Purpose | Train window (3 months) | Test / evaluation month | Usage |
|---|---|---|---|
| Development fold 1 | Jan–Mar 2025 | Apr 2025 | Model and feature design |
| Development fold 2 | Feb–Apr 2025 | May 2025 | Model and feature design |
| Development fold 3 | Mar–May 2025 | Jun 2025 | Model and feature design |
| Development fold 4 | Apr–Jun 2025 | Jul 2025 | Model and feature design |
| Development fold 5 | May–Jul 2025 | Aug 2025 | Model and feature design |
| Development fold 6 | Jun–Aug 2025 | Sep 2025 | Model and feature design |
| Development fold 7 | Jul–Sep 2025 | Oct 2025 | Hyperparameter/model selection completed here |
| **Final month 1** | Aug–Oct 2025 | **Nov 2025** | No tuning after scoring |
| **Final month 2** | Sep–Nov 2025 | **Dec 2025** | Fixed design, sequential model refit permitted |

**Hyperparameter selection:** Use the mean/variance of development-fold macro-F1 and multiclass log loss, with a prespecified tie-breaker favoring simpler/faster models. If fitting early stopping / calibrators, carve out a chronological validation slice **within the training window** (e.g. its last 2 weeks), never the next test month. For final Nov/Dec evaluation, all feature, architecture, and hyperparameter decisions are frozen. Retain every model's out-of-sample 3-vector, true class, flight ID, fold/month, and airport code. Evaluate all methods on the same November and December records.

Do not stratify randomly across dates. Tune on smaller representative chronological folds first; fit full datasets once pipelines are debugged. Document model seeds and versions, training hardware and runtime.

## 5. Phase III: classical machine-learning implementation **and analysis**

**Models:**

1. **Historical probability baseline:** from training-window global class proportions `(n0,n1,n2)/N`, predict that vector for every flight in the test month. Optionally add airport-smoothed historical baseline as a separately labeled stronger benchmark.
2. **Multinomial logistic regression:** one-hot categoricals, scaled numeric features, softmax; tune regularization `C` (e.g. `[0.01, 0.1, 1, 10]`) on development folds.
3. **Random forest classifier:** handles nonlinear interactions; tune max depth, minimum leaf size, and tree count; use a bounded first-pass grid and control runtime.
4. **XGBoost multiclass:** `objective='multi:softprob'`, `num_class=3`; tune depth, learning rate, tree count, subsampling / regularization; keep objective and exact output-class order explicit.

For each model, run the four agreed feature configurations using identical splits. Use a consistent treatment of class imbalance: baseline unweighted fits first; if class weights or resampling are evaluated, record these as **separate** experiments and recalibrate before interpreting probability quality.

### Classical-model analysis package (complete before MLP comparison)

- **Performance table:** fold mean ± SD during development; Nov and Dec separately for overall accuracy, macro-F1 (primary), class-wise precision/recall/F1, severe-delay recall/precision, one-vs-rest macro AUC-ROC (if each class is present), multiclass log loss, optionally multiclass Brier score.
- **Confusion matrices:** normalised by true class, along with sample counts, highlighting minority severe delays.
- **Probability calibration:** 3-class reliability curves / calibration error and log loss; optionally calibrate on chronological training-only validation data.
- **Distribution fidelity:** for each month/model, compare `observed_k = mean(1[Y=k])` against `predicted_k = mean(p_k)` for all three classes. Keep the `argmax` label histogram as a **separate** chart; it is not the same as averaged probabilities.
- **Feature-group ablation:** 4 configurations × 4 classical models; report absolute and relative gain in macro-F1/log loss, and whether gains are stable across months.
- **Feature importance:** logistic standardized coefficient interpretation with caution for correlated one-hots; random-forest/XGBoost **permutation importance** on held-out data; aggregate one-hot columns back to original feature names and show G1/G2/G3 importance and uncertainty. SHAP is optional if time permits.
- **Airport performance:** same model fit on the full eligible network; evaluate flights **originating** at ATL, DFW, DEN, ORD, LAX. Report n, observed class shares, mean predicted class shares, severe-risk calibration, and macro-F1 by airport. Also show whether feature-group gains vary by airport; avoid equating association with causation.

**Gate:** produce clean tables and charts, frozen out-of-sample predictions, and a short written finding per research question. Do not optimize based on final Nov/Dec numbers.

## 6. Phase IV: MLP building **and analysis**

**MLP inputs:** use the same **raw information** and the same four feature groups, folds, and examples as classical models. For full-scale 2025 data, encode high-cardinality `airline/origin/destination/prior_origin` categoricals via learned embeddings and standardize numeric features on the training window. A modest dense one-hot prototype is acceptable for smoke tests if memory permits; document the encoding difference explicitly.

**Reproducible first architecture (starting point, to be tuned only on development folds):** categorical embeddings concatenated with scaled numeric features → fully connected layer 128 → ReLU → dropout 0.2 → fully connected 64 → ReLU → output logits 3; train with cross-entropy, Adam (`lr=1e-3`), batch size 1024, maximum 30 epochs, and early stopping based on chronological within-train validation loss. Use softmax **only at inference** to return `(p0,p1,p2)`. Fix random seeds; monitor training-vs-validation loss and minority-class recall. Compare smaller 64–32 and larger 256–128 widths plus dropout/learning-rate alternatives within a constrained search budget.

**MLP analysis package:** replicate every classical-model analysis metric and chart above, including four feature configurations, November/December performance, probability distributions and calibration, five-airport breakdown, and ablation. Additionally include training/validation learning curves, early-stopping epoch, training-time/hardware, and a stability check over 3 random seeds for the chosen configuration. For attribution use group permutation importance on held-out data; explain that embedding weights themselves are **not** directly interpretable as feature effects.

**Gate:** save best checkpoint, config, preprocessing mappings, class-order metadata, per-flight out-of-sample probabilities, learning curves, and MLP analysis. No claim that MLP wins should be made until the shared comparison is computed.

## 7. Phase V: locked cross-model synthesis and research-question analysis

Use **identical Nov/Dec target-flight IDs** and fixed predictions to assemble final outputs:

1. **Model leaderboard:** global baseline vs logistic regression vs random forest vs XGBoost vs MLP; both Nov and Dec, with macro-F1 and log loss emphasized, accuracy/AUC as context, runtime/memory measured if comparable.
2. **Marginal value of feature information:** paired comparison of `G1` to `G1+G2`, `G1+G3`, and full `G1+G2+G3` for each method, with bootstrap confidence intervals **blocked by day or week** rather than treating adjacent flights as independent. A strong result might be no gain; the hypothesis is testable, not guaranteed.
3. **MLP versus classical models:** compare average and worst-month results, severe-class sensitivity, calibration, run cost and stability; distinguish a statistically small benefit from a practical benefit.
4. **Feature interpretation:** global rankings and differences across ATL/DFW/DEN/ORD/LAX, using within-airport holdout permutation with adequate sample-size warnings. Explain correlation and confounding.
5. **Airport distribution fidelity:** one table or grouped bar chart per airport with `Actual`, `Baseline`, `Logistic`, `RF`, `XGB`, `MLP` values for the three class shares. Also summarize severe-delay risk and prediction quality per airport; optionally compare high-volume routes within those airports subject to minimum cell sizes.
6. **Robustness/limits:** linkage coverage vs unlinked population, prior-leg time-gap sensitivity, smoothing/aggregation lookback sensitivity, class imbalance, exact prediction timing, schedule-known assumptions and retrospective aircraft assignment, temporal distribution shift. Do not portray five-airport findings as causal or prove other-year generalization without other-year data.

**Suggested final figure/table checklist:**

- Flow diagram and counts for cleaning/pairing; class imbalance by month.
- Development-fold rolling timeline and train/test counts.
- Five-model × four-feature-set macro-F1 and log-loss heatmaps/tables.
- Final Nov and Dec leaderboard with uncertainty.
- Severe-delay confusion matrices for best classical model vs MLP.
- Observed vs mean-predicted severity distributions across 5 airports.
- Feature-group ablation and top feature permutation chart.
- MLP learning curve; compact airport-level performance comparison.

## 8. Phase VI: report writing **after analyses are locked**

Write the report only after Phase V plots, tables, and results are frozen. Draft the **methodology sections early as notes**, but avoid writing Results/Discussion from unverified interim findings.

| Report section | Inputs from workflow | Core message |
|---|---|---|
| Introduction & motivation | Original proposal | Why three-level departure risk and operational feature sources matter |
| Dataset & prediction timing | Phases 0–2 | BTS 2025, inclusion/exclusion, exact observation/prediction definition, coverage and leakage safeguards |
| Feature engineering | Phase II | Three groups, as-of histories, categorical/numeric representations, ablation design |
| Models & tuning | Phases III–IV | Five probability-output benchmarks, shared folds, tuning policy, MLP network/training |
| Evaluation | Phases III–V | Macro-F1, minority class, probability quality, ablation, airport subgroups, temporal backtest |
| Results & discussion | Phase V | Evidence for each research question; failures and surprising findings are reported equally |
| Limitations & future work | Data audit and sensitivity | Retrospective tail linkage, data coverage, schedule-only approximation, selection bias, no causal claim, optional other-year validation |
| Reproducibility appendix | Logs and manifests | Exact columns, source hashes, package versions, random seeds, splits, model settings, plot definitions |

Re-check all statements against final figures. The proposal's outcome hypotheses are **hypotheses**, not conclusions. Audit plot units, class thresholds, which period each reported number covers, and whether the probability average rather than hard predictions was used for distribution charts.

## 9. Ownership and handoffs (proposed, team to confirm)

Split into three workstreams while maintaining one jointly approved protocol. The proposal names three members but does not assign jobs, so these are **unassigned roles**, not assumed commitments:

| Workstream | Primary responsibilities | Handoff |
|---|---|---|
| A: data engineering | Acquire and audit BTS, timezone normalization, tail linkage, as-of G1/G2/G3 features | Feature dictionary, eligible IDs, Parquet, quality/leakage tests |
| B: classical models & diagnostics | Baseline/LR/RF/XGBoost pipelines, tuning, common scoring, classical analysis | Reusable evaluation functions, predictions, classical figures |
| C: MLP & synthesis | PyTorch MLP, training/early stopping, MLP analysis, shared comparison/report assembly | Checkpoint, predictions, MLP figures, integrated results |

All three members review the eligibility rules and final findings. The evaluation contract (splits, target IDs, labels, metrics) has **one owner** and cannot be modified silently after the first results arrive. Assign ownership explicitly at kickoff.

## 10. Practical execution schedule and phase gates

Dates below are **relative workdays**, since the proposal provides no final project deadline.

| Workdays | Phase | What must be complete to proceed |
|---|---|---|
| 1 | Protocol and access | Prediction timestamp, eligibility, fields, sources and split protocol agreed; all 12 archives reachable |
| 2–4 | Data preprocessing | Parquet dataset, 2025 counts, timestamp audit, linked pairs and eligibility accounting |
| 5–6 | Feature construction | G1/G2/G3 tables, as-of unit tests, four ablation matrices, fold manifests |
| 7–9 | Classical implementation + analysis | Four benchmark models and all scheduled classical diagnostics (use a small pilot first) |
| 8–11 | MLP building + analysis | Working training loop, tuned architecture, matching probability outputs, learning curves and diagnostics |
| 12 | Locked comparison | Nov/Dec report-ready tables, airport comparisons, uncertainty and robustness checks |
| 13–15 | Report | Complete research-question narrative, final graphics, checks, reproducibility appendix |

Parallel work is feasible after the feature schema and evaluation contract are frozen; MLP can begin as soon as the first valid matrices exist. Leave a buffer for BTS download issues and chronology audits, which pose greater risk than the first model fit.

**Completion definition:** twelve months accounted for; all sample exclusions measured; no forbidden future predictor; meaningful pair audit; fully reproducible seven development folds plus two final months; five models output valid three-class probabilities; four feature ablations on a shared cohort; five-airport subgroup results; documented importance/calibration; MLP learning curves; a coherent synthesis; report written from locked evidence.

## Source references

1. Team proposal, *ISYE_6740_Project_Proposal.pdf*, pp. 1–5 (provided by the user).
2. BTS TranStats: [Reporting Carrier On-Time Performance download](https://www.transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=b0-gvzr&gnoyr_VQ=FGJ), [table profile](https://www.transtats.bts.gov/TableInfo.asp?QO_fu146_anzr=b0-gvzr&gnoyr_VQ=FGJ), [field definitions](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ), [PREZIP monthly archives](https://transtats.bts.gov/PREZIP/), [download help](https://www.transtats.bts.gov/showHelp2.asp).
3. Team's October 10, 2026 project-ideas conversation: three-class probability predictions; three months training and following month testing; incoming aircraft prediction after takeoff but before arrival; do not use the preceding arrival delay; five-airport comparisons.
