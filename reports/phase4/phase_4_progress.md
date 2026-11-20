# Phase IV progress journal

> Chronological Phase IV execution record.

**Base:** `feat/phase3-classical-models` at `bf4e9dc`
**Branch:** `feat/phase4-weather-mlp`
**Last recorded remote milestone:** `50c034e5446c59893d2b3052048a1a0724211ce3`
**Last UTC update:** 2026-11-20 20:30:48
**Primary scientific status:** PARTIAL. Real development tuning and weather acquisition are in progress.
**Final protocol:** August–October 2025 fit, frozen November–December 2025 scoring; new weather/weighted comparisons post-hoc
**Sources:** Phase IV specification, locked Phase II/III manifests and dataset fingerprints

## A. Workstream dashboard

| Track | Status (PENDING/RUNNING/VERIFIED/BLOCKED) | Latest verified evidence | Job/attempt | Blocker or next task |
|---|---|---|---|---|
| Environment + upstream dataset audit | VERIFIED | 6,461,609 aligned rows per canonical table | `upstream_all.json` | `upstream_verification.json` |
| A1: IEM station map and raw fetch | VERIFIED | All 352 station responses and archive checksums verified | v1 | `weather_source_inventory.csv` |
| A2: Five G4 feature columns and audit | VERIFIED | 6,461,609 rows and 8,145 direct oracles | `d1a0049cd663e98b5fdb2a891e3107b0a488580d06d8aa31ccb0cf96d3629aef` | `weather_g4_verification.json` |
| A3: Paired XGBoost base/weather fits | VERIFIED | Both frozen models match on train and holdout IDs | `2b5f7992…`, `758cb982…` | Aggregate publication retry pending |
| B1: Base MLP development tuning | VERIFIED | 26 candidate-fold outcomes; winning candidate 6 | `65d54af4c4f26365c27797401ac8a89b79e6ec75c0c1ef2b0a07a631f5edbf7c` | `mlp_tuning_summary.csv` and `mlp_selection.json` |
| B2: MLP A/B (no weather) | VERIFIED | Both arms completed for all three seeds | See signed fit ledger | Matched frozen holdout |
| B3: MLP C/D (with weather) | RUNNING | All three MLP-C seeds passed both months | `5c206e07…`, `5284021b…`, `413b5e1d…` | MLP-D frozen fits underway |
| C: Verified metrics, notebook and RQ summary | RUNNING | Four code cells executed with public aggregates | Aggregate-only notebook check | Phase IV result figures require predictions |

## B. Locked methodology and provenance

- Input data cohort count / ID hash: 6,461,609 checked IDs per canonical table. Twelve monthly ID digests and the upstream byte fingerprint are in `upstream_verification.json`.
- Source, configuration and weather response hashes: NOT_VERIFIED
- XGBoost locked parameters evidence: `reports/phase3/experiment_manifest.json` (verify file SHA)
- MLP candidate list, chosen configuration and epoch-selection protocol: NOT_SELECTED
- Weather location/time policy: 4 `target_origin` + 1 `target_dest`; METAR available-at `valid + 10min < prediction_time`
- Class weights: `N/(3*N_class)` calculated from each training window only
- Train/test: Aug–Oct / Nov–Dec 2025; same fitted state for both holdout months
- Numerical class ordering: `[0,1,2]`

## C. Milestone evidence

| Milestone | Status | Metrics/report SHA | Reproduction command | Runtime / memory | Verified at (UTC) |
|---|---|---|---|---|---|
| Upstream gate | VERIFIED | `2a212c2673ca3da6c70ecef09713ee1e68fe3401acf25f4719520d0a9c794334` | `python scripts/phase4.py audit all` | — | 2026-11-20 19:44:35 |
| Weather fetch | VERIFIED | `0176ab0084e84d7728dcfe2b2b5b5ad345b1e3a9a32ecc2d05c05e4ea789010e` | `python scripts/phase4.py audit-weather` | — | 2026-11-20 20:02:29 |
| G4 complete | VERIFIED | `d1a0049cd663e98b5fdb2a891e3107b0a488580d06d8aa31ccb0cf96d3629aef` | `python scripts/phase4.py build-weather` | Private build log | 2026-11-20 20:21:46 |
| XGB comparison | VERIFIED | `fa7fb14ad96055bb2d00bc8c7b368868b7c3f64ab8a061a306c282f5a1c2d60f` | `python scripts/phase4.py fit XGB-weather` | 121.33 s weather fit | 2026-11-20 20:25:44 |
| Base MLP architecture selected | VERIFIED | `65d54af4c4f26365c27797401ac8a89b79e6ec75c0c1ef2b0a07a631f5edbf7c` | `python scripts/phase4.py tune --device cpu` | See the tuning ledger | 2026-11-20 20:18:09 |
| MLP A | VERIFIED | See event log for three checkpoint digests | `python scripts/phase4.py fit MLP-A --seed SEED` | 18.25 to 34.56 s per fit | 2026-11-20 20:25:44 |
| MLP B | VERIFIED | See event log for three checkpoint digests | `python scripts/phase4.py fit MLP-B --seed SEED` | 18.56 to 19.67 s per fit | 2026-11-20 20:28:04 |
| MLP C | VERIFIED | Three matching checkpoints | `python scripts/phase4.py fit MLP-C --seed SEED` | See final experiment manifest | 2026-11-20 20:30:48 |
| MLP D | PENDING | — | — | — | — |
| Consolidated notebook | PENDING | — | — | — | — |

## D. Append-only chronological event log

### 2026-11-20 19:26:54 UTC — Repository discovery

- Nested repository located. The Phase III notebook already has a working-tree modification.
- Local feature, target, metadata and fold directories exist. Content validation is pending.
- Polars and XGBoost are available. PyTorch is absent.
- GitHub HTTPS from the offline shell failed to connect. No push or PACE submission attempted yet.

### 2026-11-20 19:44:35 UTC — Upstream, station and first development milestones

- Published the journal at `ea1e734` on `feat/phase4-weather-mlp`. The pre-existing Phase III notebook change remains outside the Phase IV commits.
- `python scripts/phase4.py audit all` first failed because the sanitized public Phase II source contract omits private per-file hashes. The rerun checked the matching private contract and passed all twelve months. The complete input SHA-256 is `2a212c2673ca3da6c70ecef09713ee1e68fe3401acf25f4719520d0a9c794334`. The development projection has 5,409,585 rows per table and SHA-256 `3b93d9c0cbfd1a7fc818cd5bbb6242722492c6e0fabc522e58841f40350a92bc`.
- `python scripts/phase4.py stations` verified all 352 origin/destination airport codes. The provisional map initially missed Saipan because its IEM network is `MP__ASOS`. Added that authoritative metadata and verified PGSN's report identity. PBI uses archive station DJT, whose one-day response contains PBI/KPBI report identities. The Atlanta sample returned 24 rows, of which 23 are original reports.
- Weather fetch `v1` first hit a TLS server error after DFW. Existing cached bytes were preserved. The retry recognizes transient TLS errors and revalidates cached files. Current saved station response count is 120.
- IEM returned occasional same-minute records tagged `MADISHF` even with routine/special request parameters. IEM documents this marker as generated high-frequency reports. The causal reader excludes that out-of-scope stream, retains the raw response, and fails on unresolved conflicting wind values at the same instant. [IEM source notes](https://mesonet.agron.iastate.edu/onsite/news.phtml?tag=asos)
- `python scripts/phase4.py tune --device cpu` completed April candidate 1 in 40.55 seconds. Six inner epochs selected epoch 1, then a fresh three-month refit scored April macro-F1 0.605902 and log loss 0.481068. Record SHA-256 `b900171f901c3e11cb839f7f7d7401298ad314aab23d03c64ee681460004547d`. This is development evidence. No architecture lock has been issued.
- PACE SSH timed out during read-only verification. No Slurm jobs or resource claims. Local PyTorch 2.9.1 CPU is in use.
- Focused suite passed 28 tests. The notebook executed all four code cells from a directory containing only public aggregates. Executed private notebook SHA-256 `e855cb2f57e1b138ac54f41e8811961c2f39550c76713cebecee9c0cf21caa7e5`. Only the Phase III reference table and Phase IV status can be verified until final artifacts exist.

### 2026-11-20 19:54:35 UTC — Frozen XGB reference and weather retry

- Base XGBoost fitted 1,663,534 August–October rows in 143.70 seconds with seed 6740, four CPU threads and the locked Phase III parameters. The training ID and fitted transformer hashes match the original Phase III reference.
- The first scoring attempt stopped on a checkpoint adapter error after saving the fit. The reused Phase III reader strips its independently validated checksum. Phase IV now recomputes the same digest from the verified record and recovered the original fitted state. No second fit was needed. Run signature `2b5f7992b84d112380aeccd442d140a14cf4dc04437a18f3183780672cb1aca1`. Booster UBJ state and train-only transformer were identical across both scored months.
- Base XGBoost passed independent holdout ID, label, class assignment, probability and checkpoint validation. It scored 519,971 November flights and 532,053 December flights. Pooled macro-F1 was 0.646663, log loss 0.439373, severe precision 0.877413, severe recall 0.380777, and severe PR-AUC (average precision) 0.572743. Metrics SHA-256 for this snapshot `afbf41bd440fbb129ecbfda37c305bbe7165ed498dcb53759128804ddd00bd22`. The paired weather effect remains pending.
- Weather acquisition reached 285 verified responses before Rochester exposed a malformed relay row containing unrelated station text. The exact response review has SHA-256 `e8067f5a3460b1501ae1074ba82bbb9d331c884876082d11ce16b53fc9b41cc4`. The parser now counts and excludes unverifiable report text from the causal stream while retaining the original station response. Acquisition resumed from the existing verified cache.
- Verified 11 candidate-fold outcomes in the real CPU MLP run against their stored prediction and attempt checksums. Candidate order and stopping rules are unchanged. Reproduction `python scripts/phase4.py tune --device cpu`. Full automated suite passed 143 tests with one skip. Added alignment, holdout and notebook tests, and the expanded Phase IV suite passed 31 tests.

### 2026-11-20 20:02:29 UTC — Full weather request inventory verified

- Verified all 352 station requests from December 30, 2024 UTC through January 2, 2026 UTC exclusive. The inventory contains 3,895,897 raw observations, with observations present for every station. All response checksums and their retrieval timestamps are public aggregates. Raw response checksum-set digest `0176ab0084e84d7728dcfe2b2b5b5ad345b1e3a9a32ecc2d05c05e4ea789010e`.
- `python scripts/phase4.py audit-weather` read the saved request metadata and recomputed the raw response hashes. Source counts include 36,527 undated correction rows, one unverifiable relay row, and 599 instances of distinct same-observation report versions. After explicit out-of-scope high-frequency filtering, the causal G4 build also checks for unresolved conflicts.
- Ran the tracked notebook with only the public aggregate snapshot. The notebook displayed the real G1–G3 XGBoost holdout scores, the available MLP development ledger and the severe-risk figures. The snapshot SHA-256 is `f48f53d0c18584f52a59577620faf4c4cc08a89c1d569532296ffb25760eeba2`.
- Persistent Phase IV completion goal is active. The data, weather, MLP development and frozen-fit checkpoint paths in this journal support recovery after interruption.

### 2026-11-20 20:18:09 UTC — MLP configuration locked before final fitting

- Six candidates completed April, July and October screening. The two finalists completed all seven April–October folds. The full ledger contains 26 verified candidate-fold evaluations and checkpoint digests. The equal-month finalist scores were candidate 6 mean log loss 0.480053 and mean macro-F1 0.612157; candidate 4 mean log loss 0.497185 and mean macro-F1 0.605218.
- Selected candidate 6 with hidden widths [256, 128], learning rate 0.0003, and dropout 0.3. Fixed final fit length is 1 epoch. The seven chronological inner stopping choices for the winning candidate were [1, 2, 1, 1, 1, 1, 1]. Selection lock SHA-256 `65d54af4c4f26365c27797401ac8a89b79e6ec75c0c1ef2b0a07a631f5edbf7c`. Final seeds are 6740, 6741 and 6742 for each arm.
- The initial tuning process stopped after all 18 screening checkpoints on a transient Windows file-sharing violation while opening a February source shard. The bounded restart verified and reused the screening predictions and completed all eight remaining finalist outcomes. Only the April–October development scope entered tuning.
- Reviewed the stored XGB-base replays for 2,048 rows in each frozen month. Both replays recovered the same state and probabilities exactly. Full regression suite passed 146 tests, with one skip and a Windows Jupyter selector warning. Focused Phase IV suite passed 31 tests. Ruff unused-name checks on the Phase IV source and tests passed.

### 2026-11-20 20:21:46 UTC — Five-column whole-cohort G4 verified

- `python scripts/phase4.py build-weather` verified the full 2025 target set, exactly five G4 predictors, strict report timing, airport identities, the Phase I prediction clock, and 12 content-checked Parquet partitions. The causal oracle matched 8,145 row comparisons. G4 signature `d1a0049cd663e98b5fdb2a891e3107b0a488580d06d8aa31ccb0cf96d3629aef`. Its target UTC range was December 31, 2024 at 22:12 through January 1, 2026 at 05:53.
- The mean missing fraction by predictor ranged from 0.169% to 1.526%. The causal coverage audit identified seven airport/month combinations with zero eligible source report days. No flight was dropped. The parser excluded 605 tagged high-frequency rows, 36,527 untimed corrections and one malformed source row. No unresolved duplicate observation times remained in the causal stream.
- Started matched XGB-weather on the verified five-column extension. MLP-A seed 6740 finished. Seed 6741 initially hit a transient Windows source-sharing error while a second process was opening a checked input. The sequential MLP queue retries only recognized sharing errors, with all other failures surfaced. These attempts preserve the same data, seeds and locked fits.

### 2026-11-20 20:25:44 UTC — Paired XGBoost and all MLP-A seeds verified

- XGB-weather finished in 121.33 seconds on CPU. Saved model and transformer hashes match across November and December. Its checkpoint files passed independent readback. All 1,052,024 holdout IDs and labels matched XGB-base. Pooled macro-F1 was 0.648207, log loss 0.436634, severe precision 0.871171, severe recall 0.383870 and severe average precision 0.577189. Weather fit record SHA-256 `fa7fb14ad96055bb2d00bc8c7b368868b7c3f64ab8a061a306c282f5a1c2d60f`. The model comparison is exploratory.
- MLP-A passed both frozen months for seeds 6740, 6741 and 6742. Pooled macro-F1 by seed was 0.624497, 0.618658 and 0.622114. Fit checksums are `d105dd2c169ec10ff50ffc10b62111aa97c5ff541568831f959fb6c2e95a45c6`, `fac4e6c29b786f68d2bea96e0b43d0a3d4cdd0155743b724c6b2ad9d5df8a1da` and `0e10a050b5b44a465a3a0febcff21e13e2448da3c9e9d003e50cb0e8a3106eff`. The saved holdout IDs and labels match the paired XGBoost runs.
- Concurrent publication and MLP fitting collided on Windows source-file sharing. Both XGBoost fits and frozen predictions were already complete. MLP-A seed 6742 recovered on its third bounded attempt. Final public subgroup aggregation will resume after the remaining MLP fits to avoid concurrent file reads.
- Started the separate private G4 age/count diagnostic export. It records station identity, strict-window report counts, report ages and missingness flags outside all predictor views.

### 2026-11-20 20:28:04 UTC — MLP-B three-seed frozen scoring verified

- MLP-B fitted inverse-frequency weights [0.404509, 2.612563, 6.891764] from August–October only. All three frozen seed outcomes passed checkpoint validation. Pooled macro-F1 by seed was 0.454002, 0.429600 and 0.451755. Severe recall by seed was 0.589464, 0.634273 and 0.616545. Fit-checkpoint hashes are `a2d8cf6b1f0b5d4091087b578e40d2513c98eb7fc90e0411430a761b769a358c`, `cfbfa437a7ceb509e9db3023f27af831df82633e48c2fd581a3e5d1d05ea3b00` and `843e50c1a2b353cff296f620b1ac1fd1e7fc2f75b2c593a6797828459dab3183`.
- MLP-C and MLP-D are proceeding from the locked architecture and verified G4 inputs. The separate audit-only window-count tests passed along with all 26 weather tests. This stage does not add model predictors.

### 2026-11-20 20:30:48 UTC — MLP-C and full audit-only weather diagnostics verified

- All three MLP-C seeds completed their fixed-epoch Weather + standard CE fits and both frozen months. Run signatures `5c206e07ba6bb395ef21f0559a21da44546817427f908f1dbcd894d6b5126508`, `5284021baa79669dce1c32879826f7bb650e3cf3d7cb58a948a2cbde60bf46b1` and `413b5e1d4853db9bbd4fa2461fa2da5f82b8e5f8ef085a86d548b74abcb15eee`. MLP-D remains in progress.
- `python scripts/phase4.py audit-g4-details` verified 12 private diagnostic partitions totaling 6,461,609 exact G4 target IDs. The public audit record contains only counts, hashes and schema. Report counts, latest observation ages, station identities and null indicators are outside all predictor views.

## E. Hard blockers and decisions needed

| ID | Type (methodological/operational) | First seen UTC | Impact | Evidence | Decision needed | Status |
|---|---|---|---|---|---|---|
| PACE-1 | operational | 2026-11-20 UTC | Cluster compute unavailable | Strict SSH timed out | Local CPU work continues | OPEN |

## F. Experiment checkpoint ledger

Record one row per signed model/fold/view/seed attempt. Store *private* artifact paths only generically in the committed log and keep concrete private paths in ignored machine-readable records.

| Experiment signature | Status | Dataset/config/code digest | Verified result SHA | Last attempted UTC | Run/recovery note |
|---|---|---|---|---|---|
| XGB-base seed 6740 | verified | Final input `2a212c26…` and locked Phase III configuration | `1d286a45e919151141c5050af839cbebaf5e1329d78edc6bf353273e58ef9515` | 2026-11-20 UTC | Original fitted state recovered and verified for both months |
| April MLP candidate 1, seed 6740 | verified | Development input `3b93d9c0…` | `b900171f901c3e11cb839f7f7d7401298ad314aab23d03c64ee681460004547d` | 2026-11-20 UTC | 40.55 s, checkpoint verified |

## G. Remote sync and monitoring

- **First successful GitHub push:** `ea1e734`
- **Visible remote branch:** `feat/phase4-weather-mlp`
- **PACE workflow/job IDs:** NOT_SUBMITTED
- **Read-only monitoring command:** `python scripts/phase4_monitor.py`
- **Caution:** unpushed local progress is not visible on GitHub; Slurm RUNNING is not scientific completion.

## H. Handoff checklist

- [x] Environment and upstream audit passed for the Phase IV working tables
- [x] G4 five-column, all-2025 causal table and IEM coverage audited
- [x] Paired frozen XGBoost comparison verified
- [x] MLP architecture selection locked from development data
- [ ] All four frozen MLP arms verified, actual seed count disclosed
- [ ] November/December pooled and per-month results and weather/weighting contrasts verified
- [ ] Notebook executes from committed aggregate-only reports
- [ ] Original Phase I/II/III outputs intact; no private data committed
- [ ] Visible pushed branch with progress log, SHA and truthful completion status
