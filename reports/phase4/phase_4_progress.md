# Phase IV progress journal

> Chronological Phase IV execution record.

**Base:** `feat/phase3-classical-models` at `bf4e9dc`
**Branch:** `feat/phase4-weather-mlp`
**Last recorded remote milestone:** `19d35daf3148b0e839a5219683152e31a4735381`
**Last UTC update:** 2026-11-20 20:02:29
**Primary scientific status:** PARTIAL. Real development tuning and weather acquisition are in progress.
**Final protocol:** August–October 2025 fit, frozen November–December 2025 scoring; new weather/weighted comparisons post-hoc
**Sources:** Phase IV specification, locked Phase II/III manifests and dataset fingerprints

## A. Workstream dashboard

| Track | Status (PENDING/RUNNING/VERIFIED/BLOCKED) | Latest verified evidence | Job/attempt | Blocker or next task |
|---|---|---|---|---|
| Environment + upstream dataset audit | VERIFIED | 6,461,609 aligned rows per canonical table | `upstream_all.json` | `upstream_verification.json` |
| A1: IEM station map and raw fetch | VERIFIED | All 352 station responses and archive checksums verified | v1 | `weather_source_inventory.csv` |
| A2: Five G4 feature columns and audit | RUNNING | Source gate passed | Local G4 build | Independent oracle and full ID audit underway |
| A3: Paired XGBoost base/weather fits | RUNNING | XGB-base fit and both frozen scores verified | `2b5f7992…` | G4 arm pending |
| B1: Base MLP development tuning | RUNNING | 11/26 candidate-fold checkpoints verified | `13e39cfc…` | `mlp_development_progress.csv` |
| B2: MLP A/B (no weather) | PENDING | — | — | — |
| B3: MLP C/D (with weather) | PENDING | — | — | — |
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
| G4 complete | PENDING | — | — | — | — |
| XGB comparison | PENDING | — | — | — | — |
| Base MLP architecture selected | PENDING | — | — | — | — |
| MLP A | PENDING | — | — | — | — |
| MLP B | PENDING | — | — | — | — |
| MLP C | PENDING | — | — | — | — |
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
- [ ] G4 five-column, all-2025 causal table and IEM coverage audited
- [ ] Paired frozen XGBoost comparison verified
- [ ] MLP architecture selection locked from development data
- [ ] All four frozen MLP arms verified, actual seed count disclosed
- [ ] November/December pooled and per-month results and weather/weighting contrasts verified
- [ ] Notebook executes from committed aggregate-only reports
- [ ] Original Phase I/II/III outputs intact; no private data committed
- [ ] Visible pushed branch with progress log, SHA and truthful completion status
