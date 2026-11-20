# Phase IV progress journal

> This is a **template**, not an execution record. Codex must create and commit `reports/phase4/phase_4_progress.md` from it at first implementation commit. Replace placeholders ONLY with observed facts. Maintain append-only historical entries for failed, superseded and completed attempts.

**Base:** `feat/phase3-classical-models` at `bf4e9dc`  
**Last pushed Git SHA:** NOT_PUSHED  
**Last UTC update:** 2026-11-20 19:26:54  
**Primary scientific status:** Preflight in progress  
**Final protocol:** August–October 2025 fit, frozen November–December 2025 scoring; new weather/weighted comparisons post-hoc  
**Sources:** Phase IV specification, locked Phase II/III manifests and dataset fingerprints

## A. Workstream dashboard

| Track | Status (PENDING/RUNNING/VERIFIED/BLOCKED) | Latest verified evidence | Job/attempt | Blocker or next task |
|---|---|---|---|---|
| Environment + upstream dataset audit | PENDING | — | — | Verify files and permissions |
| A1: IEM station map and raw fetch | PENDING | — | — | — |
| A2: Five G4 feature columns and audit | PENDING | — | — | — |
| A3: Paired XGBoost base/weather fits | PENDING | — | — | — |
| B1: Base MLP development tuning | PENDING | — | — | — |
| B2: MLP A/B (no weather) | PENDING | — | — | — |
| B3: MLP C/D (with weather) | PENDING | — | — | — |
| C: Verified metrics, notebook and RQ summary | PENDING | — | — | — |

## B. Locked methodology and provenance

- Input data cohort count / ID hash: NOT_VERIFIED
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
| Upstream gate | PENDING | — | — | — | — |
| Weather fetch | PENDING | — | — | — | — |
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

## E. Hard blockers and decisions needed

| ID | Type (methodological/operational) | First seen UTC | Impact | Evidence | Decision needed | Status |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

## F. Experiment checkpoint ledger

Record one row per signed model/fold/view/seed attempt. Store *private* artifact paths only generically in the committed log and keep concrete private paths in ignored machine-readable records.

| Experiment signature | Status | Dataset/config/code digest | Verified result SHA | Last attempted UTC | Run/recovery note |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

## G. Remote sync and monitoring

- **Last successful push to GitHub:** NOT_PUSHED
- **Visible remote branch/SHA:** NOT_VERIFIED
- **PACE workflow/job IDs:** NOT_SUBMITTED
- **Read-only monitoring command:** NOT_CONFIGURED
- **Caution:** unpushed local progress is not visible on GitHub; Slurm RUNNING is not scientific completion.

## H. Handoff checklist

- [ ] Environment and upstream exact source hashes passed
- [ ] G4 five-column, all-2025 causal table and IEM coverage audited
- [ ] Paired frozen XGBoost comparison verified
- [ ] MLP architecture selection locked from development data
- [ ] All four frozen MLP arms verified, actual seed count disclosed
- [ ] November/December pooled and per-month results and weather/weighting contrasts verified
- [ ] Notebook executes from committed aggregate-only reports
- [ ] Original Phase I/II/III outputs intact; no private data committed
- [ ] Visible pushed branch with progress log, SHA and truthful completion status
