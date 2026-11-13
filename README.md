# Flight departure delay severity — ISyE 6740

Team: Caleb Lu, Ho Young Lee, Christopher Yi.

Phase 1 implements deterministic preprocessing of the twelve 2025 **BTS Reporting Carrier** monthly exports. Phase II constructs three causal feature groups and fold-aware sparse numerical matrices on the unchanged cohort. Phase III adds classical-model tuning and frozen holdout evaluation; actual run status is recorded in `reports/phase3/PHASE3_VALIDATION.md`. MLP training remains out of scope.

## Install and run

Use Python 3.11+ (validated locally with 3.12). From this repository:

```sh
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
python -m flight_delay.cli discover --raw-dir 'path/to/monthly/inputs' --year 2025
python -m flight_delay.cli preprocess --raw-dir 'path/to/monthly/inputs' --out-dir data/processed --year 2025 --config configs/preprocessing.yaml
python -m flight_delay.cli audit --processed-dir data/processed
```

Inputs can remain in their existing folder: nothing moves or modifies them. `data/raw` is an optional ignored destination. CSV filenames are not assumed. ZIP members are inspected individually. Every month must have exactly one authoritative source. The manifest records the actual filenames, hashes, headers, row counts, missing fields, duplicate counts and coverage; discovery stops on mismatches.

### Existing Windows project workspace

When VS Code opens the parent `Project` folder, select `flight-delay/.venv/Scripts/python.exe` as its interpreter. The active environment lives inside the repository; the older parent-level `.venv` is superseded. From the repository folder, use:

```powershell
& .\.venv\Scripts\Activate.ps1
$env:PYTHONPYCACHEPREFIX = Join-Path $PWD '.venv/pycache'
python -m pip check
python -m pytest -q
python -m flight_delay.cli --help
```

The cache prefix keeps bytecode in the new environment and avoids stalled reads of old OneDrive-backed `__pycache__` files. The parent workspace's local VS Code settings configure this for new terminals and the test runner. For a single command, `python -X pycache_prefix=.venv/pycache -m pytest -q` provides the same isolation.

Select `.venv/Scripts/python.exe` when opening the repository itself. If activation is unavailable, invoke `.\.venv\Scripts\python.exe` directly. Moving a workspace between computers can leave an environment's base interpreter, editable installation and command launchers pointing to old locations. Create a fresh environment with an available Python 3.12 interpreter and install the pinned dependencies:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -e '.[dev]'
```

The base dependencies support preprocessing, feature engineering and scikit-learn models. Install the pinned Phase III XGBoost dependency with `python -m pip install -e '.[dev,modeling]'`. Neural-network frameworks remain a later-phase dependency. Installing modeling dependencies does not train models or change the research protocol.

## Outputs

All flight-level artifacts stay local under ignored `data/`. Each table uses `year=2025/month=MM/part-BB.parquet`; BB is a deterministic tail bucket. Read with `polars.scan_parquet('data/processed/modeling_cohort/**/*.parquet', hive_partitioning=False)`, or Arrow's dataset API with hive partitioning. Stored `source_month` is the target local calendar month; hive `month` and `year` are partition metadata.

| Table | Population |
|---|---|
| `flights_clean` | All noncanceled, nondiverted targets with finite signed departure delay |
| `flight_connections` | Exactly one predecessor decision per clean target |
| `modeling_cohort` | Valid immediate predecessors satisfying the configured forecast/schedule rules |
| `history_departure_events` | Valid operated departure observations, including observed departures of diverted flights |
| `folds` | Three-month train/next-month test memberships for Apr–Dec; Nov–Dec flagged final holdout |

`data/interim/<run-signature>/` caches all normalized events, including cancellations/diversions. `data/processed/audit/` contains input/source and timezone manifests, schemas, sorted logical hashes, detailed normalization counts, coverage/class CSVs, 100 local linkage examples, execution measurements, the summary and audit report. `SUCCESS.json` is created only after artifact read-back validation passes. `reports/` contains sanitized aggregate evidence safe for public review.

## Scientific contract

Labels: class 0 for signed `DepDelay < 15`; class 1 for `15 <= DepDelay < 60`; class 2 for `DepDelay >= 60`. Early departures stay in class 0.

Prediction is at prior takeoff + one minute, strictly before target scheduled departure. Aircraft adjacency is selected from the **entire master schedule**, with no skipped cancellations/diversions and no arrival outcomes or target realized outcomes used in selection. Airport continuity and a scheduled turnaround in inclusive [0,1440] minutes are required; a 720-minute sensitivity count is reported.

`target_dep_delay_min` and `target_class` are labels, never predictors. Future phases must use explicit allowlists; the G1/G2 contract is in `outputs.py`. Keep historical features causal as of prediction time. Never use random train/test splits. The workflow in [the full specification](docs/phase1_specification.md) is authoritative.

## Reproducibility and troubleshooting

Dependencies are pinned in `pyproject.toml`. Every run records source, configuration, code, mapping and dependency fingerprints. Rerunning identical code/inputs/config reuses hash-verified monthly intermediates, reconstructs outputs and verifies their canonical sorted logical hashes. Changed signatures require a fresh output folder to prevent stale-file mixing. A failed run returns nonzero and does not issue a success marker. Run `audit` to repeat all output checks independently.

CSV data are read one month at a time, with explicit string parsing, then normalized in 50,000-row batches. The join runs in 32 tail buckets. No full-year wide dense matrix is loaded. Watch disk usage: Parquet intermediates and outputs are intentionally retained for review and reruns. Missing source columns or months are errors, not invitations to substitute data. Unknown timezones and unresolved DST/rotation ambiguities are retained with invalid-link reasons.

Additional reproducibility and independent-engine checks:

```sh
python scripts/verify_reproducibility.py --raw-dir 'path/to/monthly/inputs' --processed-dir data/processed
python scripts/validate_arrow.py --processed-dir data/processed
python scripts/review_anomalies.py --processed-dir data/processed
python scripts/export_review.py --processed-dir data/processed
python scripts/build_validation_report.py
```

The last report command expects the real-data acceptance, reproducibility, Arrow and aggregate quality evidence from the validated run. See [methodology](docs/preprocessing_methodology.md), [data dictionary](docs/data_dictionary.md), and the [validation handoff](reports/VALIDATION.md) for limitations, exact counts, source hashes and Phase 2 decisions. No public raw/processed records are committed. A local `data/preliminary-run/` retains the superseded diagnostic run; `data/processed/` is the official final output location.

## Phase II: features and fold-local matrices

Phase II reads the validated Phase I partitions and its complete normalized master schedule. It never reruns linkage or changes source files. From the repository root after installation:

```sh
pytest -q
python -m flight_delay.cli features --processed-dir data/processed --out-dir data/features --config configs/phase2.yaml
python -m flight_delay.cli features-audit --features-dir data/features
python -m flight_delay.cli feature-fold --fold 2025-04 --feature-set G1_G2_G3 --encoding onehot
```

`data/features/full_features` contains ID + exactly 20 mixed-type features; `targets` holds labels; `metadata` holds timestamps, history support and inherited audit flags. Tables use target-local `year=2025/month=MM` Parquet partitions. Data and caches remain ignored. The four checked configurations in `configs/feature_groups.yaml` are G1 (9), G1_G2 (14), G1_G3 (15) and G1_G2_G3 (20).

G3 uses strictly earlier observed outcomes over 28 days, alpha=20 hierarchical smoothing, and independent seven-day **master scheduled** volume including eventual cancellations/diversions. Earlier test-month observations may update features. Every target's own outcome is excluded. Early missing histories remain null/zero as specified; no future data fills them.

```python
from flight_delay.features.transforms import build_fold
X_train, X_test, y_train, y_test, train_ids, test_ids, transformer = build_fold(
    "2025-04", "G1_G2_G3", encoding="onehot")
```

The existing Phase I fold manifest determines membership. Each transformer fits categorical vocabularies, median imputation and StandardScaler on training rows only, then transforms the test month in batches. X is Float32 sparse CSR with unknown-category support. Use `--no-scale` for an otherwise identical tree-compatible interface. Labels keep class order [0,1,2]. No model is fitted.

Rebuild verification and all-view full April smoke tests:

```sh
python scripts/validate_phase2_run.py --features-dir data/features --processed-dir data/processed
python scripts/export_phase2_tables.py --features-dir data/features
python scripts/export_phase2_review.py --features-dir data/features
```

Changed input/config/feature-code signatures require a fresh output directory; unknown existing output folders and corrupted/incomplete caches are never overwritten. Phase I intermediate master events must remain available. See [Phase II methodology](docs/phase2_methodology.md), [feature dictionary](docs/phase2_feature_dictionary.md), and [full-data validation](reports/PHASE2_VALIDATION.md). All flight-level features, numerical matrices and sample rows stay local; public evidence contains aggregates only.

An existing completed source cache can be reused in a fresh output folder with `--reuse-cache path/to/prior/features/cache`; source fingerprints must match and every cache file hash is verified before use.

## Phase III: classical models and strict frozen holdout

From the repository, with the repaired environment active and `PYTHONPYCACHEPREFIX` set as above:

```powershell
python -m pip install -e '.[dev,modeling,notebook]'
python -m pytest -q
python -m flight_delay.cli phase3-smoke --config configs/phase3.yaml
python -m flight_delay.cli phase3-validate --config configs/phase3.yaml
```

The smoke command uses private, deterministic synthetic fixtures and reduced smoke-only grids. Its scores are never study results.

For a separate real-data CSV readiness check, run `python scripts/check_phase3_csv.py --rows-per-month 2000`. It verifies both complete exports and produces private prediction files from a fixed Jan–Mar/April subset. This diagnostic does not select official hyperparameters or replace the required upstream audit; see [CSV readiness](reports/phase3/CSV_READINESS.md).

Run `python scripts/export_csv_readiness.py` to refresh its aggregate CSV/JSON reports and the notebook's separate readiness table/chart. This does not populate official study results.

Once the real-data validation gate passes:

```powershell
python -m flight_delay.cli phase3-tune --config configs/phase3.yaml --resume
python -m flight_delay.cli phase3-final --config configs/phase3.yaml --selected reports/phase3/selected_configs.json --resume
python -m flight_delay.cli phase3-report --config configs/phase3.yaml
```

`python -u scripts/run_phase3_end_to_end.py --config configs/phase3.yaml` runs validation, resumable full development, frozen holdout scoring, independent result verification, reporting and aggregate-only notebook execution sequentially. It resumes the same checkpoints after a development or frozen-holdout command reaches its wall-time budget; other errors stop with an explicit private execution status. Holdout resumption reloads completed frozen fits and never retunes them.

After an operational interruption with a passed validation gate, add `--resume-validated` to reuse that full audit after rechecking current source/configuration and development-input byte fingerprints. Final scoring separately rechecks the complete holdout input fingerprints. This option cannot bypass a failed or stale gate. If a terminated process left locks, inspect both the output-root and run-directory `.writer.lock` PIDs before removing confirmed stale files.

The current configuration reads hash-equivalent local working inputs under `data/phase3_inputs`. When recovering the existing OneDrive-backed project, the recovery helpers hydrate and hash-check the original monthly files, restore master storage with exact original byte checksums, restore feature/target storage from the validated exports, and require every original metadata logical hash to match. Original source records, linkages, features and folds remain untouched. See the storage-recovery section in [methodology](docs/phase3_methodology.md).

PACE batch execution is documented in [the cluster guide](docs/phase3_pace.md). Its fold/view arrays retain the full grids and data, use isolated verified checkpoints, and feed the original selector and frozen final runner. Account selection is explicit; model computations run on allocated compute nodes.

The baseline fits once per development fold. Logistic regression, random forest and XGBoost use the four existing views. Official candidate grids are 6/8/8, screened on April/July/October; at most two finalists are evaluated on all seven April–October development folds. Selection uses equal-month macro-F1, with deterministic declared tie breakers. No official fitting population is downsampled.

The final estimator and transformer fit once on August–October and remain unchanged for both November and December. **The primary December result does not use the December rolling training fold.** Once holdout scoring starts, persistent locks prevent retuning or changing the selected run. Resumption reloads the saved models. Inspect any interrupted `.writer.lock` PID before removing a stale lock; never delete the primary-holdout lock to retune against already examined months.

All fitted artifacts, flight-level predictions, input facades and logs remain under ignored `data/model_outputs/phase3/`. Public `reports/phase3/` contains only aggregate metrics, candidate/selection evidence, provenance and validation status. Missing real results leave a header-only metric CSV and an explicitly incomplete selection file. OneDrive data and metadata must be readable locally; stale source caches, failed audits, signature mismatches and unsupported hardlinks fail explicitly. Windows long paths are supported for private artifacts.

To execute the analysis notebook without private inputs:

```powershell
python -m ipykernel install --prefix .venv --name flight-delay-phase3 --display-name 'Flight Delay (Phase III)'
python scripts/execute_phase3_notebook.py
```

The source notebook is `notebooks/03_classical_models_analysis.ipynb`. It reads persisted aggregates, fits nothing, and marks unavailable analyses clearly. Executed notebook copies remain private. Rebuild its source with `python scripts/build_phase3_notebook.py`. See [Phase III methodology](docs/phase3_methodology.md) for selection, causal input policy, numerical probability normalization, uncertainty and resource limitations, and [validation](reports/phase3/PHASE3_VALIDATION.md) for completed and incomplete runs.

The complete single-file exports requested for handoff are `data/features/exports/full_features.csv` and `full_features.parquet`: all samples, one ID column and exactly twenty feature columns. Labels and metadata remain separate. Export verification checks every column against the validated partitioned table, preserves Parquet dtypes, independently reads every Parquet batch with Arrow, and writes SHA-256 checksums.
