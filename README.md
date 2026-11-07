# Flight departure delay severity — ISyE 6740

Team: Caleb Lu, Ho Young Lee, Christopher Yi.

Phase 1 implements deterministic preprocessing of the twelve 2025 **BTS Reporting Carrier** monthly exports. Phase II constructs three causal feature groups and fold-aware sparse numerical matrices on the unchanged cohort. No classifiers or MLP training are implemented.

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

The complete single-file exports requested for handoff are `data/features/exports/full_features.csv` and `full_features.parquet`: all samples, one ID column and exactly twenty feature columns. Labels and metadata remain separate. Export verification checks every column against the validated partitioned table, preserves Parquet dtypes, independently reads every Parquet batch with Arrow, and writes SHA-256 checksums.
