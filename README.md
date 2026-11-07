# Flight departure delay severity — ISyE 6740

Team: Caleb Lu, Ho Young Lee, Christopher Yi.

Phase 1 implements deterministic preprocessing of the twelve 2025 **BTS Reporting Carrier** monthly exports. One completed target has one three-class label and one explicit predecessor-link decision. No models or Phase 2 feature matrices are implemented.

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
