"""Read-only reconciliation and content fingerprints of validated Phase I inputs."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import polars as pl
import pyarrow.parquet as pq
from ..discovery import sha256
from ..pipeline import write_json

TABLES = ("modeling_cohort", "flights_clean", "history_departure_events", "folds", "flight_connections")


def paths(root: Path, table: str) -> list[Path]:
    return sorted((root/table).glob("year=*/month=*/*.parquet"))


def scan(root: Path, table: str) -> pl.LazyFrame:
    return pl.scan_parquet(paths(root, table), hive_partitioning=False)


def reconcile(processed: Path, schema_path: Path) -> dict:
    declared = json.loads(schema_path.read_text())
    local = json.loads((processed/"audit/schema.json").read_text())
    if declared != local:
        raise ValueError("Phase I local schema.json disagrees with repository schema.json")
    summary = json.loads((processed/"audit/preprocessing_summary.json").read_text())
    if not summary["acceptance_passed"] or not (processed/"audit/SUCCESS.json").exists():
        raise ValueError("Phase I acceptance evidence is missing or unsuccessful")
    signature = json.loads((processed/"audit/run_signature.json").read_text())
    master = processed.parent/"interim"/signature["run_key"]
    master_files = sorted(master.glob("bucket-*/*.parquet"))
    if not master_files:
        raise ValueError("Master schedule unavailable: do not substitute observed-departure volume")
    files, tables = {}, {}
    h = hashlib.sha256()
    for table in TABLES:
        part_files = paths(processed, table)
        if not part_files:
            raise ValueError(f"Missing Phase I table: {table}")
        count = 0
        for path in part_files:
            actual = {c: str(t) for c, t in pl.read_parquet_schema(path).items()}
            if actual != declared[table]:
                raise ValueError(f"Phase I schema mismatch: {path}")
            count += pq.ParquetFile(path).metadata.num_rows
            key = str(path.relative_to(processed)).replace("\\", "/")
            files[key] = sha256(path)
            h.update((key+files[key]).encode())
        if count != summary["counts"][table]:
            raise ValueError(f"Phase I row count mismatch: {table}")
        tables[table] = dict(rows=count, partitions=len(part_files), schema_matches=True)
        print(f"Phase I contract: {table}: {count:,} rows, schema verified", flush=True)
    master_rows = 0
    for path in master_files:
        schema = pl.read_parquet_schema(path)
        if schema.get("scheduled_departure_utc") != pl.Datetime("us", "UTC") or "origin" not in schema:
            raise ValueError("Master schedule schema mismatch")
        master_rows += pq.ParquetFile(path).metadata.num_rows
        key = "master/"+str(path.relative_to(master)).replace("\\", "/")
        files[key] = sha256(path)
        h.update((key+files[key]).encode())
    if master_rows != summary["counts"]["raw"]:
        raise ValueError("Master row count does not reconcile with all Phase I source events")
    return dict(input_sha256=h.hexdigest(), files=files, tables=tables, master_rows=master_rows,
                master_dir=str(master.resolve()), processed_dir=str(processed.resolve()),
                phase1_signature=signature["run_key"])


def code_hash() -> str:
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode()); h.update(path.read_bytes())
    return h.hexdigest()
