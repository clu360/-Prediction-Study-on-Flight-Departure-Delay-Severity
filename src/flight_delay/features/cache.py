"""One-pass, airport-partitioned caches bounded by Arrow batch sizes."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import polars as pl
import pyarrow.dataset as ds
from ..discovery import sha256
from ..pipeline import write_json
from .contracts import paths
from .history import PrefixHistory


def prepare_cache(processed: Path, master: Path, cache: Path) -> Path:
    marker = cache/"cache_manifest.json"
    if marker.exists():
        manifest = json.loads(marker.read_text())
        if not all((cache/p).exists() and sha256(cache/p) == h for p, h in manifest.items()):
            raise ValueError("Feature cache integrity failed; use a fresh output directory")
        print("Verified airport history/schedule/cohort cache", flush=True)
        return cache
    if cache.exists() and any(cache.iterdir()):
        raise ValueError("Incomplete cache exists; preserve it and use a fresh output directory")
    cache.mkdir(parents=True, exist_ok=True)
    for name, source_files, columns, partition in [
        ("cohort", paths(processed, "modeling_cohort"), None, "target_origin"),
        ("history", paths(processed, "history_departure_events"),
         ["record_id", "origin", "reporting_airline", "observed_departure_time_utc", "dep_delay", "departure_severity", "valid_observation"], "origin"),
        ("schedule", sorted(master.glob("bucket-*/*.parquet")), ["origin", "scheduled_departure_utc"], "origin")]:
        dataset = ds.dataset([str(p) for p in source_files], format="parquet")
        reader = dataset.scanner(columns=columns, batch_size=50000, use_threads=False).to_reader()
        ds.write_dataset(reader, str(cache/name), format="parquet", partitioning=[partition],
            basename_template="part-{i}.parquet", max_open_files=512, max_rows_per_file=250000,
            min_rows_per_group=2048, max_rows_per_group=8192,
            use_threads=False, existing_data_behavior="error")
        print(f"Cached {name} by airport", flush=True)
    global_events = pl.scan_parquet(paths(processed, "history_departure_events"), hive_partitioning=False).filter(
        pl.col("valid_observation")).select("observed_departure_time_utc", "departure_severity", "valid_observation").collect(engine="streaming")
    global_index = PrefixHistory.from_events(global_events, with_delay=False)
    for name in ["times", "moderate", "severe"]:
        np.save(cache/f"global_{name}.npy", getattr(global_index, name))
    write_json(marker, {str(p.relative_to(cache)).replace("\\", "/"): sha256(p)
                        for p in sorted(cache.rglob("*")) if p.is_file() and p != marker})
    return cache


def global_index(cache: Path) -> PrefixHistory:
    return PrefixHistory(*[np.load(cache/f"global_{name}.npy", mmap_mode="r") for name in ["times", "moderate", "severe"]])


def airport_frame(cache: Path, table: str, airport: str, schema: dict | None = None) -> pl.DataFrame:
    files = sorted((cache/table/airport).glob("*.parquet"))
    if files:
        return pl.read_parquet(files).with_columns(pl.lit(airport).alias("target_origin" if table == "cohort" else "origin"))
    if schema is None:
        raise ValueError(f"Missing airport cache: {table}/{airport}")
    return pl.DataFrame(schema=schema)
