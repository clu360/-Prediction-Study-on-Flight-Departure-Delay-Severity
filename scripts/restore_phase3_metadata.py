"""Restore only support/audit metadata; require all original Phase II hashes.

Predictors remain the checksum-verified CSV values. Original immutable Phase I
events and the unchanged Phase II history routine recover lost metadata storage.
No recovered metadata is accepted unless its original canonical hash matches.
"""

from pathlib import Path
import gc
import os
import polars as pl

from flight_delay.experiments.results import read_json, write_json
from flight_delay.features.builder import FLAGS
from flight_delay.features.cache import prepare_cache, global_index, airport_frame
from flight_delay.features.config import FeatureConfig
from flight_delay.features.contracts import scan
from flight_delay.features.history import history_features
from flight_delay.pipeline import logical_hash


def restore(root):
    root = root.resolve()
    if os.name == "nt":
        root = Path("\\\\?\\" + str(root).removeprefix("\\\\?\\"))
    base = root / "data/phase3_inputs"
    processed, features = base / "processed", base / "features"
    signature = read_json(processed / "audit/run_signature.json")["run_key"]
    cache = prepare_cache(processed, base / "interim" / signature, features / "cache")
    index = global_index(cache)
    expected = read_json(features / "audit/logical_hashes.json")
    anomalies = (
        scan(processed, "flight_connections")
        .filter(pl.any_horizontal([pl.col(c) for c in FLAGS]))
        .select("target_record_id", *FLAGS)
        .collect(engine="streaming")
    )
    history_schema = (
        scan(processed, "history_departure_events")
        .select(
            "record_id",
            "origin",
            "reporting_airline",
            "observed_departure_time_utc",
            "dep_delay",
            "departure_severity",
            "valid_observation",
        )
        .collect_schema()
    )
    schedule_schema = {
        "origin": pl.String,
        "scheduled_departure_utc": pl.Datetime("us", "UTC"),
    }
    airports = sorted(p.name for p in (cache / "cohort").iterdir() if p.is_dir())
    verified, rows = {}, 0
    for i, airport in enumerate(airports, 1):
        cohort = airport_frame(cache, "cohort", airport).sort(
            ["prediction_time_utc", "target_record_id"]
        )
        events = airport_frame(cache, "history", airport, history_schema)
        schedule = airport_frame(cache, "schedule", airport, schedule_schema)
        _, diagnostics = history_features(
            cohort, events, schedule, index, FeatureConfig()
        )
        metadata = (
            cohort.select(
                "target_record_id",
                "prior_record_id",
                "flight_date",
                "source_month",
                "target_origin",
                "target_carrier",
                "prediction_time_utc",
                "target_crs_dep_utc",
                "timestamp_quality_flag",
                "takeoff_time_source",
            )
            .join(diagnostics, on="target_record_id", validate="1:1")
            .join(anomalies, on="target_record_id", how="left", validate="1:1")
            .with_columns([pl.col(c).fill_null(False) for c in FLAGS])
        )
        for (month,), part in metadata.partition_by(
            "source_month", as_dict=True
        ).items():
            part = part.sort("target_record_id")
            relative = f"metadata/year=2025/month={month:02}/part-{airport}.parquet"
            actual = logical_hash(part, ["target_record_id"])
            if actual != expected.get(relative):
                raise ValueError(
                    f"Recovered metadata differs from original: {relative}"
                )
            target = features / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            part.write_parquet(target)
            verified[relative] = actual
            rows += part.height
        if i % 20 == 0 or i == len(airports):
            print(
                f"Metadata: {i}/{len(airports)} airports; {rows:,} original rows hash-verified",
                flush=True,
            )
        del cohort, events, schedule, diagnostics, metadata
        gc.collect()
    if set(verified) != {p for p in expected if p.startswith("metadata/")}:
        raise ValueError("Metadata partition population differs from original")
    write_json(
        features / "metadata_recovery.json",
        dict(
            rows=rows, partitions=len(verified), all_original_logical_hashes_match=True
        ),
    )
    print("All original metadata values and dtypes restored exactly", flush=True)


if __name__ == "__main__":
    restore(Path(__file__).resolve().parents[1])
