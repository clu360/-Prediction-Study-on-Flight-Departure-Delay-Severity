"""Recover storage partitions from a validated CSV; require every original logical hash.

No feature is recomputed: CSV values are cast back to the recorded schema and
compared with the original Phase II canonical hashes before being accepted.
"""

from pathlib import Path
import json
import os
import pyarrow.csv as acsv
import polars as pl
from flight_delay.experiments.config import file_hash
from flight_delay.experiments.results import write_json
from flight_delay.features.views import ALL
from flight_delay.pipeline import logical_hash


def restore(root, kind="full_features"):
    root = root.resolve()
    if os.name == "nt":
        root = Path("\\\\?\\" + str(root).removeprefix("\\\\?\\"))
    csv = root / (
        "data/features/exports/full_features.csv"
        if kind == "full_features"
        else "data/processed/csv/modeling_cohort.csv"
    )
    expected_csv = (
        (root / "data/features/exports/checksums.sha256").read_text().split()[0]
        if kind == "full_features"
        else file_hash(csv)
    )
    if file_hash(csv) != expected_csv:
        raise ValueError("CSV differs from the validated feature export")
    print("Feature CSV byte checksum matches original validated export", flush=True)
    schema = {"target_record_id": pl.String, **{c: pl.Float64 for c in ALL}}
    for c in ["g1_carrier", "g1_origin", "g1_dest", "g2_prior_origin"]:
        schema[c] = pl.String
    for c in ["g1_day_of_week", "g1_month"]:
        schema[c] = pl.Int8
    for c in ["g3_airport_scheduled_volume_7d", "g3_airline_airport_count_28d"]:
        schema[c] = pl.Int64
    if kind == "targets":
        schema = {
            "target_record_id": pl.String,
            "target_class": pl.Int8,
            "target_dep_delay_min": pl.Float64,
        }
    group_schema = (
        {"source_month": pl.Int8, "target_origin": pl.String}
        if kind == "targets"
        else {}
    )
    input_schema = dict(schema, **group_schema)
    group_columns = (
        ["g1_month", "g1_origin"]
        if kind == "full_features"
        else ["source_month", "target_origin"]
    )
    columns = ["target_record_id", *ALL] if kind == "full_features" else list(schema)
    arrow_schema = pl.DataFrame(schema=input_schema).to_arrow().schema
    destination = root / "data/phase3_inputs/features"
    chunks = root / "data/phase3_inputs/csv_chunks" / kind
    chunks.mkdir(parents=True, exist_ok=True)
    reader = acsv.open_csv(
        csv,
        read_options=acsv.ReadOptions(block_size=16 * 1024**2),
        convert_options=acsv.ConvertOptions(
            column_types=arrow_schema, include_columns=list(input_schema)
        ),
    )
    rows = 0
    groups = set()
    for i, batch in enumerate(reader):
        frame = pl.from_arrow(batch).cast(input_schema)
        for (month, airport), part in frame.partition_by(
            group_columns, as_dict=True
        ).items():
            folder = chunks / f"month={month:02}" / airport
            folder.mkdir(parents=True, exist_ok=True)
            part.select(columns).write_parquet(folder / f"chunk-{i:04}.parquet")
            groups.add((month, airport))
        rows += frame.height
        if i % 10 == 0:
            print(f"Streamed {rows:,} CSV rows", flush=True)
    expected = json.loads(
        (root / "data/features/audit/logical_hashes.json").read_text()
    )
    hashes = {}
    for i, (month, airport) in enumerate(sorted(groups), 1):
        part = (
            pl.read_parquet(
                sorted((chunks / f"month={month:02}" / airport).glob("*.parquet"))
            )
            .select(columns)
            .cast(schema)
        )
        relative = f"{kind}/year=2025/month={month:02}/part-{airport}.parquet"
        actual = logical_hash(part, ["target_record_id"])
        if actual != expected.get(relative):
            raise ValueError(
                f"Recovered values/dtypes differ from original Phase II: {relative}"
            )
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        part.write_parquet(output)
        hashes[relative] = actual
        if i % 500 == 0:
            print(
                f"Verified/restored {i:,}/{len(groups):,} original feature partitions",
                flush=True,
            )
    expected_names = {p for p in expected if p.startswith(kind + "/")}
    if set(hashes) != expected_names:
        raise ValueError("Restored feature partition population differs from original")
    write_json(
        destination / f"{kind}_csv_recovery.json",
        dict(
            rows=rows,
            partitions=len(groups),
            csv_sha256=expected_csv,
            every_original_logical_hash_matches=True,
        ),
    )
    print(
        f"Recovered {rows:,} unchanged feature rows; all {len(groups):,} original hashes match",
        flush=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kind", choices=["full_features", "targets"], default="full_features"
    )
    restore(Path(__file__).resolve().parents[1], parser.parse_args().kind)
