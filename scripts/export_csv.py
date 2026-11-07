"""Stream a processed Parquet dataset to CSV and verify counts and columns."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--dataset", default="modeling_cohort", choices=[
        "modeling_cohort", "flights_clean", "flight_connections",
        "history_departure_events", "folds",
    ])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.processed_dir / "csv" / f"{args.dataset}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix(".csv.partial")
    sources = sorted((args.processed_dir / args.dataset).glob("year=*/month=*/*.parquet"))
    if not sources:
        raise ValueError(f"No Parquet inputs for {args.dataset}")
    source = pl.scan_parquet(sources, hive_partitioning=False)
    print(f"Exporting {args.dataset} from {len(sources)} Parquet partitions", flush=True)
    source.sink_csv(pending, datetime_format="%Y-%m-%dT%H:%M:%S%.6f%:z", maintain_order=True)
    exported = pl.scan_csv(pending, infer_schema=False)
    expected = source.select(pl.len()).collect(engine="streaming").item()
    actual = exported.select(pl.len()).collect(engine="streaming").item()
    if actual != expected or exported.collect_schema().names() != source.collect_schema().names():
        raise ValueError("CSV row count or column names do not match Parquet")
    class_counts = None
    if "target_class" in source.collect_schema():
        expected_classes = source.group_by("target_class").len().sort("target_class").collect(engine="streaming")
        actual_classes = exported.with_columns(pl.col("target_class").cast(pl.Int8)).group_by(
            "target_class").len().sort("target_class").collect(engine="streaming")
        if not expected_classes.equals(actual_classes):
            raise ValueError("CSV class counts do not match Parquet")
        class_counts = actual_classes.to_dicts()
    pending.replace(output)
    metadata = dict(dataset=args.dataset, rows=actual, columns=source.collect_schema().names(),
                    size_bytes=output.stat().st_size, class_counts=class_counts,
                    timestamps="ISO 8601 UTC, explicit +00:00 offset", verified=True)
    output.with_suffix(".validation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
