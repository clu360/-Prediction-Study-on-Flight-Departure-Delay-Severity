"""Recover unavailable master storage only if all original byte hashes match.

Uses the unchanged normalization routine on the original hash-verified monthly
CSVs. Does not rerun linkage or modify cohorts, folds, features, or source bytes.
"""

from pathlib import Path
import argparse
import os
import polars as pl

from flight_delay.config import Config
from flight_delay.discovery import read_source
from flight_delay.experiments.config import file_hash
from flight_delay.normalize import normalize, timezone_map
from flight_delay.experiments.results import read_json, write_json


def restore(root, raw_dir):
    root, raw_dir = root.resolve(), raw_dir.resolve()
    if os.name == "nt":
        root = Path("\\\\?\\" + str(root).removeprefix("\\\\?\\"))
    processed = root / "data/phase3_inputs/processed"
    features = root / "data/phase3_inputs/features"
    signature = read_json(processed / "audit/run_signature.json")
    cfg = Config(**signature["config"])
    entries = read_json(processed / "audit/input_manifest.json")["inputs"]
    contract = read_json(features / "audit/source_contract.json")
    expected = {
        p.removeprefix("master/"): h
        for p, h in contract["files"].items()
        if p.startswith("master/")
    }
    destination = processed.parent / "interim" / signature["run_key"]
    mapping = timezone_map()
    restored, matched = 0, {}
    for entry in entries:
        source = raw_dir / entry["filename"]
        print(f"Checking original source bytes: {entry['filename']}", flush=True)
        if file_hash(source) != entry["sha256"]:
            raise ValueError(
                f"Original monthly input bytes differ: {entry['filename']}"
            )
        month = entry["month"]
        required = {p: h for p, h in expected.items() if f"/month-{month:02}-" in p}
        if all(readable_hash(destination / p) == h for p, h in required.items()):
            matched.update(required)
            print(
                f"Month {month:02}: every original master byte hash already matches",
                flush=True,
            )
            continue
        print(
            f"Restoring only original master storage for month {month:02}", flush=True
        )
        raw = read_source(source, entry["zip_member"], selected=True)
        if "flight_number" in raw:
            keys = [
                "flight_date",
                "reporting_airline",
                "flight_number",
                "origin",
                "dest",
            ]
            raw = raw.with_columns(
                (
                    (pl.len().over(keys) > 1) & pl.col("flight_number").is_not_null()
                ).alias("source_service_duplicate_flag")
            )
        else:
            raw = raw.with_columns(pl.lit(False).alias("source_service_duplicate_flag"))
        for batch_index, offset in enumerate(
            range(0, raw.height, signature["batch_rows"])
        ):
            normalized, _ = normalize(
                raw.slice(offset, signature["batch_rows"]), entry, cfg, mapping
            )
            normalized = normalized.with_columns(
                (pl.col("raw_row_number") + offset).alias("raw_row_number")
            )
            normalized = normalized.with_columns(
                pl.concat_str(
                    pl.lit(f"2025-{month:02}-{entry['sha256'][:20]}-"),
                    pl.col("raw_row_number").cast(pl.String).str.zfill(7),
                ).alias("record_id")
            )
            normalized = normalized.with_columns(
                (pl.col("tail_number").hash(seed=6740) % signature["buckets"])
                .fill_null(0)
                .cast(pl.Int8)
                .alias("tail_bucket")
            )
            for (bucket,), part in normalized.partition_by(
                "tail_bucket", as_dict=True
            ).items():
                relative = f"bucket-{bucket:02}/month-{month:02}-batch-{batch_index:02}.parquet"
                path = destination / relative
                original_hash = expected[relative]
                if readable_hash(path) == original_hash:
                    matched[relative] = original_hash
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                pending = path.with_suffix(".restoring")
                part.write_parquet(pending, compression="zstd", statistics=True)
                if file_hash(pending) != original_hash:
                    raise ValueError(
                        f"Restored master does not match original bytes: {relative}"
                    )
                pending.replace(path)
                matched[relative] = original_hash
                restored += 1
        del raw
        write_json(
            destination / "storage_recovery.json",
            dict(verified_files=len(matched), restored_files=restored, complete=False),
        )
        print(
            f"Month {month:02}: original byte hashes verified; {len(matched):,}/{len(expected):,} files",
            flush=True,
        )
    if set(matched) != set(expected):
        raise ValueError("Original master file population not fully recovered")
    write_json(
        destination / "storage_recovery.json",
        dict(
            verified_files=len(matched),
            restored_files=restored,
            complete=True,
            all_original_byte_hashes_match=True,
        ),
    )
    print(
        "All original master storage byte hashes match; no scientific tables were changed",
        flush=True,
    )


def readable_hash(path):
    try:
        return file_hash(path) if path.is_file() else None
    except OSError:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("../Data/Raw Datasets 2025")
    )
    args = parser.parse_args()
    restore(Path(__file__).resolve().parents[1], args.raw_dir)
