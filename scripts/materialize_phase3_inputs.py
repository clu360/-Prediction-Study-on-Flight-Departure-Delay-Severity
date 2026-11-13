"""Byte-identical local working copies of validated inputs; no preprocessing."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import os
from pathlib import Path
import time

from flight_delay.experiments.config import file_hash
from flight_delay.experiments.results import read_json, write_json


def materialize(root: Path, destination: Path, workers: int = 4):
    root, destination = root.resolve(), destination.resolve()
    if os.name == "nt":
        root = Path("\\\\?\\" + str(root).removeprefix("\\\\?\\"))
        destination = Path("\\\\?\\" + str(destination).removeprefix("\\\\?\\"))
    if not destination.is_relative_to(root / "data") or destination in [
        root / "data/processed",
        root / "data/features",
    ]:
        raise ValueError("Working copies must have a separate ignored data directory")
    destination.mkdir(parents=True, exist_ok=True)
    signature = read_json(root / "data/processed/audit/run_signature.json")["run_key"]
    jobs = []
    for label, folders in [
        (
            "processed",
            [
                "flights_clean",
                "flight_connections",
                "modeling_cohort",
                "history_departure_events",
                "folds",
                "audit",
            ],
        ),
        ("features", ["audit"]),
    ]:
        for folder in folders:
            source_root = root / "data" / label / folder
            jobs.extend(
                (p, destination / label / p.relative_to(root / "data" / label))
                for p in source_root.rglob("*")
                if p.is_file()
            )
    master_root = root / "data/interim" / signature
    jobs.extend(
        (p, destination / "interim" / signature / p.relative_to(master_root))
        for p in master_root.glob("bucket-*/*.parquet")
    )
    manifest_path = destination / "materialization.json"
    previous = (
        read_json(manifest_path).get("files", {}) if manifest_path.exists() else {}
    )
    files, errors = {}, []
    started = time.monotonic()

    def copy_one(source, target):
        relative = str(target.relative_to(destination)).replace("\\", "/")
        if (
            relative in previous
            and target.is_file()
            and file_hash(target) == previous[relative]["sha256"]
        ):
            return relative, previous[relative]
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".copying")
        hasher = hashlib.sha256()
        with source.open("rb") as src, partial.open("wb") as dst:
            for block in iter(lambda: src.read(1024 * 1024), b""):
                dst.write(block)
                hasher.update(block)
        sha = hasher.hexdigest()
        if partial.stat().st_size != source.stat().st_size or file_hash(partial) != sha:
            raise ValueError(f"Local copy verification failed: {relative}")
        partial.replace(target)
        return relative, {"sha256": sha, "bytes": target.stat().st_size}

    print(
        f"Materializing {len(jobs):,} original files without changing their bytes",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(copy_one, source, target): source for source, target in jobs
        }
        for i, future in enumerate(as_completed(pending), 1):
            try:
                relative, evidence = future.result()
                files[relative] = evidence
            except (OSError, ValueError) as exc:
                errors.append(
                    {"file": str(pending[future].relative_to(root)), "error": str(exc)}
                )
            if i % 250 == 0 or i == len(jobs):
                write_json(
                    manifest_path,
                    {
                        "complete": i == len(jobs) and not errors,
                        "files": files,
                        "errors": errors,
                    },
                )
                print(
                    f"Copied/verified {i:,}/{len(jobs):,}; errors={len(errors)}; elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
    if errors:
        raise ValueError(
            f"{len(errors)} source files remain unreadable; private materialization manifest records them"
        )
    print("Local inputs ready for independent validation", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("data/phase3_inputs"))
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    materialize(Path(__file__).resolve().parents[1], args.destination, args.workers)


if __name__ == "__main__":
    main()
