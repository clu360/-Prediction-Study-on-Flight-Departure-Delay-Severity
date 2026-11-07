"""Resource-bounded stage orchestration, caches, and deterministic artifacts."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import time
import psutil
import polars as pl
from .config import Config
from .discovery import discover, read_source, sha256
from .normalize import mapping_provenance, normalize, timezone_map
from .linkage import folds, link_events
from . import outputs

BUCKETS = 32
BATCH_ROWS = 50000
TABLES = ("flights_clean", "flight_connections", "modeling_cohort", "history_departure_events", "folds")


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str, sort_keys=True), encoding="utf-8")


def code_fingerprint() -> str:
    h = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()


def logical_hash(df: pl.DataFrame, keys: list[str]) -> str:
    # Canonical sorted logical values. Arrow IPC buffers can contain unused null
    # backing bytes, so hash value-level NDJSON rather than physical buffers.
    canonical = df.sort(keys).write_ndjson()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def merge_stats(items: list[dict]) -> dict:
    result = {}
    for item in items:
        for k, value in item.items():
            if k == "month":
                result[k] = value
            elif isinstance(value, dict):
                result.setdefault(k, {})
                for col, n in value.items():
                    result[k][col] = result[k].get(col, 0) + (n or 0)
            elif k == "takeoff_sources":
                result.setdefault(k, {})
                for row in value:
                    key = row["takeoff_time_source"]
                    result[k][key] = result[k].get(key, 0) + row["len"]
            else:
                result[k] = result.get(k, 0) + (value or 0)
    return result


def select_review_examples(examples: pl.DataFrame, limit: int = 100) -> pl.DataFrame:
    examples = examples.sort("record_id").group_by(
        ["source_month", "example_stratum", "link_reason"], maintain_order=True).head(2)
    # Round-robin over month and timing stratum; taking the first 100 IDs would
    # overrepresent the first months and fail the review-coverage requirement.
    return examples.sort("record_id").with_columns(
        pl.col("record_id").cum_count().over(["source_month", "example_stratum"]).alias("_sample_rank"))\
        .sort(["_sample_rank", "source_month", "example_stratum", "record_id"]).head(limit).drop("_sample_rank")


def write_partitioned(df: pl.DataFrame, root: Path, year: int, bucket: int,
                      table: str, schemas: dict, hashes: dict) -> None:
    schemas.setdefault(table, {c: str(t) for c, t in df.schema.items()})
    if schemas[table] != {c: str(t) for c, t in df.schema.items()}:
        raise ValueError(f"Schema drift in {table}")
    for key, part in df.partition_by("source_month", as_dict=True).items():
        month = key[0]
        path = root / table / f"year={year}" / f"month={month:02}" / f"part-{bucket:02}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = ["target_record_id", "fold_id", "role"] if table == "folds" else [
            "record_id" if "record_id" in part.columns else "target_record_id"]
        part = part.sort(keys)
        part.write_parquet(path, compression="zstd", statistics=True)
        hashes[str(path.relative_to(root)).replace("\\", "/")] = logical_hash(part, keys)


def run(raw_dir: Path, out: Path, year: int, cfg: Config) -> dict:
    start = time.monotonic()
    audit_dir = out / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    # A marker is only issued after the independent audit. Remove it on every new
    # attempt so a partial failure can never masquerade as a completed run.
    marker = audit_dir / "SUCCESS.json"
    marker.unlink(missing_ok=True)
    entries = discover(raw_dir, year, audit_dir / "input_manifest.json")
    signature = dict(config=asdict(cfg), config_sha256=cfg.fingerprint(), code_sha256=code_fingerprint(),
        dependencies={p: version(p) for p in ["polars", "polars-runtime-32", "pyarrow", "airportsdata", "tzdata"]},
        source_sha256={str(e["month"]): e["sha256"] for e in entries}, buckets=BUCKETS, batch_rows=BATCH_ROWS)
    run_key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    interim = out.parent / "interim" / run_key
    previous_run = audit_dir / "run_signature.json"
    if previous_run.exists() and json.loads(previous_run.read_text()).get("run_key") != run_key:
        # Do not mix stale partitions when changing configuration/source/code.
        raise ValueError("Output directory belongs to another run signature. Choose a fresh --out-dir.")
    write_json(previous_run, dict(run_key=run_key, **signature))
    mapping = timezone_map()
    observed: set[str] = set()
    normalization_stats = []
    peak_rss = 0
    for entry in entries:
        month = entry["month"]
        done = interim / f"month-{month:02}.json"
        if done.exists():
            cached = json.loads(done.read_text())
            if all((interim / p).exists() and sha256(interim / p) == h
                   for p, h in cached["files"].items()):
                normalization_stats.append(cached["stats"])
                observed.update(cached["airports"])
                print(f"Month {month:02}: verified normalized cache", flush=True)
                continue
        raw = read_source(Path(entry["path"]), entry["zip_member"], selected=True)
        # Detect service conflicts across tails before bucketing. Flight number is
        # optional: a missing number cannot establish a service-duplicate key.
        if "flight_number" in raw:
            key = ["flight_date", "reporting_airline", "flight_number", "origin", "dest"]
            raw = raw.with_columns(((pl.len().over(key) > 1) & pl.col("flight_number").is_not_null())
                                  .alias("source_service_duplicate_flag"))
        else:
            raw = raw.with_columns(pl.lit(False).alias("source_service_duplicate_flag"))
        month_stats, month_airports, written = [], set(), {}
        for batch_index, offset in enumerate(range(0, raw.height, BATCH_ROWS)):
            batch = raw.slice(offset, BATCH_ROWS)
            normalized, stats = normalize(batch, entry, cfg, mapping)
            # Row indexes are defined relative to the complete original CSV.
            normalized = normalized.with_columns((pl.col("raw_row_number") + offset).alias("raw_row_number"))
            normalized = normalized.with_columns(pl.concat_str(
                pl.lit(f'{year}-{month:02}-{entry["sha256"][:20]}-'),
                pl.col("raw_row_number").cast(pl.String).str.zfill(7)).alias("record_id"))
            month_airports.update(normalized["origin"].drop_nulls().to_list())
            month_airports.update(normalized["dest"].drop_nulls().to_list())
            normalized = normalized.with_columns((pl.col("tail_number").hash(seed=6740) % BUCKETS)
                .fill_null(0).cast(pl.Int8).alias("tail_bucket"))
            for key, part in normalized.partition_by("tail_bucket", as_dict=True).items():
                path = interim / f"bucket-{key[0]:02}" / f"month-{month:02}-batch-{batch_index:02}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                part.write_parquet(path, compression="zstd", statistics=True)
                written[str(path.relative_to(interim))] = sha256(path)
            month_stats.append(stats)
            peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
            del normalized, batch
        merged = merge_stats(month_stats)
        normalization_stats.append(merged)
        observed.update(month_airports)
        write_json(done, dict(stats=merged, airports=sorted(month_airports), files=written))
        print(f"Month {month:02}: normalized {raw.height:,} rows; RSS {psutil.Process().memory_info().rss/2**20:.0f} MiB", flush=True)
        del raw
    provenance = mapping_provenance(mapping, observed)
    write_json(audit_dir / "airport_timezones.json", provenance)
    write_json(audit_dir / "normalization_quality.json", normalization_stats)
    schemas, hashes, bucket_stats, example_parts = {}, {}, [], []
    for bucket in range(BUCKETS):
        paths = sorted((interim / f"bucket-{bucket:02}").glob("*.parquet"))
        if not paths:
            continue
        events = pl.read_parquet(paths)
        linked = link_events(events, cfg)
        tables = {"flights_clean": outputs.clean(linked), "flight_connections": outputs.connections(linked),
                  "modeling_cohort": outputs.cohort(linked), "history_departure_events": outputs.history(events)}
        tables["folds"] = folds(tables["modeling_cohort"], year)
        for table, df in tables.items():
            write_partitioned(df, out, year, bucket, table, schemas, hashes)
        # Keep local examples only. Deterministic strata include month, cross-zone,
        # overnight and unusual timings, rather than only easy daytime links.
        example_cols = ["record_id", "prior_record_id", "tail_number", "flight_date", "source_month", "origin", "dest",
            "prior_origin", "prior_dest", "scheduled_departure_utc", "prior_scheduled_departure_utc",
            "prior_scheduled_arrival_utc", "prior_actual_gate_departure_utc", "prior_actual_wheels_off_utc",
            "prediction_time_utc", "actual_gate_departure_utc", "scheduled_turnaround_min", "prior_dep_delay",
            "link_reason", "takeoff_day_crossing", "origin_timezone", "dest_timezone", "resolved_taxi_out_min",
            "audit_target_departed_before_prediction", "audit_prior_arrival_after_target_departure"]
        examples = linked.with_columns(
            pl.when(pl.col("resolved_taxi_out_min") > 180).then(pl.lit("unusual_taxi"))
            .when(pl.col("takeoff_day_crossing")).then(pl.lit("day_crossing"))
            .when(pl.col("origin_timezone") != pl.col("dest_timezone")).then(pl.lit("cross_zone"))
            .otherwise(pl.lit("ordinary")).alias("example_stratum"))
        example_parts.append(examples.sort("record_id").group_by(["source_month", "example_stratum", "link_status"], maintain_order=True)
            .head(1).select(example_cols + ["example_stratum"]))
        bucket_stats.append(dict(bucket=bucket, master=events.height, clean=linked.height,
            linked=tables["modeling_cohort"].height,
            conflicts=events.filter(pl.col("source_service_duplicate_flag")).height))
        peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
        print(f"Bucket {bucket+1}/{BUCKETS}: {events.height:,} master, {linked.height:,} targets, {tables['modeling_cohort'].height:,} linked", flush=True)
        del events, linked, tables
    write_json(audit_dir / "schema.json", schemas)
    write_json(audit_dir / "logical_hashes.json", hashes)
    examples = select_review_examples(pl.concat(example_parts))
    examples.write_csv(audit_dir / "linkage_examples.csv")
    write_json(audit_dir / "execution.json", dict(elapsed_seconds=time.monotonic()-start,
        sampled_peak_rss_bytes=peak_rss, process_peak_rss_bytes=getattr(psutil.Process().memory_info(), "peak_wset", peak_rss),
        buckets=bucket_stats, raw_files_unchanged=all(sha256(Path(e["path"])) == e["sha256"] for e in entries)))
    from .audit import audit
    result = audit(out, verify_hashes=True)
    if not result["acceptance_passed"]:
        raise ValueError("Artifact validation failed; inspect preprocessing_summary.json")
    write_json(marker, dict(run_key=run_key, acceptance_passed=True))
    print(json.dumps(result["counts"], indent=2), flush=True)
    return result
