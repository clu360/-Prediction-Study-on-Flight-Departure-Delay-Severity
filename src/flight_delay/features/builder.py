"""Build the three joined-by-ID artifacts, preserving all Phase I samples."""
from __future__ import annotations
import hashlib
import gc
import shutil
from importlib.metadata import version
import json
from pathlib import Path
import time
import psutil
import polars as pl
import pyarrow as pa
from ..discovery import sha256
from ..pipeline import logical_hash, write_json
from .cache import airport_frame, global_index, prepare_cache
from .config import FeatureConfig
from .contracts import code_hash, reconcile, scan
from .history import history_features
from .incoming import incoming_features
from .schedule import schedule_features
from .views import ALL, VIEWS, load_views

FLAGS = ["audit_target_departed_before_prediction", "audit_prior_arrived_before_prediction", "audit_prior_arrival_after_target_departure"]


def build(processed: Path, out: Path, cfg: FeatureConfig, schema_path: Path = Path("reports/schema.json"),
          views_path: Path = Path("configs/feature_groups.yaml"), reuse_cache: Path | None = None) -> dict:
    start = time.monotonic()
    load_views(views_path)
    source = reconcile(processed, schema_path)
    signature = dict(input_sha256=source["input_sha256"], code_sha256=code_hash(), config=cfg.dictionary(),
        versions={p: version(p) for p in ["polars", "pyarrow", "numpy", "scikit-learn", "scipy"]})
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    marker = out/"audit/run_signature.json"
    if marker.exists():
        if json.loads(marker.read_text())["run_key"] != key:
            raise ValueError("Feature output signature differs; choose a fresh --out-dir")
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Refusing to overwrite an unknown feature output directory")
    (out/"audit").mkdir(parents=True, exist_ok=True)
    (out/"audit/SUCCESS.json").unlink(missing_ok=True)
    write_json(marker, dict(run_key=key, **signature))
    write_json(out/"audit/source_contract.json", source)
    if reuse_cache is not None and not (out/"cache").exists():
        original = json.loads((reuse_cache.parent/"audit/source_contract.json").read_text())
        if original["input_sha256"] != source["input_sha256"] or not (reuse_cache/"cache_manifest.json").exists():
            raise ValueError("Cannot reuse a cache from different source bytes or an incomplete cache")
        shutil.copytree(reuse_cache, out/"cache")
    cache = prepare_cache(processed, Path(source["master_dir"]), out/"cache")
    gc.collect()
    pa.default_memory_pool().release_unused()
    index = global_index(cache)
    anomalies = scan(processed, "flight_connections").filter(pl.any_horizontal([pl.col(c) for c in FLAGS])).select(
        "target_record_id", *FLAGS).collect(engine="streaming")
    history_schema = scan(processed, "history_departure_events").select(
        "record_id", "origin", "reporting_airline", "observed_departure_time_utc", "dep_delay", "departure_severity", "valid_observation").collect_schema()
    schedule_schema = {"origin": pl.String, "scheduled_departure_utc": pl.Datetime("us", "UTC")}
    hashes, schemas, airports = {}, {}, sorted(p.name for p in (cache/"cohort").iterdir() if p.is_dir())
    counts = {k: 0 for k in ["full_features", "targets", "metadata"]}
    for i, airport in enumerate(airports):
        cohort = airport_frame(cache, "cohort", airport).sort(["prediction_time_utc", "target_record_id"])
        events = airport_frame(cache, "history", airport, history_schema)
        schedule = airport_frame(cache, "schedule", airport, schedule_schema)
        g3, diagnostics = history_features(cohort, events, schedule, index, cfg)
        features = schedule_features(cohort).join(incoming_features(cohort), on="target_record_id", validate="1:1").join(
            g3, on="target_record_id", validate="1:1").select("target_record_id", *ALL)
        metadata = cohort.select("target_record_id", "prior_record_id", "flight_date", "source_month", "target_origin", "target_carrier",
            "prediction_time_utc", "target_crs_dep_utc", "timestamp_quality_flag", "takeoff_time_source").join(
            diagnostics, on="target_record_id", validate="1:1").join(anomalies, on="target_record_id", how="left", validate="1:1").with_columns(
            [pl.col(c).fill_null(False) for c in FLAGS])
        labels = cohort.select("target_record_id", "target_class", "target_dep_delay_min")
        # Add partition keys temporarily; source_month is NOT an X column.
        keys = cohort.select("target_record_id", "source_month", pl.col("flight_date").dt.year().alias("_year"))
        for name, frame in [("full_features", features), ("targets", labels), ("metadata", metadata)]:
            counts[name] += frame.height
            schemas.setdefault(name, {c: str(t) for c, t in frame.schema.items()})
            if schemas[name] != {c: str(t) for c, t in frame.schema.items()}:
                raise ValueError(f"Phase II schema drift: {name}/{airport}")
            keyed = frame.join(keys.drop("source_month") if "source_month" in frame else keys,
                               on="target_record_id", validate="1:1")
            for period, part in keyed.partition_by(["_year", "source_month"], as_dict=True).items():
                year, month = period
                part = part.drop("_year")
                if name != "metadata":
                    part = part.drop("source_month")
                part = part.select(frame.columns).sort("target_record_id")
                path = out/name/f"year={year}"/f"month={month:02}"/f"part-{airport}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                part.write_parquet(path, compression="zstd", statistics=True)
                hashes[str(path.relative_to(out)).replace("\\", "/")] = logical_hash(part, ["target_record_id"])
        if i % 20 == 0 or i == len(airports)-1:
            print(f"Features {i+1}/{len(airports)} airports: {sum(counts.values())//3:,} targets; RSS {psutil.Process().memory_info().rss/2**20:.0f} MiB", flush=True)
        del cohort, events, schedule, features, metadata, labels, g3, diagnostics
    if any(n != source["tables"]["modeling_cohort"]["rows"] for n in counts.values()):
        raise ValueError("Phase II output counts do not match Phase I")
    write_json(out/"audit/feature_schema.json", schemas)
    write_json(out/"audit/logical_hashes.json", hashes)
    write_json(out/"audit/execution.json", dict(counts=counts, build_seconds=time.monotonic()-start,
        source_cache_reused=reuse_cache is not None,
        process_peak_rss_bytes=getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss)))
    from .validation import audit_features
    result = audit_features(out, processed)
    if not result["acceptance_passed"]:
        raise ValueError("Phase II acceptance failed; inspect phase2_summary.json")
    write_json(out/"audit/SUCCESS.json", dict(run_key=key, acceptance_passed=True))
    return result
