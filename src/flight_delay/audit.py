"""Independent read-back acceptance checks and aggregate quality reporting."""
from __future__ import annotations
import json
from pathlib import Path
import polars as pl
from .normalize import severity
from .pipeline import TABLES, logical_hash, write_json


def scan(root: Path, table: str) -> pl.LazyFrame:
    return pl.scan_parquet(str(root / table / "year=*" / "month=*" / "*.parquet"), hive_partitioning=False)


def collect(lf: pl.LazyFrame) -> pl.DataFrame:
    return lf.collect(engine="streaming")


def count(lf: pl.LazyFrame) -> int:
    return collect(lf.select(pl.len())).item()


def audit(root: Path, verify_hashes: bool = True) -> dict:
    folder = root / "audit"
    manifest = json.loads((folder / "input_manifest.json").read_text())
    signature = json.loads((folder / "run_signature.json").read_text())
    cfg = signature["config"]
    execution = json.loads((folder / "execution.json").read_text())
    expected_schemas = json.loads((folder / "schema.json").read_text())
    clean, connections, cohort, history, folds = [scan(root, name) for name in TABLES]
    counts = dict(raw=sum(e["row_count"] for e in manifest["inputs"]),
        flights_clean=count(clean), flight_connections=count(connections), modeling_cohort=count(cohort),
        history_departure_events=count(history), folds=count(folds))
    counts["excluded_targets"] = counts["raw"]-counts["flights_clean"]
    counts["valid_link_percentage"] = 100*counts["modeling_cohort"]/counts["flights_clean"] if counts["flights_clean"] else 0
    checks = {}
    checks["twelve_months"] = sorted(e["month"] for e in manifest["inputs"]) == list(range(1, 13)) and not manifest["errors"]
    checks["input_bytes_preserved"] = execution["raw_files_unchanged"]
    checks["raw_clean_accounting"] = (sum(b["master"] for b in execution["buckets"]) == counts["raw"] and
        sum(b["clean"] for b in execution["buckets"]) == counts["flights_clean"])
    checks["valid_labels"] = count(clean.filter((pl.col("target_class") != severity(pl.col("target_dep_delay_min"))) |
        pl.col("target_dep_delay_min").is_null() | ~pl.col("target_dep_delay_min").is_finite() |
        ~pl.col("target_class").is_in([0, 1, 2]) | pl.col("target_class").is_null() |
        (pl.col("cancelled") != 0) | (pl.col("diverted") != 0))) == 0
    checks["one_connection_per_target_count"] = counts["flights_clean"] == counts["flight_connections"]
    checks["cohort_matches_valid_link_count"] = count(connections.filter(pl.col("link_status") == "valid")) == counts["modeling_cohort"]
    violations = cohort.filter(pl.col("prior_record_id").is_null() | pl.col("prediction_time_utc").is_null() |
        (pl.col("prediction_time_utc") >= pl.col("target_crs_dep_utc")) |
        (pl.col("scheduled_turnaround_min") < 0) | (pl.col("scheduled_turnaround_min") > cfg["max_turnaround_minutes"]) |
        (pl.col("prior_dest") != pl.col("target_origin")) |
        (pl.col("minutes_until_target_crs_dep_at_prediction") <= 0) |
        (pl.col("prediction_time_utc") != pl.col("prior_wheels_off_utc") + pl.duration(minutes=1)))
    checks["valid_cohort_timing_and_continuity"] = count(violations) == 0
    checks["fold_calendar_membership"] = count(folds.filter(
        ((pl.col("role") == "test") & (pl.col("source_month") != pl.col("test_month"))) |
        ((pl.col("role") == "train") & ((pl.col("source_month") >= pl.col("test_month")) |
           (pl.col("source_month") < pl.col("test_month") - 3))) |
        (pl.col("is_final_holdout") != (pl.col("test_month") >= 11)))) == 0
    # Validate uniqueness/IDs/physical adjacency one deterministic tail bucket at a
    # time; never load the year's labels or wide source rows into one matrix.
    checks.update(unique_keys=True, connection_target_ids_match=True, cohort_target_ids_match=True,
                  immediate_predecessor_same_tail=True, history_observations_valid=True,
                  all_parquet_readable_schema_stable=True, utc_schema=True, logical_hashes_match=True)
    intermediate = root.parent / "interim" / signature["run_key"]
    for bucket in range(signature["buckets"]):
        master_files = sorted((intermediate / f"bucket-{bucket:02}").glob("*.parquet"))
        if not master_files:
            continue
        master = pl.read_parquet(master_files, columns=["record_id", "tail_number", "scheduled_departure_utc",
            "eligible_target", "dest", "origin", "cancelled", "diverted", "dep_delay", "actual_wheels_off_utc"])
        master = master.sort(["tail_number", "scheduled_departure_utc", "record_id"], nulls_last=True).with_columns(
            pl.col("record_id").shift(1).over("tail_number").alias("expected_prior_id"))
        frames = {}
        for table in TABLES:
            paths = sorted((root / table).glob(f"year=*/month=*/part-{bucket:02}.parquet"))
            if not paths:
                continue
            frames[table] = pl.read_parquet(paths)
            df = frames[table]
            keys = ["target_record_id", "fold_id", "role"] if table == "folds" else [
                "record_id" if "record_id" in df else "target_record_id"]
            checks["unique_keys"] &= df.select(keys).n_unique() == df.height
        a = frames["flights_clean"]["record_id"].sort()
        b = frames["flight_connections"]["target_record_id"].sort()
        checks["connection_target_ids_match"] &= a.equals(b.rename("record_id"))
        valid = frames["flight_connections"].filter(pl.col("link_status") == "valid")
        checks["cohort_target_ids_match"] &= frames.get("modeling_cohort", pl.DataFrame({"target_record_id": []}))[
            "target_record_id"].sort().equals(valid["target_record_id"].sort())
        selected = valid.select("target_record_id", "prior_record_id").join(
            master.select(pl.col("record_id").alias("target_record_id"), "expected_prior_id", "tail_number", "origin"),
            on="target_record_id", how="left").join(master.select(pl.col("record_id").alias("prior_record_id"),
                pl.col("tail_number").alias("prior_tail"), pl.col("scheduled_departure_utc").alias("prior_schedule"),
                pl.col("dest").alias("prior_dest"), "cancelled", "diverted", "dep_delay", "actual_wheels_off_utc"),
                on="prior_record_id", how="left")
        checks["immediate_predecessor_same_tail"] &= selected.filter(
            pl.col("tail_number").is_null() | (pl.col("tail_number") != pl.col("prior_tail")) |
            (pl.col("expected_prior_id") != pl.col("prior_record_id")) |
            (pl.col("origin") != pl.col("prior_dest")) | (pl.col("cancelled") != 0) |
            (pl.col("diverted") != 0) | pl.col("dep_delay").is_null() |
            pl.col("actual_wheels_off_utc").is_null()).height == 0
        if "history_departure_events" in frames:
            h = frames["history_departure_events"]
            checks["history_observations_valid"] &= h.filter(~pl.col("valid_observation") |
                pl.col("observed_departure_time_utc").is_null() | pl.col("dep_delay").is_null() |
                pl.col("gate_departure_clock_ok").eq(False).fill_null(False) |
                (pl.col("cancelled") != 0)).height == 0
        del master, frames
    expected_hashes = json.loads((folder / "logical_hashes.json").read_text())
    actual_files = {str(p.relative_to(root)).replace("\\", "/") for name in TABLES
                    for p in (root / name).rglob("*.parquet")}
    checks["no_stale_or_missing_partitions"] = actual_files == set(expected_hashes)
    for relative in sorted(actual_files):
        path = root / relative
        table = relative.split("/")[0]
        df = pl.read_parquet(path)
        checks["all_parquet_readable_schema_stable"] &= {c: str(t) for c, t in df.schema.items()} == expected_schemas[table]
        checks["utc_schema"] &= all(t.time_zone == "UTC" for t in df.schema.values() if isinstance(t, pl.Datetime))
        if verify_hashes:
            keys = ["target_record_id", "fold_id", "role"] if table == "folds" else [
                "record_id" if "record_id" in df else "target_record_id"]
            checks["logical_hashes_match"] &= logical_hash(df, keys) == expected_hashes.get(relative)
    reasons = collect(connections.group_by("link_status", "link_reason").agg(pl.len().alias("count")).sort("link_reason"))
    reasons.write_csv(folder / "exclusion_counts.csv")
    classes = {}
    for table, frame in [("flights_clean", clean), ("modeling_cohort", cohort)]:
        dist = collect(frame.group_by("target_class").agg(pl.len().alias("count")).sort("target_class"))
        total = dist["count"].sum()
        classes[table] = dist.with_columns((100*pl.col("count")/total).alias("percentage")).to_dicts()
    target_metadata = clean.select(pl.col("record_id").alias("target_record_id"), "source_month", "origin", "reporting_airline", "target_class")
    coverage = connections.select("target_record_id", "link_status").join(target_metadata, on="target_record_id", how="left")
    for group, columns in [("month", ["source_month"]), ("airport", ["origin"]), ("carrier", ["reporting_airline"])]:
        report = collect(coverage.group_by(columns).agg(pl.len().alias("eligible"),
            (pl.col("link_status") == "valid").sum().alias("valid_links"))
            .with_columns((100*pl.col("valid_links")/pl.col("eligible")).alias("valid_link_percentage")).sort(columns))
        report.write_csv(folder / f"coverage_by_{group}.csv")
    airport_counts = collect(coverage.filter(pl.col("origin").is_in(cfg["evaluation_airports"]))
        .group_by("source_month", "origin", "link_status", "target_class").agg(pl.len().alias("count"))
        .sort(["source_month", "origin", "link_status", "target_class"]))
    airport_counts.write_csv(folder / "five_airport_monthly_class_counts.csv")
    present = collect(cohort.select("target_origin").unique())["target_origin"].to_list()
    source_origins = collect(clean.select("origin").unique())["origin"].to_list()
    checks["five_evaluation_airports_present"] = all(a in present for a in cfg["evaluation_airports"] if a in source_origins)
    takeoff = collect(cohort.group_by("takeoff_time_source").agg(pl.len().alias("count"))).to_dicts()
    sensitivity = count(cohort.filter(pl.col("scheduled_turnaround_min") <= cfg["sensitivity_turnaround_minutes"]))
    audit_flags = collect(connections.filter(pl.col("link_status") == "valid").select([
        pl.col(c).sum() for c in ["audit_target_departed_before_prediction", "audit_prior_arrived_before_prediction",
                                 "audit_prior_arrival_after_target_departure"]])).to_dicts()[0]
    anomaly_fields = ["audit_target_departed_before_prediction", "audit_prior_arrived_before_prediction",
                      "audit_prior_arrival_after_target_departure"]
    collect(connections.filter((pl.col("link_status") == "valid") &
            pl.any_horizontal([pl.col(c) for c in anomaly_fields]))).write_csv(folder / "retrospective_anomalies.csv")
    examples = pl.read_csv(folder / "linkage_examples.csv")
    checks["review_examples_count"] = min(50, counts["flights_clean"]) <= examples.height <= 100
    target_months = collect(clean.select("source_month").unique())["source_month"].to_list()
    checks["review_examples_month_coverage"] = set(examples["source_month"].to_list()) == set(target_months)
    normalization = json.loads((folder / "normalization_quality.json").read_text())
    target_exclusions = {}
    for month in normalization:
        for reason, n in month["target_exclusion_counts"].items():
            target_exclusions[reason] = target_exclusions.get(reason, 0) + n
    checks["target_exclusion_accounting"] = (sum(target_exclusions.values()) == counts["raw"] and
        target_exclusions.get("eligible", 0) == counts["flights_clean"])
    result = dict(counts=counts, classes=classes, exclusion_counts=reasons.to_dicts(),
        takeoff_sources_primary=takeoff, sensitivity_12h_count=sensitivity,
        retrospective_valid_link_audit_flags=audit_flags, acceptance_checks=checks,
        acceptance_passed=all(checks.values()), execution=execution,
        missing_tail=sum(m["missing_tail"] for m in normalization),
        invalid_flags=sum(m["invalid_flags"] for m in normalization),
        target_exclusion_counts=target_exclusions,
        exact_duplicate_raw_rows=sum(e["exact_duplicate_rows"] for e in manifest["inputs"]))
    write_json(folder / "preprocessing_summary.json", result)
    report = ["# Phase 1 preprocessing audit", "", f"Acceptance passed: **{result['acceptance_passed']}**", "",
        "## Record flow", "", "```json", json.dumps(counts, indent=2), "```", "",
        "## Class distributions", "", "```json", json.dumps(classes, indent=2), "```", "",
        "## Link reasons", "", reasons.__str__(), "", "## Retrospective flags (do not select features)", "",
        json.dumps(audit_flags, indent=2), "", f"12-hour sensitivity cohort: {sensitivity:,}", "",
        "## Acceptance checks", "", *[f"- {k}: {v}" for k, v in checks.items()], "",
        "## Interpretation and limitations", "",
        "Target departure and prior arrival audit failures are flagged, not used to silently change the schedule-selected cohort.",
        "A Phase 2 operational deployment must decide how to handle retrospective violations of the intended forecast window.",
        "WheelsOff without TaxiOut uses the uniquely nearest local calendar date, requires nonnegative taxi, and rejects unresolved DST folds.",
        "Unobserved ferry movements, retrospective tail assignments, and December 2024 predecessors remain unobserved.",
        "The 100 local examples are reproducibly sampled for human review; automatic checks do not constitute human certification.",
        "Full schema: schema.json. Input hashes/coverage: input_manifest.json. Timezones: airport_timezones.json.",
        "Normalization diagnostics: normalization_quality.json. Monthly airport/class counts and coverage tables are adjacent CSVs."]
    (folder / "preprocessing_audit.md").write_text("\n".join(report), encoding="utf-8")
    return result
