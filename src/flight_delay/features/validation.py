"""Independent artifact, value, range-boundary and direct-history audits."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import polars as pl
from ..pipeline import logical_hash, write_json
from .cache import airport_frame, global_index
from .contracts import scan as source_scan
from .history import DAY
from .views import ALL, CATEGORICAL, G1_MAP, G2_MAP, G3, VIEWS, load_views


def scan(root: Path, table: str) -> pl.LazyFrame:
    return pl.scan_parquet(str(root/table/"year=*"/"month=*"/"*.parquet"), hive_partitioning=False)


def collect(frame: pl.LazyFrame) -> pl.DataFrame:
    return frame.collect(engine="streaming")


def direct_oracle(root: Path, sample: pl.DataFrame) -> dict:
    """Independent boolean-window reductions, not searchsorted/prefix queries."""
    cache = root/"cache"
    global_events = global_index(cache)
    moderate = np.diff(global_events.moderate)
    severe = np.diff(global_events.severe)
    checks = []
    for key, queries in sample.partition_by("target_origin", as_dict=True).items():
        airport = key[0]
        events = airport_frame(cache, "history", airport).filter(pl.col("valid_observation"))
        schedule = airport_frame(cache, "schedule", airport)
        for row in queries.to_dicts():
            t = int(row["prediction_time_utc"].timestamp()*1_000_000)
            mask = (global_events.times >= t-28*DAY) & (global_events.times < t)
            gn, gm, gs = int(mask.sum()), int(moderate[mask].sum()), int(severe[mask].sum())
            ev = events.filter((pl.col("observed_departure_time_utc") >= row["prediction_time_utc"]-__import__('datetime').timedelta(days=28)) &
                (pl.col("observed_departure_time_utc") < row["prediction_time_utc"]))
            own = ev.filter(pl.col("record_id") == row["target_record_id"])
            if own.height:
                gn -= own.height; gm -= own.filter(pl.col("departure_severity") == 1).height
                gs -= own.filter(pl.col("departure_severity") == 2).height
            ev = ev.filter(pl.col("record_id") != row["target_record_id"])
            cell = ev.filter(pl.col("reporting_airline") == row["target_carrier"])
            an, cn = ev.height, cell.height
            am = (ev.filter(pl.col("departure_severity") == 1).height + 20*gm/gn)/(an+20) if gn else None
            asev = (ev.filter(pl.col("departure_severity") == 2).height + 20*gs/gn)/(an+20) if gn else None
            csev = (cell.filter(pl.col("departure_severity") == 2).height+20*asev)/(cn+20) if gn else None
            volume = schedule.filter((pl.col("scheduled_departure_utc") >= row["prediction_time_utc"]-__import__('datetime').timedelta(days=7)) &
                (pl.col("scheduled_departure_utc") < row["prediction_time_utc"])).height
            expected = [am, asev, ev["dep_delay"].mean(), volume, csev, cn]
            for name, value in zip(G3, expected):
                actual = row[name]
                checks.append(actual is None if value is None else actual is not None and np.isclose(actual, value, rtol=1e-11, atol=1e-10))
    return dict(sample_rows=sample.height, scalar_feature_comparisons=len(checks), passed=all(checks))


def audit_features(root: Path, processed: Path | None = None) -> dict:
    audit = root/"audit"
    contract = json.loads((audit/"source_contract.json").read_text())
    processed = processed or Path(contract["processed_dir"])
    signature = json.loads((audit/"run_signature.json").read_text())
    expected_hashes = json.loads((audit/"logical_hashes.json").read_text())
    schemas = json.loads((audit/"feature_schema.json").read_text())
    checks = dict(exact_20_features=True, schema_stable=True, all_files_readable=True,
        logical_hashes_match=True, unique_ids=True, one_to_one_ids=True, labels_match_phase1=True,
        g1_g2_match_phase1=True, utc_timestamps=True, strict_history_cutoff=True,
        finite_or_null_features=True, valid_rates=True, nonnegative_counts=True,
        fallback_hierarchy=True, no_predictor_metadata_or_labels=True, phase1_folds_unchanged=True)
    load_views()
    counts = {table: 0 for table in schemas}
    samples, missing_rows = [], []
    feature_files = sorted((root/"full_features").glob("year=*/month=*/*.parquet"), key=lambda p: (p.stem, str(p.parent)))
    if not feature_files:
        raise ValueError("No feature artifacts")
    current_airport, cohort = None, None
    for i, path in enumerate(feature_files):
        airport = path.stem.removeprefix("part-")
        if airport != current_airport:
            cohort = airport_frame(root/"cache", "cohort", airport)
            current_airport = airport
        frames = {}
        for table in schemas:
            p = root/table/path.relative_to(root/"full_features")
            frame = pl.read_parquet(p)
            frames[table] = frame
            counts[table] += frame.height
            checks["schema_stable"] &= {c: str(t) for c, t in frame.schema.items()} == schemas[table]
            checks["utc_timestamps"] &= all(t.time_zone == "UTC" for t in frame.schema.values() if isinstance(t, pl.Datetime))
            checks["unique_ids"] &= frame["target_record_id"].n_unique() == frame.height
            relative = str(p.relative_to(root)).replace("\\", "/")
            checks["logical_hashes_match"] &= logical_hash(frame, ["target_record_id"]) == expected_hashes.get(relative)
        features, labels, meta = [frames[k] for k in ["full_features", "targets", "metadata"]]
        checks["exact_20_features"] &= features.columns == ["target_record_id"]+ALL
        checks["no_predictor_metadata_or_labels"] &= all(c.startswith(("g1_", "g2_", "g3_")) for c in features.columns[1:])
        ids = features["target_record_id"].sort()
        month = int(path.parent.name.split("=")[1])
        source = cohort.filter(pl.col("source_month") == month)
        checks["one_to_one_ids"] &= all(ids.equals(f["target_record_id"].sort()) for f in [labels, meta, source])
        joined = features.join(source, on="target_record_id", validate="1:1")
        for name, source_name in {**G1_MAP, **G2_MAP}.items():
            checks["g1_g2_match_phase1"] &= joined[name].equals(joined[source_name].rename(name))
        hours = joined["target_crs_dep_local_hour"].to_numpy()*2*np.pi/24
        checks["g1_g2_match_phase1"] &= bool(np.allclose(joined["g1_dep_hour_sin"].to_numpy(), np.sin(hours), atol=1e-14) and
            np.allclose(joined["g1_dep_hour_cos"].to_numpy(), np.cos(hours), atol=1e-14))
        label_match = labels.join(source.select("target_record_id", pl.col("target_class").alias("_class"),
            pl.col("target_dep_delay_min").alias("_delay")), on="target_record_id", validate="1:1")
        checks["labels_match_phase1"] &= label_match.filter(~pl.col("target_class").eq_missing(pl.col("_class")) |
            ~pl.col("target_dep_delay_min").eq_missing(pl.col("_delay"))).height == 0
        checks["strict_history_cutoff"] &= meta.filter(pl.col("history_max_event_utc") >= pl.col("prediction_time_utc")).height == 0
        numeric = [c for c in ALL if c not in CATEGORICAL]
        checks["finite_or_null_features"] &= features.select(pl.any_horizontal([
            (pl.col(c).is_not_null() & ~pl.col(c).is_finite()) for c in numeric]).any()).item() == False
        rate_bad = pl.any_horizontal([(pl.col(c) < 0) | (pl.col(c) > 1) for c in [G3[0], G3[1], G3[4]]])
        checks["valid_rates"] &= features.filter(rate_bad | (pl.col(G3[0])+pl.col(G3[1]) > 1+1e-12)).height == 0
        combined = features.join(meta, on="target_record_id", validate="1:1")
        missing_rows.append(dict(source_month=month, target_origin=airport, rows=features.height,
                                 **{c: features[c].null_count() for c in ALL}))
        checks["nonnegative_counts"] &= combined.filter((pl.col(G3[3]) < 0) | (pl.col(G3[5]) < 0) |
            (pl.col(G3[5]) > pl.col("airport_history_count_28d")) |
            (pl.col("airport_history_count_28d") > pl.col("global_history_count_28d"))).height == 0
        checks["fallback_hierarchy"] &= combined.filter(
            (pl.col("no_global_history") & (pl.col(G3[0]).is_not_null() | pl.col(G3[1]).is_not_null() | pl.col(G3[4]).is_not_null())) |
            (pl.col("no_airport_history") & pl.col(G3[2]).is_not_null()) |
            (pl.col("no_airline_airport_history") &
             ((pl.col(G3[4]).is_null() != pl.col(G3[1]).is_null()) |
              ((pl.col(G3[4])-pl.col(G3[1])).abs() > 1e-12)))).height == 0
        if airport in signature["config"]["evaluation_airports"]:
            samples.append(combined.head(1))
        if i % 500 == 0:
            print(f"Audit: {i+1}/{len(feature_files)} feature shards", flush=True)
    expected = contract["tables"]["modeling_cohort"]["rows"]
    checks["complete_phase1_population"] = all(n == expected for n in counts.values())
    actual_files = {str(p.relative_to(root)).replace("\\", "/") for table in schemas for p in (root/table).rglob("*.parquet")}
    checks["no_stale_partitions"] = actual_files == set(expected_hashes)
    sample = pl.concat(samples).sort("target_record_id")
    sample.write_csv(audit/"feature_samples.csv")
    oracle = direct_oracle(root, sample)
    checks["independent_history_oracle"] = oracle["passed"]
    features, metadata = scan(root, "full_features"), scan(root, "metadata")
    distributions = []
    for name in ALL:
        expressions = [pl.len().alias("rows"), pl.col(name).null_count().alias("missing_count")]
        if name in CATEGORICAL:
            expressions += [pl.col(name).drop_nulls().n_unique().alias("cardinality")]
        else:
            expressions += [getattr(pl.col(name), method)().alias(method) for method in ["min", "max", "mean", "median", "std"]]
            expressions += [pl.col(name).quantile(.01).alias("q01"), pl.col(name).quantile(.99).alias("q99")]
        stats = collect(features.select(expressions)).to_dicts()[0]
        distributions.append(dict(feature=name, dtype=schemas["full_features"][name], group=name[:2].upper(),
                                  missing_fraction=stats["missing_count"]/stats["rows"], **stats))
    pl.DataFrame(distributions, infer_schema_length=None).write_csv(audit/"feature_missingness.csv")
    for dimension, columns in [("month", ["source_month"]), ("airport", ["target_origin"])]:
        coverage = collect(metadata.group_by(columns).agg(pl.len().alias("rows"),
            *[pl.col(c).sum().alias(c) for c in ["no_global_history", "no_airport_history", "no_airline_airport_history", "own_outcome_excluded"]],
            *[getattr(pl.col("airport_history_count_28d"), method)().alias("airport_count_"+method) for method in ["mean", "median", "min", "max"]],
            pl.col("history_available_days").min().alias("history_days_min"),
            pl.col("history_available_days").mean().alias("history_days_mean"))).sort(columns)
        coverage.write_csv(audit/f"history_coverage_by_{dimension}.csv")
        missing = pl.DataFrame(missing_rows).group_by(columns).agg(pl.col("rows").sum(),
            *[pl.col(c).sum() for c in ALL]).sort(columns)
        missing.write_csv(audit/f"feature_missingness_by_{dimension}.csv")
    checks["phase1_folds_unchanged"] = all(__import__('flight_delay.discovery', fromlist=['sha256']).sha256(processed/p) == h
        for p, h in contract["files"].items() if p.startswith("folds/"))
    flags = collect(metadata.select([pl.col(c).sum() for c in [
        "no_global_history", "no_airport_history", "no_airline_airport_history", "own_outcome_excluded",
        "audit_target_departed_before_prediction", "audit_prior_arrived_before_prediction", "audit_prior_arrival_after_target_departure"]])).to_dicts()[0]
    result = dict(counts=counts, source_features=20, feature_views={k: len(v) for k, v in VIEWS.items()},
        acceptance_checks=checks, acceptance_passed=all(checks.values()), history_flags=flags,
        independent_oracle=oracle, feature_distributions=distributions,
        execution=json.loads((audit/"execution.json").read_text()))
    write_json(audit/"phase2_summary.json", result)
    return result
