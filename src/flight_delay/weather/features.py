"""Causal station-specific range features for exact Phase II target IDs."""

from collections import defaultdict
from datetime import timedelta
import math

import numpy as np
import polars as pl

from .fetch import Archive, bounds
from . import metar
from .metar import LAG
from .stations import airport_codes, load_map
from ..experiments.config import digest, file_hash
from ..experiments.inputs import table_scan
from ..experiments.phase4_inputs import ID
from ..experiments.results import exclusive, read_json, write_frame, write_json
from ..pipeline import logical_hash

MINUTE = np.timedelta64(1, "m")


def station_features(predictions, reports):
    """Use vector searches for binary/max windows and latest-observation wind."""
    times = np.array([row[0].replace(tzinfo=None) for row in reports], dtype="datetime64[us]")
    right = times.searchsorted(predictions, side="left")

    def any_recent(event_times, minutes):
        previous = event_times.searchsorted(predictions, side="left") - 1
        if not len(event_times):
            return np.zeros(len(predictions), dtype=bool)
        return (previous >= 0) & (event_times[np.maximum(previous, 0)] >= predictions - minutes * MINUTE)

    def flag(column, minutes):
        result = np.full(len(predictions), np.nan)
        covered = any_recent(times, minutes)
        positive = np.array([bool(row[column]) for row in reports], dtype=bool)
        result[covered] = any_recent(times[positive], minutes)[covered]
        return result

    category = np.full(len(predictions), np.nan)
    for level in range(4):
        relevant = np.array([row[1] == level for row in reports], dtype=bool)
        category[any_recent(times[relevant], 180)] = level
    wind = np.full(len(predictions), np.nan)
    if len(times):
        # Source order has no meaning. Distinct reports sharing the latest
        # timestamp contribute to windows. An unresolved wind tie fails closed.
        winds_by_time = defaultdict(set)
        for when, _, _, _, speed in reports:
            winds_by_time[when].add(speed)
        ambiguous = np.array([len(winds_by_time[row[0]]) > 1 for row in reports])
        latest = np.maximum(right - 1, 0)
        usable = (right > 0) & (times[latest] >= predictions - 170 * MINUTE)
        if (ambiguous[latest] & usable).any():
            raise ValueError("Simultaneous source reports disagree on the latest wind")
        wind[usable] = np.asarray([row[4] for row in reports], dtype=float)[latest[usable]]
    return category, flag(2, 180), flag(3, 60), wind


def direct_oracle(prediction, origin, destination):
    def window(reports, minutes):
        return [row for row in reports if prediction - timedelta(minutes=minutes) <= row[0] < prediction]

    def worst(reports):
        return max((row[1] for row in window(reports, 180) if row[1] is not None), default=None)

    def indicator(minutes, column):
        rows = window(origin, minutes)
        return int(any(row[column] for row in rows)) if rows else None

    latest = max((row for row in origin if row[0] < prediction), key=lambda row: row[0], default=None)
    wind = latest[4] if latest and prediction - (latest[0] - LAG) <= timedelta(minutes=180) else None
    return worst(origin), indicator(180, 2), indicator(60, 3), wind, worst(destination)


def monthly_queries(inputs, month):
    features = inputs.path("features_dir")
    airports = table_scan(features, "full_features", [month]).select(ID, "g1_origin", "g1_dest")
    meta = table_scan(features, "metadata", [month]).select(ID, "prediction_time_utc")
    cohort = table_scan(inputs.path("processed_dir"), "modeling_cohort", [month]).select(
        ID, "target_origin", "target_dest", "prior_wheels_off_utc",
        pl.col("prediction_time_utc").alias("phase1_cutoff"))
    query = airports.join(meta, on=ID, how="left", validate="1:1").join(
        cohort, on=ID, how="left", validate="1:1").sort(ID).collect()
    invalid = query.filter(
        ~pl.col("g1_origin").eq_missing(pl.col("target_origin"))
        | ~pl.col("g1_dest").eq_missing(pl.col("target_dest"))
        | ~pl.col("prediction_time_utc").eq_missing(pl.col("phase1_cutoff"))
        | ~pl.col("prediction_time_utc").eq_missing(pl.col("prior_wheels_off_utc") + pl.duration(minutes=1))
    )
    if invalid.height:
        raise ValueError("Phase I airport or prediction-clock alignment failed")
    return query.select(ID, "g1_origin", "g1_dest", "prediction_time_utc")


def build(inputs, version):
    all_input_sha = inputs.verify_source("all")
    station_map = inputs.path("station_map")
    mapping = load_map(station_map)
    by_airport = {row["iata"]: row for row in mapping}
    if sorted(by_airport) != airport_codes(inputs):
        raise ValueError("Station map does not cover both canonical airport columns")
    start, end = bounds(inputs.weather)
    archive = Archive(inputs.path("raw_weather_dir"), start, end, version)
    reports, archives = {}, {}
    for row in mapping:
        code = row["iata"]
        reports[code], record = archive.load(row["station"], routine_and_special_only=True)
        matching = set(row.get("metar_identity", row["icao"]).split("|"))
        if set(record["audit"]["metar_identities"]) - matching:
            raise ValueError(f"Unexpected report identity for {code}")
        archives[code] = record
    names = inputs.weather["features"]
    root = inputs.path("weather_dir")
    manifest = root / "audit" / "verified.json"
    context = dict(input_sha256=all_input_sha, code_sha256=file_hash(__file__),
                   config_sha256=file_hash(inputs.path("weather")), station_sha256=file_hash(station_map),
                   parser_sha256=file_hash(metar.__file__),
                   raw={code: record["sha256"] for code, record in archives.items()})
    with exclusive(root):
        if manifest.exists():
            old = read_json(manifest)
            if old["context"] != context:
                raise ValueError("Existing G4 build has a different signature")
            for relative, hashed in old["files"].items():
                if file_hash(root / relative) != hashed:
                    raise ValueError("Existing G4 parquet checksum differs")
            return old
        cover, files, month_hashes, monthly_ranges = [], {}, {}, {}
        audited_rows = 0
        for month in range(1, 13):
            query = monthly_queries(inputs, month)
            if (query["prediction_time_utc"].null_count()
                or query["prediction_time_utc"].min() - timedelta(minutes=190) < start
                or query["prediction_time_utc"].max() >= end):
                raise ValueError("Weather archive date range does not serve flight timestamps")
            month_min, month_max = query["prediction_time_utc"].min(), query["prediction_time_utc"].max()
            monthly_ranges[str(month)] = {"min_utc": month_min.isoformat(), "max_utc": month_max.isoformat(),
                                          "rows": query.height}
            times = query["prediction_time_utc"].to_numpy()
            data = np.full((query.height, 5), np.nan)
            for field, indexes in (("g1_origin", slice(0, 4)), ("g1_dest", slice(4, 5))):
                codes = query[field].to_numpy()
                order = np.argsort(codes, kind="stable")
                grouped = codes[order]
                for ids in np.split(order, np.flatnonzero(grouped[1:] != grouped[:-1]) + 1):
                    code = codes[ids[0]]
                    result = station_features(times[ids], reports[code])
                    data[ids, indexes] = np.column_stack(result if field == "g1_origin" else result[:1])
            for airport, sample in query.with_row_index("_row").group_by("g1_origin"):
                examples = sample.head(2).iter_rows(named=True)
                for row in examples:
                    expected = direct_oracle(row["prediction_time_utc"], reports[airport[0]], reports[row["g1_dest"]])
                    if any((a is None and not math.isnan(b)) or (a is not None and a != b)
                           for a, b in zip(expected, data[row["_row"]])):
                        raise ValueError(f"Independent weather oracle mismatch for {airport[0]}/{month}")
                    audited_rows += 1
            frame = query.select(ID).with_columns(
                [pl.Series(name, data[:, i], nan_to_null=True).cast(pl.Float32) for i, name in enumerate(names)]
            )
            path = root / "g4_features" / "year=2025" / f"month={month:02}" / "part-00.parquet"
            write_frame(path, frame)
            readback = pl.read_parquet(path)
            if not frame.equals(readback) or frame[ID].n_unique() != frame.height:
                raise ValueError("G4 readback or one-to-one alignment failed")
            files[path.relative_to(root).as_posix()] = file_hash(path)
            month_hashes[str(month)] = logical_hash(readback, [ID])
            aligned = frame.join(query.select(ID, "g1_origin", "g1_dest"), on=ID, validate="1:1")
            origins = aligned.group_by("g1_origin").agg(
                pl.len().alias("origin_flights"), *[pl.col(name).null_count().alias(f"{name}_nulls") for name in names[:4]]
            )
            dests = aligned.group_by("g1_dest").agg(
                pl.len().alias("dest_flights"), pl.col(names[4]).null_count().alias(f"{names[4]}_nulls")
            )
            coverages = pl.DataFrame({"airport": sorted(by_airport)}).join(
                origins, left_on="airport", right_on="g1_origin", how="left").join(
                dests, left_on="airport", right_on="g1_dest", how="left").fill_null(0)
            for entry in coverages.to_dicts():
                code = entry["airport"]
                days = archives[code]["causal_stream_audit"]["days"]
                cover.append(dict(month=month, archive_station=by_airport[code]["station"],
                                  report_days_utc=sum(day.startswith(f"2025-{month:02}") for day in days), **entry))
            print(f"G4 month {month:02}: {frame.height} aligned target flights", flush=True)
        coverage = inputs.path("reports_dir") / "weather_coverage.csv"
        write_frame(coverage, pl.DataFrame(cover).sort("month", "airport"))
        result = dict(context=context, signature=digest(context), files=files, audited_rows=audited_rows,
                      monthly_logical_sha256=month_hashes, coverage_sha256=file_hash(coverage),
                      prediction_time_ranges=monthly_ranges,
                      archive_counts={k: {f: v for f, v in record["causal_stream_audit"].items() if f != "days"}
                                      for k, record in archives.items()})
        write_json(manifest, result)
        return result
