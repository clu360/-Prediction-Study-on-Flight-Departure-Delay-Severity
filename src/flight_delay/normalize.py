"""Month-at-a-time cleaning and DST-aware absolute event normalization."""
from __future__ import annotations
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import airportsdata
import polars as pl
from .config import Config
from .discovery import OPTIONAL, parse_date, read_source

UTC = pl.Datetime("us", "UTC")
NUMERIC = ["crs_elapsed_time", "distance", "dep_delay", "cancelled", "diverted", "taxi_out",
           "arr_delay", "actual_elapsed_time"]
CLOCKS = ["crs_dep_time", "crs_arr_time", "dep_time", "wheels_off", "arr_time", "wheels_on"]


def duration_minutes(expr: pl.Expr) -> pl.Expr:
    return pl.duration(microseconds=(expr * 60_000_000).round().cast(pl.Int64))


def clock_minutes(expr: pl.Expr) -> pl.Expr:
    """0000 is day-start; 2400 is next-day midnight, never 24:xx."""
    n = expr.cast(pl.Float64, strict=False)
    whole = n.is_finite() & (n == n.floor())
    valid = whole & (((n >= 0) & (n < 2400) & (n % 100 < 60)) | (n == 2400))
    return pl.when(valid).then((n // 100 * 60 + n % 100).cast(pl.Int32)).otherwise(None)


def severity(expr: pl.Expr) -> pl.Expr:
    return (pl.when(expr.is_null() | ~expr.is_finite()).then(None)
            .when(expr < 15).then(0).when(expr < 60).then(1).otherwise(2).cast(pl.Int8))


def timezone_map() -> dict[str, str]:
    return {k: v["tz"] for k, v in airportsdata.load("IATA").items()}


def localize(expr: pl.Expr, zone: str, fold: str = "null") -> pl.Expr:
    return expr.dt.replace_time_zone(zone, ambiguous=fold, non_existent="null").dt.convert_time_zone("UTC")


def clock_delta(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    d = (a % 1440 - b % 1440).abs()
    return pl.min_horizontal(d, 1440 - d)


def clock_at(timestamp: pl.Expr, zone: str) -> pl.Expr:
    local = timestamp.dt.convert_time_zone(zone)
    return local.dt.hour().cast(pl.Int32) * 60 + local.dt.minute().cast(pl.Int32)


def resolve_zone(group: pl.DataFrame, zone: str, cfg: Config) -> pl.DataFrame:
    group = group.with_columns(
        localize(pl.col("_local_dep"), zone, "earliest").alias("_early"),
        localize(pl.col("_local_dep"), zone, "latest").alias("_late"))
    group = group.with_columns((pl.col("_early") != pl.col("_late")).fill_null(False).alias("_dst_ambiguous"))
    # Independent elapsed + destination clock can disambiguate a fall-back schedule.
    parts = []
    for dest_key, sub in group.partition_by("dest_timezone", as_dict=True).items():
        dest_zone = dest_key[0]
        if dest_zone:
            checks = []
            for fold in ("early", "late"):
                arrival = pl.col(f"_{fold}") + duration_minutes(pl.col("crs_elapsed_time"))
                checks.append((clock_delta(clock_at(arrival, dest_zone), pl.col("crs_arr_time_min"))
                               <= cfg.clock_tolerance_minutes).fill_null(False).alias(f"_{fold}_ok"))
            sub = sub.with_columns(checks)
        else:
            sub = sub.with_columns(pl.lit(False).alias("_early_ok"), pl.lit(False).alias("_late_ok"))
        sub = sub.with_columns(
            pl.when(~pl.col("_dst_ambiguous")).then(pl.col("_early"))
            .when(pl.col("_early_ok") & ~pl.col("_late_ok")).then(pl.col("_early"))
            .when(pl.col("_late_ok") & ~pl.col("_early_ok")).then(pl.col("_late"))
            .otherwise(None).cast(UTC).alias("scheduled_departure_utc"))
        sub = sub.with_columns(
            (pl.col("scheduled_departure_utc") + duration_minutes(pl.col("crs_elapsed_time")))
                .alias("scheduled_arrival_utc"))
        if dest_zone:
            sub = sub.with_columns((clock_delta(clock_at(pl.col("scheduled_arrival_utc"), dest_zone),
                  pl.col("crs_arr_time_min")) <= cfg.clock_tolerance_minutes)
                  .alias("scheduled_arrival_clock_ok"))
        else:
            sub = sub.with_columns(pl.lit(None, dtype=pl.Boolean).alias("scheduled_arrival_clock_ok"))
        parts.append(sub)
    group = pl.concat(parts).with_columns(
        (pl.col("scheduled_departure_utc") + duration_minutes(pl.col("dep_delay")))
            .alias("actual_gate_departure_utc"))
    # For absent TaxiOut: test neighboring local dates and retain only a uniquely
    # nearest, nonnegative takeoff. DST folds remain ambiguous without TaxiOut.
    gate_local_midnight = (pl.col("actual_gate_departure_utc").dt.convert_time_zone(zone)
                           .dt.replace_time_zone(None).dt.truncate("1d"))
    group = group.with_columns(gate_local_midnight.alias("_gate_day"))
    # 2400 on an actual clock has the same clock face as 0000; date is resolved
    # against gate departure, independently of the schedule date.
    group = group.with_columns([
        (pl.col("_gate_day") + pl.duration(days=offset)
         + pl.duration(minutes=pl.col("wheels_off_min") % 1440)).alias(f"_candidate_local_{offset+1}")
        for offset in (-1, 0, 1)])
    for i in range(3):
        group = group.with_columns(
            localize(pl.col(f"_candidate_local_{i}"), zone, "earliest").alias(f"_candidate_{i}"),
            localize(pl.col(f"_candidate_local_{i}"), zone, "latest").alias(f"_candidate_late_{i}"))
        group = group.with_columns((pl.col(f"_candidate_{i}") - pl.col("actual_gate_departure_utc"))
                .dt.total_microseconds().abs().alias(f"_distance_{i}"))
    # If same-day clock is nonexistent, do not silently substitute a neighboring day.
    group = group.with_columns(pl.min_horizontal([f"_distance_{i}" for i in range(3)]).alias("_nearest"))
    group = group.with_columns(
        pl.coalesce([pl.when(pl.col(f"_distance_{i}") == pl.col("_nearest"))
                    .then(pl.col(f"_candidate_{i}")) for i in range(3)]).alias("_nearest_utc"),
        pl.sum_horizontal([(pl.col(f"_distance_{i}") == pl.col("_nearest")).cast(pl.Int8)
                           for i in range(3)]).alias("_nearest_count"),
        pl.any_horizontal([(pl.col(f"_distance_{i}") == pl.col("_nearest")) &
                           (pl.col(f"_candidate_{i}") != pl.col(f"_candidate_late_{i}"))
                           for i in range(3)]).alias("_takeoff_ambiguous"),
        (pl.col("actual_gate_departure_utc") + duration_minutes(pl.col("taxi_out"))).alias("_taxi_expected"))
    observed_good = (pl.col("wheels_off_min").is_not_null() & pl.col("actual_gate_departure_utc").is_not_null())
    with_taxi = (pl.col("taxi_out").is_not_null() & (pl.col("taxi_out") >= 0))
    taxi_agrees = (clock_delta(clock_at(pl.col("_taxi_expected"), zone), pl.col("wheels_off_min"))
                   <= cfg.clock_tolerance_minutes)
    nearest_good = ((pl.col("_nearest_count") == 1) & ~pl.col("_takeoff_ambiguous") &
                    (pl.col("_nearest_utc") >= pl.col("actual_gate_departure_utc")) &
                    pl.col("_candidate_1").is_not_null())
    group = group.with_columns(
        pl.when(observed_good & with_taxi & taxi_agrees).then(pl.col("_taxi_expected"))
        .when(observed_good & ~with_taxi & nearest_good).then(pl.col("_nearest_utc"))
        .when(pl.col("wheels_off_min").is_null() & with_taxi).then(pl.col("_taxi_expected"))
        .otherwise(None).cast(UTC).alias("actual_wheels_off_utc"))
    group = group.with_columns(
        pl.when(pl.col("actual_wheels_off_utc").is_null()).then(pl.lit("invalid"))
        .when(pl.col("wheels_off_min").is_null()).then(pl.lit("derived"))
        .otherwise(pl.lit("parsed")).alias("takeoff_time_source"),
        (clock_delta(clock_at(pl.col("actual_gate_departure_utc"), zone), pl.col("dep_time_min"))
            <= cfg.clock_tolerance_minutes).alias("gate_departure_clock_ok"),
        (pl.col("actual_wheels_off_utc") - pl.col("actual_gate_departure_utc"))
            .dt.total_seconds().truediv(60).alias("resolved_taxi_out_min"),
        (pl.col("actual_wheels_off_utc").dt.convert_time_zone(zone).dt.date() != pl.col("flight_date"))
            .alias("takeoff_day_crossing"))
    # An independently reported gate clock can reveal a DST-crossing delay-clock
    # inconsistency. Do not silently offset the locked schedule+DepDelay formula.
    group = group.with_columns(
        pl.when(pl.col("gate_departure_clock_ok") == False).then(None)
            .otherwise(pl.col("actual_wheels_off_utc")).cast(UTC).alias("actual_wheels_off_utc"),
        pl.when(pl.col("gate_departure_clock_ok") == False).then(pl.lit("invalid"))
            .otherwise(pl.col("takeoff_time_source")).alias("takeoff_time_source"))
    return group


def normalize(raw: pl.DataFrame, entry: dict, cfg: Config, mapping: dict[str, str]) -> tuple[pl.DataFrame, dict]:
    df = raw.with_row_index("raw_row_number")
    for col in OPTIONAL | set(NUMERIC) | set(CLOCKS):
        if col not in df:
            df = df.with_columns(pl.lit(None, dtype=pl.String).alias(col))
    df = df.with_columns([pl.col(c).str.strip_chars().str.to_uppercase().replace("", None).alias(c)
                          for c in ["tail_number", "origin", "dest", "reporting_airline", "flight_number"]])
    # Retain invalid/null counts independently of downstream target eligibility.
    numeric_invalid = {c: df.select((pl.col(c).is_not_null() &
        (~pl.col(c).cast(pl.Float64, strict=False).is_finite()).fill_null(True)).sum()).item()
        for c in NUMERIC}
    df = df.with_columns([pl.when(pl.col(c).cast(pl.Float64, strict=False).is_finite())
        .then(pl.col(c).cast(pl.Float64, strict=False)).otherwise(None).alias(c) for c in NUMERIC])
    df = df.with_columns(parse_date().alias("flight_date"),
        *[clock_minutes(pl.col(c)).alias(c + "_min") for c in CLOCKS],
        pl.lit(entry["month"], dtype=pl.Int8).alias("source_month"),
        pl.lit(entry["month"], dtype=pl.Int8).alias("raw_source_month"),
        pl.lit(entry["filename"]).alias("raw_source_file"),
        pl.concat_str(pl.lit(f'{entry["year"]}-{entry["month"]:02}-{entry["sha256"][:20]}-'),
                      pl.col("raw_row_number").cast(pl.String).str.zfill(7)).alias("record_id"),
        pl.col("origin").replace_strict(mapping, default=None, return_dtype=pl.String).alias("origin_timezone"),
        pl.col("dest").replace_strict(mapping, default=None, return_dtype=pl.String).alias("dest_timezone"))
    df = df.with_columns(
        (pl.col("flight_date").cast(pl.Datetime("us")) + pl.duration(minutes=pl.col("crs_dep_time_min")))
            .alias("_local_dep"),
        severity(pl.col("dep_delay")).alias("target_class"),
        ((pl.col("cancelled") == 0) & (pl.col("diverted") == 0) & pl.col("dep_delay").is_not_null())
            .fill_null(False).alias("eligible_target"))
    # Nonpositive elapsed time is not a valid scheduled arrival; retain target
    # labels but prohibit links depending on that schedule.
    df = df.with_columns(pl.when(pl.col("crs_elapsed_time") > 0).then(pl.col("crs_elapsed_time"))
                          .otherwise(None).alias("crs_elapsed_time"))
    parts = []
    for key, group in df.partition_by("origin_timezone", as_dict=True).items():
        zone = key[0]
        if zone:
            parts.append(resolve_zone(group, zone, cfg))
        else:
            # Preserve rows for accounting and chain guards; null absolute times.
            group = group.with_columns([pl.lit(None, dtype=UTC).alias(c) for c in
                ["scheduled_departure_utc", "scheduled_arrival_utc", "actual_gate_departure_utc", "actual_wheels_off_utc"]])
            group = group.with_columns(pl.lit(False).alias("_dst_ambiguous"),
                pl.lit(None, dtype=pl.Boolean).alias("scheduled_arrival_clock_ok"),
                pl.lit(None, dtype=pl.Boolean).alias("gate_departure_clock_ok"),
                pl.lit(None, dtype=pl.Boolean).alias("takeoff_day_crossing"),
                pl.lit(None, dtype=pl.Float64).alias("resolved_taxi_out_min"),
                pl.lit("invalid").alias("takeoff_time_source"))
            parts.append(group)
    # Intermediate helper column sets vary for unknown zones: remove before concat.
    df = pl.concat([p.select([c for c in p.columns if not c.startswith("_")]+["_dst_ambiguous"])
                    for p in parts], how="diagonal_relaxed").sort("raw_row_number")
    df = df.with_columns(
        pl.when(pl.col("scheduled_departure_utc").is_null() | pl.col("dest_timezone").is_null())
          .then(pl.lit("invalid_timezone_or_schedule"))
        .when(pl.col("_dst_ambiguous")).then(pl.lit("dst_resolved_by_arrival_schedule"))
        .when(pl.col("gate_departure_clock_ok") == False).then(pl.lit("gate_departure_clock_mismatch"))
        .when(pl.col("scheduled_arrival_clock_ok") == False).then(pl.lit("scheduled_arrival_clock_mismatch"))
        .when(pl.col("takeoff_time_source") == "invalid").then(pl.lit("invalid_takeoff"))
        .otherwise(pl.lit("ok")).alias("timestamp_quality_flag"))
    invalids = {c: df.select((pl.col(c).is_not_null() & pl.col(c+"_min").is_null()).sum()).item() for c in CLOCKS}
    exclusion = (pl.when(~pl.col("cancelled").is_in([0, 1]) | ~pl.col("diverted").is_in([0, 1]) |
                        pl.col("cancelled").is_null() | pl.col("diverted").is_null()).then(pl.lit("invalid_status_flags"))
        .when(pl.col("cancelled") == 1).then(pl.lit("cancelled"))
        .when(pl.col("diverted") == 1).then(pl.lit("diverted"))
        .when(pl.col("dep_delay").is_null()).then(pl.lit("missing_or_invalid_dep_delay"))
        .otherwise(pl.lit("eligible")))
    exclusion_counts = df.select(exclusion.alias("reason")).group_by("reason").len().to_dicts()
    stats = dict(month=entry["month"], rows=df.height, numeric_invalid=numeric_invalid,
        clock_invalid=invalids, null_counts=dict(zip(df.columns, df.null_count().row(0))),
        target_exclusion_counts={row["reason"]: row["len"] for row in exclusion_counts},
        nonpositive_distance=df.filter(pl.col("distance") <= 0).height,
        missing_distance=df["distance"].null_count(), missing_elapsed=df["crs_elapsed_time"].null_count(),
        clock_2400_counts={c: df.filter(pl.col(c+"_min") == 1440).height for c in CLOCKS},
        dst_transition_date_rows=df.filter(pl.col("flight_date").dt.strftime("%m-%d").is_in(["03-09", "11-02"])).height,
        missing_tail=df["tail_number"].null_count(),
        invalid_airport_codes=df.select((~pl.col("origin").str.contains(r"^[A-Z]{3}$") |
            ~pl.col("dest").str.contains(r"^[A-Z]{3}$")).fill_null(True).sum()).item(),
        invalid_flags=df.select((~pl.col("cancelled").is_in([0, 1]) |
                                ~pl.col("diverted").is_in([0, 1])).fill_null(True).sum()).item(),
        ambiguous_schedule=df["_dst_ambiguous"].sum(),
        invalid_schedule=df["scheduled_departure_utc"].null_count(),
        arrival_clock_mismatches=df.filter(pl.col("scheduled_arrival_clock_ok") == False).height,
        gate_clock_mismatches=df.filter(pl.col("gate_departure_clock_ok") == False).height,
        takeoff_day_crossing=df["takeoff_day_crossing"].sum(),
        cross_zone=df.filter(pl.col("origin_timezone") != pl.col("dest_timezone")).height,
        negative_delays=df.filter(pl.col("dep_delay") < 0).height,
        taxi_out_over_180_flag_only=df.filter(pl.col("resolved_taxi_out_min") > 180).height,
        takeoff_sources=df.group_by("takeoff_time_source").len().to_dicts())
    return df.rename({"_dst_ambiguous": "schedule_dst_ambiguous"}), stats


def mapping_provenance(mapping: dict[str, str], observed: set[str]) -> dict:
    subset = {c: mapping.get(c) for c in sorted(observed)}
    return dict(package="airportsdata", version=version("airportsdata"),
        source="https://github.com/mborsetti/airportsdata", tzdata_version=version("tzdata"),
        polars_version=version("polars"),
        timezone_engine="Pinned polars-runtime-32/chrono-tz; date-dependent IANA rules",
        mapping=subset, unresolved=[c for c, tz in subset.items() if tz is None],
        mapping_sha256=hashlib.sha256(json.dumps(subset, sort_keys=True).encode()).hexdigest())
