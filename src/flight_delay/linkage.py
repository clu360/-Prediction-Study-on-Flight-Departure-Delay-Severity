"""Immediate scheduled predecessor selection using schedule and prior departure only."""
from __future__ import annotations
import polars as pl
from .config import Config
from .normalize import duration_minutes

PRIOR_FIELDS = ["record_id", "origin", "dest", "reporting_airline", "flight_number", "dep_delay",
    "cancelled", "diverted", "scheduled_departure_utc", "scheduled_arrival_utc",
    "actual_gate_departure_utc", "actual_wheels_off_utc", "crs_elapsed_time", "takeoff_time_source",
    "timestamp_quality_flag", "scheduled_arrival_clock_ok", "ambiguous_tail_rotation_flag",
    "actual_elapsed_time", "origin_timezone", "dest_timezone"]


def link_events(events: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Return every eligible target. Arrival outcomes never affect link selection."""
    df = events.sort(["tail_number", "scheduled_departure_utc", "record_id"], nulls_last=True)
    df = df.with_columns(
        ((pl.len().over(["tail_number", "scheduled_departure_utc"]) > 1) & pl.col("tail_number").is_not_null())
            .alias("ambiguous_tail_rotation_flag"),
        pl.col("scheduled_departure_utc").is_null().any().over("tail_number").alias("tail_has_unknown_order"))
    # Conflicting purported service duplicates must not produce convenient alternative links.
    natural_key = ["flight_date", "reporting_airline", "flight_number", "origin", "dest"]
    service_conflict = (pl.len().over(natural_key) > 1) & pl.col("flight_number").is_not_null()
    if "source_service_duplicate_flag" in df:
        service_conflict = service_conflict | pl.col("source_service_duplicate_flag")
    df = df.with_columns(service_conflict.alias("conflicting_service_duplicate_flag"))
    df = df.with_columns((pl.col("ambiguous_tail_rotation_flag") |
        pl.col("conflicting_service_duplicate_flag")).alias("ambiguous_tail_rotation_flag"))
    df = df.with_columns([pl.col(c).shift(1).over("tail_number").alias("prior_" + c) for c in PRIOR_FIELDS])
    df = df.with_columns([pl.when(pl.col("tail_number").is_not_null()).then(pl.col("prior_" + c))
                          .otherwise(None).alias("prior_" + c) for c in PRIOR_FIELDS])
    df = df.filter(pl.col("eligible_target")).with_columns(
        (pl.col("prior_actual_wheels_off_utc") + duration_minutes(pl.lit(cfg.prediction_offset_minutes)))
            .alias("prediction_time_utc"),
        ((pl.col("scheduled_departure_utc") - pl.col("prior_scheduled_arrival_utc"))
            .dt.total_microseconds() / 60_000_000).alias("scheduled_turnaround_min"),
        ((pl.col("scheduled_departure_utc") - pl.col("prior_scheduled_departure_utc"))
            .dt.total_microseconds() / 60_000_000).alias("prior_to_target_schedule_gap_min"),
        (pl.col("prior_dest") == pl.col("origin")).fill_null(False).alias("prior_route_continuity_ok"))
    # Stable priority: exactly one primary reason, retaining independent diagnostic flags.
    reason = (pl.when(pl.col("tail_number").is_null()).then(pl.lit("missing_tail"))
        .when(pl.col("scheduled_departure_utc").is_null() | pl.col("tail_has_unknown_order") |
              pl.col("origin_timezone").is_null() | pl.col("dest_timezone").is_null())
            .then(pl.lit("invalid_timezone"))
        .when(pl.col("ambiguous_tail_rotation_flag") | pl.col("prior_ambiguous_tail_rotation_flag").fill_null(False))
            .then(pl.lit("ambiguous_rotation"))
        .when(pl.col("prior_record_id").is_null()).then(pl.lit("no_previous_record"))
        .when((pl.col("prior_cancelled") != 0) | (pl.col("prior_diverted") != 0) |
              pl.col("prior_cancelled").is_null() | pl.col("prior_diverted").is_null())
            .then(pl.lit("intervening_cancelled_or_diverted"))
        .when(pl.col("prior_dep_delay").is_null()).then(pl.lit("predecessor_missing_delay"))
        .when(pl.col("prior_actual_wheels_off_utc").is_null()).then(pl.lit("predecessor_missing_takeoff"))
        .when(~pl.col("prior_route_continuity_ok")).then(pl.lit("airport_mismatch"))
        .when(pl.col("prior_scheduled_departure_utc") >= pl.col("scheduled_departure_utc"))
            .then(pl.lit("ambiguous_rotation"))
        .when(pl.col("prediction_time_utc") >= pl.col("scheduled_departure_utc"))
            .then(pl.lit("late_prior_takeoff"))
        .when(pl.col("prior_scheduled_arrival_utc").is_null() |
              pl.col("prior_scheduled_arrival_clock_ok").eq(False).fill_null(True))
            .then(pl.lit("invalid_prior_schedule"))
        .when(pl.col("scheduled_turnaround_min") < 0).then(pl.lit("negative_scheduled_turnaround"))
        .when(pl.col("scheduled_turnaround_min") > cfg.max_turnaround_minutes).then(pl.lit("gap_over_24h"))
        .otherwise(pl.lit("valid")))
    df = df.with_columns(reason.alias("link_reason"))
    df = df.with_columns(
        pl.when(pl.col("link_reason") == "valid").then(pl.lit("valid")).otherwise(pl.lit("invalid")).alias("link_status"),
        (pl.col("actual_gate_departure_utc") <= pl.col("prediction_time_utc"))
            .fill_null(False).alias("audit_target_departed_before_prediction"),
        (pl.col("prior_actual_gate_departure_utc") + duration_minutes(pl.col("prior_actual_elapsed_time")))
            .alias("audit_prior_observed_arrival_utc"))
    df = df.with_columns(
        (pl.col("audit_prior_observed_arrival_utc") <= pl.col("prediction_time_utc"))
            .fill_null(False).alias("audit_prior_arrived_before_prediction"),
        (pl.col("audit_prior_observed_arrival_utc") > pl.col("actual_gate_departure_utc"))
            .fill_null(False).alias("audit_prior_arrival_after_target_departure"))
    return df


def folds(cohort: pl.DataFrame, year: int) -> pl.DataFrame:
    parts = []
    for test in range(4, 13):
        train = list(range(test - 3, test))
        subset = cohort.filter(pl.col("target_month").is_in(train + [test]))
        parts.append(subset.select("target_record_id", "source_month").with_columns(
            pl.lit(f"{year}-{test:02}").alias("fold_id"),
            pl.lit(",".join(f"{year}-{m:02}" for m in train)).alias("train_months"),
            pl.lit(test, dtype=pl.Int8).alias("test_month"),
            pl.when(pl.col("source_month") == test).then(pl.lit("test")).otherwise(pl.lit("train")).alias("role"),
            pl.lit(test >= 11).alias("is_final_holdout")))
    return pl.concat(parts)
