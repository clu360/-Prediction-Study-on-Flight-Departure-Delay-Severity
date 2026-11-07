"""Explicit publication schemas and leakage boundaries."""
from __future__ import annotations
import polars as pl

# Later modeling code must choose explicit views from these schedule/prior fields.
# This is a contract, not a Phase 2 feature matrix.
G1_ALLOWED = ("target_carrier", "target_origin", "target_dest", "target_crs_dep_local_hour",
    "target_day_of_week", "target_month", "target_crs_elapsed_min", "target_distance_miles")
G2_ALLOWED = ("prior_origin", "prior_dep_delay_min", "prior_crs_elapsed_min", "scheduled_turnaround_min",
    "minutes_until_target_crs_dep_at_prediction")
LABEL_COLUMNS = ("target_class", "target_dep_delay_min")


def clean(df: pl.DataFrame) -> pl.DataFrame:
    return df.select("record_id", "source_month", "flight_date", "reporting_airline", "flight_number",
        "tail_number", "origin", "dest", "crs_dep_time", "crs_arr_time", "crs_elapsed_time", "distance",
        pl.col("scheduled_departure_utc").alias("target_crs_dep_utc"),
        pl.col("scheduled_arrival_utc").alias("target_crs_arr_utc"),
        pl.col("dep_delay").alias("target_dep_delay_min"), "target_class", "cancelled", "diverted",
        "raw_source_month", "raw_source_file", "raw_row_number", "timestamp_quality_flag")


def connections(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(pl.col("record_id").alias("target_record_id"), "prior_record_id", "source_month",
        "link_status", "link_reason", "prediction_time_utc",
        pl.col("prior_actual_wheels_off_utc").alias("prior_wheels_off_utc"),
        "prior_actual_gate_departure_utc", pl.col("prior_takeoff_time_source").alias("takeoff_time_source"),
        "prior_scheduled_departure_utc", "prior_scheduled_arrival_utc", "scheduled_turnaround_min",
        "prior_to_target_schedule_gap_min", "prior_route_continuity_ok", "timestamp_quality_flag",
        "ambiguous_tail_rotation_flag", "tail_has_unknown_order", "conflicting_service_duplicate_flag",
        "audit_target_departed_before_prediction", "audit_prior_arrived_before_prediction",
        "audit_prior_arrival_after_target_departure", "audit_prior_observed_arrival_utc")


def cohort(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("link_status") == "valid").select(
        pl.col("record_id").alias("target_record_id"), "prior_record_id", "flight_date",
        pl.col("origin").alias("target_origin"), pl.col("dest").alias("target_dest"),
        pl.col("reporting_airline").alias("target_carrier"), pl.col("flight_number").alias("target_flight_number"),
        pl.col("scheduled_departure_utc").alias("target_crs_dep_utc"),
        (pl.col("crs_dep_time_min") % 1440 // 60).cast(pl.Int8).alias("target_crs_dep_local_hour"),
        pl.col("flight_date").dt.weekday().cast(pl.Int8).alias("target_day_of_week"),
        pl.col("flight_date").dt.month().cast(pl.Int8).alias("target_month"),
        pl.col("crs_elapsed_time").alias("target_crs_elapsed_min"), pl.col("distance").alias("target_distance_miles"),
        "prior_origin", "prior_dest", pl.col("prior_dep_delay").alias("prior_dep_delay_min"),
        pl.col("prior_crs_elapsed_time").alias("prior_crs_elapsed_min"),
        pl.col("prior_actual_wheels_off_utc").alias("prior_wheels_off_utc"), "prediction_time_utc",
        "scheduled_turnaround_min", ((pl.col("scheduled_departure_utc") - pl.col("prediction_time_utc"))
            .dt.total_microseconds() / 60_000_000).alias("minutes_until_target_crs_dep_at_prediction"),
        "target_class", pl.col("dep_delay").alias("target_dep_delay_min"), "source_month",
        "timestamp_quality_flag", pl.col("prior_takeoff_time_source").alias("takeoff_time_source"))


def history(df: pl.DataFrame) -> pl.DataFrame:
    # A canceled row is not an observed gate departure. Diverted flights with valid
    # departure observations belong in histories, with diversion flags retained.
    return df.filter((pl.col("cancelled") == 0) & pl.col("diverted").is_in([0, 1]) &
        pl.col("dep_delay").is_not_null() & pl.col("actual_gate_departure_utc").is_not_null() &
        pl.col("gate_departure_clock_ok").ne(False).fill_null(True)).select(
        "record_id", "source_month", "flight_date", "origin", "reporting_airline",
        pl.col("actual_gate_departure_utc").alias("observed_departure_time_utc"), "dep_delay",
        pl.col("target_class").alias("departure_severity"), "cancelled", "diverted",
        pl.lit(True).alias("valid_observation"), "gate_departure_clock_ok", "timestamp_quality_flag")
