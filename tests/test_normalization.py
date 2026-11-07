from datetime import datetime, timezone
import polars as pl
import pytest
from flight_delay.normalize import clock_minutes, severity
from conftest import leg


def test_label_thresholds_and_nulls():
    df = pl.DataFrame({"x": [-5., 0., 14.999, 15., 59.999, 60., 180., None, float("nan"), float("inf")]})
    assert df.select(severity(pl.col("x")).alias("severity"))["severity"].to_list() == [0,0,0,1,1,2,2,None,None,None]


def test_hhmm_validation():
    df = pl.DataFrame({"x": ["0", "0000", "5", "059", "2400", "2360", "2401", "-1", "25.5", None]})
    assert df.select(clock_minutes(pl.col("x")))["x"].to_list() == [0,0,5,59,1440,None,None,None,None,None]


def test_cross_zone_and_negative_delay(normalized):
    df = normalized([leg(dep_delay=-5, dep_time="0755", wheels_off="0810")])
    row = df.row(0,named=True)
    assert row["scheduled_departure_utc"] == datetime(2025,1,1,13,tzinfo=timezone.utc)
    assert row["scheduled_arrival_utc"] == datetime(2025,1,1,15,tzinfo=timezone.utc)
    assert row["actual_gate_departure_utc"] == datetime(2025,1,1,12,55,tzinfo=timezone.utc)
    assert row["actual_wheels_off_utc"] == datetime(2025,1,1,13,10,tzinfo=timezone.utc)
    assert row["scheduled_arrival_clock_ok"]


def test_overnight_and_2400(normalized):
    row = normalized([leg(crs_dep_time="2350", crs_arr_time="0050", wheels_off="0010", dep_time="2350")]).row(0,named=True)
    assert row["actual_wheels_off_utc"] == datetime(2025,1,2,5,10,tzinfo=timezone.utc)
    row = normalized([leg(crs_dep_time="2400", crs_arr_time="0100", wheels_off="0015", dep_time="0000")]).row(0,named=True)
    assert row["scheduled_departure_utc"] == datetime(2025,1,2,5,tzinfo=timezone.utc)


def test_dst_spring_nonexistent(normalized):
    row = normalized([leg(flight_date="2025-03-09", crs_dep_time="0230")]).row(0,named=True)
    assert row["scheduled_departure_utc"] is None


def test_dst_fall_ambiguous_and_resolved(normalized):
    # Both folds give a 01:30 arrival for a twenty-minute same-zone flight.
    row = normalized([leg(flight_date="2025-11-02", origin="ATL",dest="ATL",crs_dep_time="0110",
                          crs_arr_time="0130",crs_elapsed_time="20")]).row(0,named=True)
    assert row["scheduled_departure_utc"] is None
    # Only the first 01:30 departure yields a 02:30 arrival two hours later.
    row = normalized([leg(flight_date="2025-11-02", origin="ATL",dest="ATL",crs_dep_time="0130",
                          crs_arr_time="0230",crs_elapsed_time="120")]).row(0,named=True)
    assert row["scheduled_departure_utc"] == datetime(2025,11,2,5,30,tzinfo=timezone.utc)


def test_missing_wheels_derivation_and_inconsistency(normalized):
    row = normalized([leg(wheels_off=None,taxi_out="20")]).row(0,named=True)
    assert row["takeoff_time_source"] == "derived"
    row = normalized([leg(wheels_off="0900",taxi_out="20")]).row(0,named=True)
    assert row["takeoff_time_source"] == "invalid"
    row = normalized([leg(wheels_off="0750")]).row(0,named=True)
    assert row["takeoff_time_source"] == "invalid"


def test_unknown_timezone(normalized):
    row = normalized([leg(origin="ZZZ")]).row(0,named=True)
    assert row["scheduled_departure_utc"] is None
    assert row["timestamp_quality_flag"] == "invalid_timezone_or_schedule"


def test_gate_clock_inconsistency_is_not_silently_corrected(normalized):
    row=normalized([leg(dep_time="0900",wheels_off="0915",dep_delay="0")]).row(0,named=True)
    assert row["actual_wheels_off_utc"] is None
    assert row["takeoff_time_source"]=="invalid"
    assert row["timestamp_quality_flag"]=="gate_departure_clock_mismatch"
