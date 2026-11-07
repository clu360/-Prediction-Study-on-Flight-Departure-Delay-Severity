"""Export only aggregate validation evidence and generate exact schema documentation.

Run from the repository root after preprocessing. Never exports flight-level rows,
local source paths, linkage examples or intermediate records.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import polars as pl

from flight_delay.discovery import ALIASES
from flight_delay.outputs import G1_ALLOWED, G2_ALLOWED

FORMULAS = {
    "record_id": ("source hash + month + row index", "YYYY-MM-SHA256[:20]-CSV row index"),
    "target_record_id": ("record_id", "Target record identity"),
    "prior_record_id": ("master schedule record_id", "Immediately preceding same-tail scheduled record"),
    "source_month": ("FlightDate", "Target local calendar month"),
    "flight_date": ("FL_DATE / FlightDate", "Parse originating scheduled local date"),
    "target_crs_dep_utc": ("FlightDate, CRSDepTime, origin timezone", "DST-aware local schedule to UTC"),
    "target_crs_arr_utc": ("target_crs_dep_utc, CRSElapsedTime", "Departure UTC + scheduled elapsed minutes"),
    "target_dep_delay_min": ("DepDelay", "Finite signed departure delay; evaluation only"),
    "target_class": ("DepDelay", "0 if <15; 1 if [15,60); 2 if >=60"),
    "departure_severity": ("DepDelay", "Same severity thresholds; historical observed event"),
    "prediction_time_utc": ("prior WheelsOff / derived takeoff", "Prior actual takeoff UTC + 1 minute"),
    "prior_wheels_off_utc": ("prior WheelsOff; optional prior TaxiOut", "Resolve takeoff date against prior actual gate departure"),
    "prior_actual_gate_departure_utc": ("prior schedule, prior DepDelay", "Prior scheduled departure UTC + prior signed DepDelay"),
    "prior_scheduled_departure_utc": ("prior schedule", "Prior local scheduled departure localized to UTC"),
    "prior_scheduled_arrival_utc": ("prior schedule, prior CRSElapsedTime", "Prior scheduled departure UTC + prior scheduled duration"),
    "scheduled_turnaround_min": ("target departure, prior scheduled arrival", "Target scheduled departure UTC minus prior scheduled arrival UTC, minutes"),
    "prior_to_target_schedule_gap_min": ("target and prior scheduled departures", "Target minus prior scheduled departure UTC, minutes"),
    "minutes_until_target_crs_dep_at_prediction": ("target schedule, prediction instant", "Target scheduled departure UTC minus prediction time UTC, minutes"),
    "observed_departure_time_utc": ("schedule, DepDelay", "Valid actual gate departure = scheduled departure UTC + signed delay"),
    "target_crs_dep_local_hour": ("CRSDepTime", "floor((parsed clock minutes mod 1440)/60)"),
    "target_day_of_week": ("FlightDate", "ISO weekday, Monday=1 .. Sunday=7"),
    "target_month": ("FlightDate", "Local calendar month 1..12"),
    "target_crs_elapsed_min": ("CRSElapsedTime", "Finite positive scheduled duration in minutes"),
    "prior_crs_elapsed_min": ("prior CRSElapsedTime", "Prior finite positive scheduled duration in minutes"),
    "target_distance_miles": ("Distance", "Finite distance, miles; nulls retained/flagged"),
    "target_origin": ("Origin", "Strip and uppercase IATA code"),
    "target_dest": ("Dest", "Strip and uppercase IATA code"),
    "target_carrier": ("Reporting_Airline / OP_UNIQUE_CARRIER", "Strip and uppercase reporting-carrier code"),
    "target_flight_number": ("Flight_Number_Reporting_Airline / OP_CARRIER_FL_NUM", "Identifier text, not a unique row ID"),
    "prior_dep_delay_min": ("prior DepDelay", "Finite signed prior departure delay, minutes"),
    "prior_origin": ("prior Origin", "Strip and uppercase prior origin IATA code"),
    "prior_dest": ("prior Dest", "Strip and uppercase prior destination IATA code"),
    "tail_number": ("Tail_Number / TAIL_NUM", "Strip/uppercase nonempty aircraft identifier"),
    "reporting_airline": ("Reporting_Airline / OP_UNIQUE_CARRIER", "Reporting-carrier identifier"),
    "flight_number": ("Flight_Number_Reporting_Airline / OP_CARRIER_FL_NUM", "Identifier text; absence explicitly logged"),
    "origin": ("Origin", "Strip and uppercase IATA code"),
    "dest": ("Dest", "Strip and uppercase IATA code"),
    "crs_dep_time": ("CRSDepTime", "Source local HHMM text; parsed minute clock validated in master"),
    "crs_arr_time": ("CRSArrTime", "Source destination-local HHMM text; arrival cross-check"),
    "crs_elapsed_time": ("CRSElapsedTime", "Finite positive scheduled duration in minutes"),
    "distance": ("Distance", "Finite miles, missingness retained and audited"),
    "cancelled": ("Cancelled", "Numeric 0/1; invalid flags excluded from targets"),
    "diverted": ("Diverted", "Numeric 0/1; diverted targets excluded; history retains departure observations"),
    "dep_delay": ("DepDelay", "Finite signed historical departure delay in minutes"),
    "raw_source_month": ("source manifest", "Canonical input month"),
    "raw_source_file": ("source manifest", "Original source filename, local only"),
    "raw_row_number": ("CSV position", "Zero-based data row index, excludes header"),
    "link_status": ("linkage rules", "valid or invalid; exactly one decision per eligible target"),
    "link_reason": ("ordered linkage checks", "Stable primary failure reason, or valid"),
    "prior_route_continuity_ok": ("prior Dest, target Origin", "Standardized airports equal; null compares false"),
    "takeoff_time_source": ("prior takeoff resolution", "parsed, derived, invalid; null if no predecessor"),
    "timestamp_quality_flag": ("target timestamp checks", "Target schedule/takeoff quality category; metadata only"),
    "ambiguous_tail_rotation_flag": ("master same-tail schedule and service keys", "Target schedule tie or conflicting service duplicate"),
    "conflicting_service_duplicate_flag": ("date, airline, flight number, origin, dest", "Multiple purported same-service records without deduplication"),
    "tail_has_unknown_order": ("master tail schedule", "Any same-tail record has unresolved scheduled departure"),
    "audit_target_departed_before_prediction": ("target actual gate time, prediction time", "Target actual departure <= prediction, retrospective only"),
    "audit_prior_arrived_before_prediction": ("prior observed arrival, prediction time", "Prior observed arrival <= prediction, retrospective only"),
    "audit_prior_arrival_after_target_departure": ("prior observed arrival, target actual departure", "Prior arrival > target actual departure, retrospective only"),
    "audit_prior_observed_arrival_utc": ("prior gate departure, prior ActualElapsedTime", "Prior gate departure + actual elapsed minutes; audit only"),
    "valid_observation": ("history eligibility checks", "True: operated departure with valid absolute time and finite delay"),
    "gate_departure_clock_ok": ("actual gate UTC, DepTime", "Derived local actual gate clock agrees within tolerance"),
    "fold_id": ("rolling-fold protocol", "YYYY-MM of test month, Apr through Dec"),
    "train_months": ("rolling-fold protocol", "Comma-separated three immediately previous calendar months"),
    "test_month": ("rolling-fold protocol", "Test calendar month 4..12"),
    "role": ("target local FlightDate", "train for prior 3 months; test for following month"),
    "is_final_holdout": ("test_month", "True for November and December folds"),
}


def dictionary(schemas: dict) -> str:
    lines = ["# Processed data dictionary", "",
        "All stored columns use snake_case. UTC timestamps are timezone-aware microsecond timestamps; `source_month` uses target local FlightDate. Hive year/month directories are additional partition metadata. The exact machine-readable schema is `reports/schema.json`.", "",
        "G1/G2 identify allowable information for future explicit feature views; this release does not construct model inputs. IDs, labels, metadata, audit outcomes and raw timestamps are excluded from those allowlists. History events are inputs to later causal G3 calculations, not current-flight predictors.", "",
        "Labels and audited facts become known after the relevant outcome; they must never enter prediction feature matrices. Missing/invalid target labels are excluded. Missing predictor ingredients are retained and flagged until a later fold-local imputation policy. Valid cohort linkage/timing fields are required nonnull. Invalid connections keep nullable prior facts and a reason. No invented defaults are used.", ""]
    for table, fields in schemas.items():
        lines.extend([f"## {table}", "", "| Column | dtype | Source | Transformation | Group | Availability | Missingness policy |",
                      "|---|---|---|---|---|---|---|"])
        for col, dtype in fields.items():
            if col not in FORMULAS:
                raise ValueError(f"Missing dictionary definition: {col}")
            source, formula = FORMULAS[col]
            group = "G1" if col in G1_ALLOWED else "G2" if col in G2_ALLOWED else "metadata"
            if col in {"target_class", "target_dep_delay_min", "departure_severity", "dep_delay"}:
                group = "label" if table != "history_departure_events" else "metadata (history outcome)"
            availability = "Before prediction (schedule)" if group == "G1" else "At prediction (prior departure/schedule)" if group == "G2" else "Offline metadata; not a predictor"
            if col.startswith("prior_") or col in {"prediction_time_utc","takeoff_time_source","minutes_until_target_crs_dep_at_prediction"}:
                availability = "At prediction (prior departure and schedule)"
            if col in {"source_month","flight_date","origin","dest","reporting_airline","flight_number","crs_dep_time","crs_arr_time","crs_elapsed_time","distance","target_crs_dep_utc","target_crs_arr_utc","prior_to_target_schedule_gap_min","scheduled_turnaround_min"}:
                availability = "Before prediction (schedule/identity)"
            if col in {"cancelled","diverted","timestamp_quality_flag"}:
                availability = "Retrospective target/event audit; not a predictor"
            if col.startswith("audit_") or group.startswith("label") or table == "history_departure_events":
                availability = "After relevant event; audit/label/history only"
            missing = "Required for every retained row" if col in {"record_id", "target_record_id", "source_month", "flight_date", "target_class", "target_dep_delay_min", "link_status", "link_reason", "fold_id", "role"} else (
                "Required in valid cohort; nullable in invalid links" if col.startswith("prior_") or col in {"prediction_time_utc", "scheduled_turnaround_min"} else "Retain null/invalid flags; no imputation")
            lines.append(f"| `{col}` | {dtype} | {source} | {formula} | {group} | {availability} | {missing} |")
        lines.append("")
    lines.extend(["## Accepted raw-to-canonical mapping", "", "Source headers are compared after uppercasing and removing underscores/punctuation. Only the explicit aliases below are accepted; canonical intermediate names then map to processed columns above.", "", "| Normalized source header | Canonical name |", "|---|---|"])
    lines.extend(f"| `{raw}` | `{processed}` |" for raw, processed in sorted(ALIASES.items()))
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--processed-dir",type=Path,default=Path("data/processed"))
    args=parser.parse_args()
    audit=args.processed_dir/"audit"
    dest=Path("reports")
    dest.mkdir(exist_ok=True)
    signature=json.loads((audit/"run_signature.json").read_text())
    master=list((args.processed_dir.parent/"interim"/signature["run_key"]).glob("bucket-*/*.parquet"))
    lf=pl.scan_parquet(master)
    extra=lf.select(
        pl.col("crs_elapsed_time").max().alias("max_scheduled_duration_min"),
        (pl.col("crs_elapsed_time")>720).sum().alias("scheduled_duration_over_720_flag_only"),
        ((pl.col("crs_elapsed_time")>720)&(pl.col("cancelled")==1)).sum().alias("long_scheduled_cancelled"),
        ((pl.col("crs_elapsed_time")>720)&(pl.col("cancelled")==0)).sum().alias("long_scheduled_operated"),
        pl.col("resolved_taxi_out_min").max().alias("max_resolved_taxi_out_min"),
        pl.col("dep_delay").min().alias("min_signed_departure_delay_min"),
        pl.col("dep_delay").max().alias("max_signed_departure_delay_min")
    ).collect(engine="streaming").to_dicts()[0]
    (audit/"additional_quality.json").write_text(json.dumps(extra,indent=2),encoding="utf-8")
    for name in ["preprocessing_summary.json","schema.json","normalization_quality.json","airport_timezones.json",
                 "preprocessing_audit.md","exclusion_counts.csv","coverage_by_month.csv","coverage_by_airport.csv",
                 "coverage_by_carrier.csv","five_airport_monthly_class_counts.csv","run_signature.json"]:
        shutil.copyfile(audit/name,dest/name)
    for name in ["reproducibility.json","arrow_validation.json","additional_quality.json","retrospective_anomaly_review_summary.json"]:
        if (audit/name).exists():
            shutil.copyfile(audit/name,dest/name)
    manifest=json.loads((audit/"input_manifest.json").read_text())
    for entry in manifest["inputs"]:
        entry.pop("path",None)
    (dest/"input_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    schemas=json.loads((audit/"schema.json").read_text())
    Path("docs/data_dictionary.md").write_text(dictionary(schemas),encoding="utf-8")
    print("Exported sanitized aggregate evidence; no flight-level records.")


if __name__=="__main__":
    main()
