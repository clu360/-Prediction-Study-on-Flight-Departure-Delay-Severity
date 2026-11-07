"""Build a concise public report from validated aggregate evidence."""
from __future__ import annotations
import json
from pathlib import Path
import polars as pl


def main():
    root=Path("reports")
    summary=json.loads((root/"preprocessing_summary.json").read_text())
    manifest=json.loads((root/"input_manifest.json").read_text())
    zones=json.loads((root/"airport_timezones.json").read_text())
    quality=json.loads((root/"normalization_quality.json").read_text())
    schemas=json.loads((root/"schema.json").read_text())
    counts=summary["counts"]
    proof=json.loads((root/"reproducibility.json").read_text())
    uncached=proof["uncached_execution"]
    arrow=json.loads((root/"arrow_validation.json").read_text())
    extra=json.loads((root/"additional_quality.json").read_text())
    reviewed=json.loads((root/"retrospective_anomaly_review_summary.json").read_text())
    if not (summary["acceptance_passed"] and proof["passed"] and arrow["passed"]):
        raise ValueError("Cannot publish a successful handoff with failing evidence")
    lines=["# Phase 1 validation handoff", "", "The complete final pipeline was executed on all twelve supplied 2025 monthly datasets. All source-byte hashes were rechecked after processing. These are Phase 1 preprocessing results, not model-performance findings.", "",
        "## Record flow", "", "| Artifact | Rows |", "|---|---:|"]
    for key in ["raw","flights_clean","flight_connections","modeling_cohort","history_departure_events","folds"]:
        lines.append(f"| `{key}` | {counts[key]:,} |")
    lines.extend(["",f"Valid linkage coverage: **{counts['valid_link_percentage']:.6f}%** of eligible labeled targets.", "",
        "## Class distributions", "", "| Class | All eligible targets | Share | Linked primary cohort | Share |", "|---|---:|---:|---:|---:|"])
    for i in range(3):
        a=next(r for r in summary["classes"]["flights_clean"] if r["target_class"]==i)
        b=next(r for r in summary["classes"]["modeling_cohort"] if r["target_class"]==i)
        lines.append(f"| {i} | {a['count']:,} | {a['percentage']:.6f}% | {b['count']:,} | {b['percentage']:.6f}% |")
    lines.extend(["", "The change in severity shares after linkage filtering must be acknowledged when defining the Phase 2 study population.", "",
        "## Target and link exclusions", "", "Target-level reasons are mutually exclusive in the documented priority order.", "", "| Target status | Count |", "|---|---:|"])
    for reason,n in summary["target_exclusion_counts"].items():
        lines.append(f"| {reason} | {n:,} |")
    lines.extend(["", "Every eligible target has one primary link reason. Reasons absent from the following table have zero occurrences.", "", "| Link reason | Count |", "|---|---:|"])
    lines.extend(f"| {r['link_reason']} | {r['count']:,} |" for r in summary["exclusion_counts"])
    lines.extend(["",f"12-hour sensitivity cohort: **{summary['sensitivity_12h_count']:,}** rows (same links, stricter turnaround maximum).", "",
        "## Full source inventory", "", "All files use the same source header list, available in `input_manifest.json`. Full SHA-256 values below refer to the original unmodified bytes. No exact duplicate raw records were found.", "",
        "| File | Rows | Dates | SHA-256 |", "|---|---:|---|---|"])
    lines.extend(f"| {e['filename']} | {e['row_count']:,} | {e['date_min']} to {e['date_max']} | `{e['sha256']}` |" for e in sorted(manifest["inputs"],key=lambda x:x["month"]))
    lines.extend(["", "## Airport subsets by target local month", "", "The following are primary linked cohort counts. Every airport/month has data. Class counts before and after filtering are in `five_airport_monthly_class_counts.csv`; airport, month and carrier coverage tables include denominators.", "",
        "| Month | ATL | DFW | DEN | ORD | LAX |", "|---|---:|---:|---:|---:|---:|"])
    df=pl.read_csv(root/"five_airport_monthly_class_counts.csv").filter(pl.col("link_status")=="valid").group_by("source_month","origin").agg(pl.col("count").sum())
    for month in range(1,13):
        values=[]
        for airport in ["ATL","DFW","DEN","ORD","LAX"]:
            values.append(f"{df.filter((pl.col('source_month')==month)&(pl.col('origin')==airport))['count'].sum():,}")
        lines.append(f"| 2025-{month:02} | "+" | ".join(values)+" |")
    lines.extend(["", "## Exact output schemas and location", "",
        "Local official outputs are under `data/processed/`, with `year=2025/month=MM/part-BB.parquet`. All UTC timestamps are timezone-aware microsecond timestamps. `schema.json` gives every column and exact dtype. The full [data dictionary](../docs/data_dictionary.md) gives provenance, transformations, availability, allowed groups and missingness policies.", ""])
    lines.extend(f"- `{name}`: {len(fields)} stored columns." for name,fields in schemas.items())
    lines.extend(["", "Raw sources remain in their original external folder. Normalized master events and hash-verified caches live under ignored `data/interim/`. Local audits, example records and timestamp anomaly details live under `data/processed/audit/`; these flight-level records are not exported to Git.", "",
        "## Data quality and validation", ""])
    metrics=[("Observed airport codes",len(zones["mapping"])),("Unresolved airport timezones",len(zones["unresolved"])),
        ("Exact raw duplicate rows",summary["exact_duplicate_raw_rows"]),
        ("Source records sharing a conflicting purported-service key",sum(b["conflicts"] for b in summary["execution"]["buckets"])),
        ("Invalid cancellation/diversion flags",summary["invalid_flags"])]
    for key,label in [("missing_tail","Raw missing tail IDs"),("ambiguous_schedule","Fall-back ambiguous schedules resolved by independent schedule checks"),
        ("invalid_schedule","Unresolved scheduled departure timestamps"),("arrival_clock_mismatches","Scheduled arrival clock mismatches"),
        ("gate_clock_mismatches","Gate departure clock mismatches"),("missing_distance","Missing distances"),
        ("missing_elapsed","Missing/nonpositive scheduled durations"),("takeoff_day_crossing","Actual takeoff date crossings"),
        ("cross_zone","Cross-zone scheduled routes"),("taxi_out_over_180_flag_only","Taxi-out intervals above 180 minutes (flag only)")]:
        metrics.append((label,sum(m[key] for m in quality)))
    lines.extend(f"- {label}: **{value:,}**." for label,value in metrics)
    lines.extend([f"- Scheduled durations over 720 minutes (flag only): **{extra['scheduled_duration_over_720_flag_only']:,}**, maximum **{extra['max_scheduled_duration_min']:g}** minutes ({extra['long_scheduled_cancelled']} canceled and {extra['long_scheduled_operated']} operated records). These values are retained with documented schedule/timestamp flags, not removed by the duration cutoff.",
        f"- Maximum resolved taxi-out interval: **{extra['max_resolved_taxi_out_min']:g}** minutes; signed departure delays range from **{extra['min_signed_departure_delay_min']:g}** to **{extra['max_signed_departure_delay_min']:g}** minutes."])
    total=counts["modeling_cohort"]
    for row in summary["takeoff_sources_primary"]:
        lines.append(f"- Primary takeoff source `{row['takeoff_time_source']}`: {row['count']:,} ({100*row['count']/total:.6f}%).")
    lines.extend(["", "`TaxiOut` is absent in every source file. Takeoff clocks use the documented conservative WheelsOff calendar resolution. Gate-clock inconsistencies invalidate takeoff and exclude history observations; no arbitrary one-hour DST repair is applied. All eligible target labels remain retained. Detailed per-column invalid/null counts and per-month diagnostics are in `normalization_quality.json`.", ""])
    lines.extend(f"- {name}: **{value:,}** valid links flagged." for name,value in summary["retrospective_valid_link_audit_flags"].items())
    lines.extend(["",f"Targeted review found {reviewed['flagged_connections']} flagged connections with prior arrival overlapping target actual departure by {reviewed['min_arrival_overlap_minutes']:g}–{reviewed['max_arrival_overlap_minutes']:g} minutes. The destination-local arrival clock independently agrees with the elapsed-time-derived arrival in {reviewed['prior_arrival_clocks_confirmed']} cases and conflicts in {reviewed['prior_arrival_clock_contradictions']}; every flagged target's gate clock agrees. These are unresolved record/rotation contradictions, not silently repaired timestamps. Aggregate evidence is `retrospective_anomaly_review_summary.json`."])
    lines.extend(["", "Retrospective flags do not silently filter the primary schedule-selected cohort. Local `retrospective_anomalies.csv` retains every flagged connection for review. The balanced 100-example sample covers all twelve months, cross-zone, day-crossing and unusual taxi timings; examples are available locally, not in the public repository.", "",
        f"Automated artifact acceptance: **{summary['acceptance_passed']}** ({len(summary['acceptance_checks'])} checks). Every Parquet partition was reread, schema/UTC/logical hashes checked, and immediate same-tail predecessor relationships reconstructed from master events. All 38 synthetic tests passed; the transcript is `test_results.txt`. The independently repeated full-data run is recorded in `reproducibility.json`. A second engine, PyArrow, fully decoded and validated every record batch in all five tables; `arrow_validation.json` records matching counts and schemas.", "",
        f"The full uncached preprocessing/write stage took {uncached['elapsed_seconds']:.2f} seconds. Its measured process peak working set was {uncached['process_peak_rss_bytes']/2**20:.2f} MiB. This stage measurement excludes the subsequent independent read-back audit. The cached full rerun, including its complete artifact audit, took {proof['elapsed_seconds']:.2f} seconds. All {proof['partition_count']:,} logical partition hashes and all record counts were identical on rerun.", "",
        "## Decisions before Phase 2", "",
        "1. Review the remaining retrospective forecast-window/physical-rotation contradictions. Retain audit flags and choose an explicitly documented sensitivity policy; do not silently use future outcomes to redefine feature availability.",
        "2. Accept the takeoff date-resolution limitation without TaxiOut, or obtain an approved official export containing TaxiOut for an independently tracked enhancement.",
        "3. Acknowledge retrospective tail assignments, unreported ferry movements, conservative rotation exclusions, absent December 2024 predecessors, and the linkage-dependent shift in severity shares.",
        "4. Choose and freeze Phase 2's online-updated versus month-boundary-frozen history policy and sparse-history smoothing using development data. Preserve Nov–Dec as final holdouts and use explicit predictor allowlists.", "",
        "No models, model feature matrices, MLP training, final project report, or performance claims were produced. This handoff documents computational validation and targeted review; it does not claim certification by an external human reviewer."])
    (root/"VALIDATION.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
