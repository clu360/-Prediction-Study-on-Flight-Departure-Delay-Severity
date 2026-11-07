"""Export aggregate-only Phase II evidence and generate the feature dictionary."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import polars as pl
from flight_delay.features.views import ALL, G1_MAP, G2_MAP, VIEWS
from flight_delay.features.validation import scan

DESCRIPTIONS = {
    "g1_carrier": ("target_carrier", "Reporting carrier", "Before prediction"),
    "g1_origin": ("target_origin", "Departure IATA code", "Before prediction"),
    "g1_dest": ("target_dest", "Destination IATA code", "Before prediction"),
    "g1_dep_hour_sin": ("target_crs_dep_local_hour", "sin(2*pi*hour/24)", "Before prediction"),
    "g1_dep_hour_cos": ("target_crs_dep_local_hour", "cos(2*pi*hour/24)", "Before prediction"),
    "g1_day_of_week": ("target_day_of_week", "ISO weekday: Monday=1, Sunday=7; categorical", "Before prediction"),
    "g1_month": ("target_month", "Month 1..12; categorical", "Before prediction"),
    "g1_crs_elapsed_min": ("target_crs_elapsed_min", "Scheduled minutes, unchanged", "Before prediction"),
    "g1_distance_miles": ("target_distance_miles", "Scheduled miles, unchanged", "Before prediction"),
    "g2_prior_origin": ("prior_origin", "Incoming leg origin, unchanged", "At prediction"),
    "g2_prior_dep_delay_min": ("prior_dep_delay_min", "Signed prior departure delay, unchanged", "At prediction"),
    "g2_prior_crs_elapsed_min": ("prior_crs_elapsed_min", "Prior scheduled minutes, unchanged", "Before prediction"),
    "g2_scheduled_turnaround_min": ("scheduled_turnaround_min", "Target scheduled departure minus prior scheduled arrival, unchanged", "Before prediction"),
    "g2_minutes_until_crs_dep_at_prediction": ("minutes_until_target_crs_dep_at_prediction", "Target scheduled departure minus prediction instant, unchanged", "At prediction"),
    "g3_airport_moderate_rate_28d": ("Valid prior departure observations", "(M_a+20*p_global_mod)/(N_a+20)", "Strictly earlier observations"),
    "g3_airport_severe_rate_28d": ("Valid prior departure observations", "(S_a+20*p_global_sev)/(N_a+20)", "Strictly earlier observations"),
    "g3_airport_mean_dep_delay_28d": ("Valid prior signed DepDelay", "Airport signed-delay sum/N_a", "Strictly earlier observations"),
    "g3_airport_scheduled_volume_7d": ("Full normalized master schedule", "Count origin schedules in [t-7 days,t), including eventual cancellations/diversions", "Earlier scheduled instants"),
    "g3_airline_airport_severe_rate_28d": ("Valid prior departure observations", "(S_ac+20*airport_severe)/(N_ac+20)", "Strictly earlier observations"),
    "g3_airline_airport_count_28d": ("Valid prior departure observations", "N_ac in [t-28 days,t)", "Strictly earlier observations"),
}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--features-dir",type=Path,default=Path("data/features"))
    args=parser.parse_args()
    audit=args.features_dir/"audit"
    summary=json.loads((audit/"phase2_summary.json").read_text())
    proof=json.loads((audit/"reproducibility.json").read_text())
    smoke=json.loads((audit/"matrix_smoke.json").read_text())
    own=json.loads((audit/"self_outcome_oracle.json").read_text())
    if not (summary["acceptance_passed"] and proof["passed"] and own["passed"] and len(smoke)==4):
        raise ValueError("Incomplete or failing Phase II acceptance evidence")
    public=Path("reports/phase2"); public.mkdir(parents=True,exist_ok=True)
    names=["phase2_summary.json","feature_schema.json","history_coverage_by_month.csv","history_coverage_by_airport.csv",
        "feature_missingness.csv","feature_missingness_by_month.csv","feature_missingness_by_airport.csv",
        "reproducibility.json","matrix_smoke.json","self_outcome_oracle.json","run_signature.json","single_file_exports.json"]
    for name in names:
        shutil.copyfile(audit/name,public/name)
    source=json.loads((audit/"source_contract.json").read_text())
    sanitized={k:source[k] for k in ["input_sha256","tables","master_rows","phase1_signature"]}
    sanitized.update(source_partition_count=len(source["files"]),processed_dir="data/processed",
                     master_dir="data/interim/"+source["phase1_signature"])
    (public/"source_contract.json").write_text(json.dumps(sanitized,indent=2)+"\n",encoding="utf-8")
    fields=json.loads((audit/"feature_schema.json").read_text())
    stats={r["feature"]:r for r in summary["feature_distributions"]}
    count_distributions={}
    for name in ["global_history_count_28d","airport_history_count_28d"]:
        count_distributions[name]=scan(args.features_dir,"metadata").select([
            getattr(pl.col(name),method)().alias(method) for method in ["mean","median","min","max"]
        ]).collect(engine="streaming").to_dicts()[0]
    count_distributions["airline_airport_count_28d"]={method:stats["g3_airline_airport_count_28d"][method]
        for method in ["mean","median","min","max"]}
    (audit/"history_count_distributions.json").write_text(json.dumps(count_distributions,indent=2)+"\n",encoding="utf-8")
    shutil.copyfile(audit/"history_count_distributions.json",public/"history_count_distributions.json")
    dictionary=["# Phase II feature dictionary", "", "One row per unchanged Phase I target_record_id. The full_features table contains one metadata ID column plus exactly twenty predictors; labels and other metadata are separate. Nulls are retained in source features and handled by training-fold-only imputation. All G3 outcome windows are [t−28 days,t); the schedule window is [t−7 days,t). alpha=20. The target's own outcome is explicitly excluded.", "",
        "| Feature | dtype | Source | Formula | Availability | Group | Missing count | Missing fraction |", "|---|---|---|---|---|---|---:|---:|"]
    for name in ALL:
        src,formula,available=DESCRIPTIONS[name]
        row=stats[name]
        dictionary.append(f"| `{name}` | {fields['full_features'][name]} | {src} | {formula} | {available} | {name[:2].upper()} | {row['missing_count']:,} | {row['missing_fraction']:.8f} |")
    dictionary.extend(["", "## Missingness and support", "", "No global history yields null rates. With global but no airport history, airport rates use global rates and the airport mean stays null. With no airline-at-origin history, its severe rate equals the airport severe rate and its count is zero. Counts/volumes are nonnegative integers. Early January has no December 2024 buffer. No rows are dropped.", "",
        "## Non-predictor contracts", "", "`targets`: target_record_id, target_class (Int8, 0/1/2), target_dep_delay_min (Float64 signed minutes). Only target_class is y. Both outcome columns are forbidden in X.", "",
        "`metadata`: IDs, local flight_date/source_month, origin/carrier, UTC prediction/scheduled times, timestamp/takeoff quality, support/fallback diagnostics and Phase I retrospective flags. All timestamps remain timezone-aware UTC. None of these metadata columns is automatically selected as a predictor.", "",
        "Exact feature, target and metadata dtypes are in `reports/phase2/feature_schema.json`. Numeric distributions and categorical cardinalities for all twenty features are in `reports/phase2/feature_missingness.csv`; grouped missing counts and history coverage are adjacent CSVs. Unknown test categories use train-fitted all-zero one-hot blocks; numeric nulls use train-only medians (zero only for entirely null training columns)."])
    Path("docs/phase2_feature_dictionary.md").write_text("\n".join(dictionary)+"\n",encoding="utf-8")
    lines=["# Phase II validation handoff", "", "The complete feature pipeline was executed against the validated 2025 Phase I outputs. No aircraft linkage was recomputed and no classifier or MLP was trained.", "",
        "## Sample and schema reconciliation", "",f"Phase I cohort: **{source['tables']['modeling_cohort']['rows']:,}** targets. Full master schedule: **{source['master_rows']:,}** events, including cancellations/diversions. History event store: **{source['tables']['history_departure_events']['rows']:,}** observations. All Phase I partition schemas match the repository declaration and all counts reconcile.", "",
        "| Phase II artifact | Rows | Stored columns |", "|---|---:|---:|"]
    lines.extend(f"| {table} | {summary['counts'][table]:,} | {len(fields[table])} |" for table in fields)
    lines.extend(["", "The mixed-type sample-by-feature matrix is N×20; target_record_id is an additional join column. All three artifacts have exactly the same unique IDs, labels match Phase I, and no sample is dropped for missing history. Local output: `data/features/{full_features,targets,metadata}/year=2025/month=MM/part-AIRPORT.parquet`.", "",
        "## Four feature configurations", "", "| View | Source columns |", "|---|---:|"])
    lines.extend(f"| {view} | {len(columns)} |" for view,columns in VIEWS.items())
    lines.extend(["", "The exact ordered column lists are `configs/feature_groups.yaml`. G1 has nine columns, G2 five and G3 six. No minimal-delay-rate predictor, raw ID, timestamp, target outcome or prior arrival outcome enters X. Source/formula/availability/dtype details appear in [the dictionary](../docs/phase2_feature_dictionary.md).", "",
        "## Full April matrix smoke tests", "", "All rows in each three-month training window and following test month were used. Every view has identical train IDs and identical test IDs. Encoders, median imputers and optional scalers fit only January–March and remain frozen for April. Matrices are sparse Float32 CSR; labels remain 0/1/2.", "",
        "| View | X_train | X_test | Train CSR MiB | Test CSR MiB |", "|---|---|---|---:|---:|"])
    lines.extend(f"| {r['feature_set']} | {r['X_train_shape']} | {r['X_test_shape']} | {r['train_csr_bytes']/2**20:.2f} | {r['test_csr_bytes']/2**20:.2f} |" for r in smoke)
    lines.extend(["", "Test-only categories, including April's unseen month category, are handled by train-fitted unknown-category rules; counts are recorded per view in `phase2/matrix_smoke.json`. No test distributions were used to fit transformations.", "",
        "## History policy and coverage", "", "Outcomes use strict 28-day as-of histories, independently of 7-day master scheduled volume. Global-to-airport-to-airline/airport smoothing uses alpha=20. Earlier test-month observations may update later feature values, while transformations and later model parameters remain frozen. Null/zero history fallbacks are retained and audited; no December 2024 data are invented.", ""])
    lines.extend(f"- {name}: **{count:,}**." for name,count in summary["history_flags"].items())
    lines.extend(["", "| 28-day history support | Mean | Median | Min | Max |", "|---|---:|---:|---:|---:|"])
    lines.extend(f"| {name} | {values['mean']:.4f} | {values['median']:g} | {values['min']:,} | {values['max']:,} |"
                 for name,values in count_distributions.items())
    lines.extend(["", "All fourteen retrospective rotation contradictions are retained as metadata. The one already-departed target is explicitly protected against self-outcome leakage; its event is subtracted from its own global/airport/cell history without changing the sample. An independent direct oracle verifies this safeguard. History availability is left-censored in early January.", "",
        "## Feature distributions and grouped missingness", "", "`phase2/feature_missingness.csv` gives dtype, group, null count/fraction and every numeric feature's mean/median/min/max/std/1st/99th percentiles or categorical cardinality. Grouped missingness is in feature_missingness_by_month.csv and feature_missingness_by_airport.csv. History support/fallback counts and mean/median/min/max airport counts are in history_coverage_by_month.csv and history_coverage_by_airport.csv, including ATL, DFW, DEN, ORD and LAX.", "",
        "| Feature | Missing | Min / cardinality | Median | Max |", "|---|---:|---|---|---|"])
    for name in ALL:
        s=stats[name]
        lines.append(f"| {name} | {s['missing_count']:,} | {s.get('cardinality',s.get('min'))} | {s.get('median','—')} | {s.get('max','—')} |")
    exports=json.loads((audit/"single_file_exports.json").read_text())
    lines.extend(["", "## Complete single-file exports", "", "Alongside the partitioned tables, the requested complete annual feature table is exported as two separate files under `data/features/exports/`: `full_features.csv` and `full_features.parquet`. Each has the same target IDs and exactly twenty predictors plus the ID column. Labels and metadata remain separate to preserve the X/y boundary. Every column/row was checked against the validated feature table, and the Parquet file was independently decoded with Arrow.", ""])
    lines.extend(f"- {format}: {exports[format]['rows']:,} rows; {exports[format]['size_bytes']:,} bytes; SHA-256 `{exports[format]['sha256']}`." for format in ["csv","parquet"])
    execution=proof["initial_execution"]
    lines.extend(["", "## Verification and reproducibility", "",
        f"All **{len(summary['acceptance_checks'])}** full-data acceptance checks pass. Every output shard was reread and its schema/UTC/logical hash checked. G1/G2 values and all target IDs/labels match Phase I. Numeric finiteness, rate/simplex bounds, count bounds, fallback hierarchy and strict history cutoffs pass.", "",
        f"The independent boolean-window oracle checked {summary['independent_oracle']['sample_rows']} real samples / {summary['independent_oracle']['scalar_feature_comparisons']} scalar G3 values. The own-outcome oracle checked {own['sample_rows']} additional anomalous samples. Synthetic test results are in `phase2/test_results.txt`.", "",
        f"A complete rerun produced identical counts and all **{proof['partitions']:,}** canonical sorted logical hashes. Combined logical SHA-256: `{proof['combined_logical_sha256']}`. Source byte/configuration/code/dependency fingerprints are recorded. Phase I source fingerprints and fold bytes remain unchanged.", "",
        f"Initial validated source reconciliation/build/write stage: {execution['build_seconds']:.2f} seconds; source cache imported: {execution['source_cache_reused']}; peak process working set before the independent audit: {execution['process_peak_rss_bytes']/2**20:.2f} MiB. Cached complete rerun including its audit: {proof['elapsed_seconds']:.2f} seconds. Matrix summaries record cumulative process peak working set across validation and successive views, not isolated allocations; exact individual CSR sizes appear in the table above. The maximum observed cumulative peak was {max(r['process_peak_rss_bytes'] for r in smoke)/2**30:.3f} GiB. Full-year dense one-hot expansion was never allocated.", "",
        "## Review before Phase III", "", "Review a later sensitivity policy for retained retrospective contradictions; acknowledge January left censoring and Phase I tail/takeoff limitations; review unknown-category/all-null-column transformation rules. History regime, windows, smoothing and sample are locked here and were not selected using final holdout model scores. November/December remain final holdouts. No classifier fitting, predictive metrics, tuning, feature-importance analysis or final report conclusions were produced."])
    Path("reports/PHASE2_VALIDATION.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("Exported aggregate Phase II review evidence; flight-level artifacts remain local.")


if __name__=="__main__":
    main()
