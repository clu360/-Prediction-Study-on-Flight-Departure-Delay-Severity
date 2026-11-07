"""Investigate retrospective timing flags without changing linkage eligibility."""
from __future__ import annotations
import argparse
from datetime import timedelta
import json
from pathlib import Path
from zoneinfo import ZoneInfo, reset_tzpath
import polars as pl


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--processed-dir",type=Path,default=Path("data/processed"))
    args=parser.parse_args()
    reset_tzpath([])  # Force the pinned tzdata package rather than host tz files.
    root=args.processed_dir
    audit=root/"audit"
    signature=json.loads((audit/"run_signature.json").read_text())
    flags=pl.read_csv(audit/"retrospective_anomalies.csv")
    ids=flags["target_record_id"].to_list()+flags["prior_record_id"].to_list()
    paths=list((root.parent/"interim"/signature["run_key"]).glob("bucket-*/*.parquet"))
    events=pl.scan_parquet(paths).filter(pl.col("record_id").is_in(ids)).select(
        "record_id","actual_gate_departure_utc","actual_elapsed_time","arr_time","dest_timezone",
        "gate_departure_clock_ok").collect(engine="streaming")
    lookup={r["record_id"]:r for r in events.to_dicts()}
    rows=[]
    for flag in flags.to_dicts():
        target=lookup[flag["target_record_id"]]
        prior=lookup[flag["prior_record_id"]]
        arrival=prior["actual_gate_departure_utc"]+timedelta(minutes=prior["actual_elapsed_time"])
        local=arrival.astimezone(ZoneInfo(prior["dest_timezone"]))
        raw=int(float(prior["arr_time"])) if prior["arr_time"] else None
        agrees=(raw//100*60+raw%100)%1440 == local.hour*60+local.minute if raw is not None else None
        rows.append(dict(target_record_id=flag["target_record_id"],prior_record_id=flag["prior_record_id"],
            arrival_overlap_minutes=(arrival-target["actual_gate_departure_utc"]).total_seconds()/60,
            prior_arrival_clock_matches=agrees,target_gate_clock_ok=target["gate_departure_clock_ok"]))
    if rows:
        pl.DataFrame(rows).write_csv(audit/"retrospective_anomaly_review.csv")
    summary=dict(flagged_connections=len(rows),
        min_arrival_overlap_minutes=min((r["arrival_overlap_minutes"] for r in rows),default=None),
        max_arrival_overlap_minutes=max((r["arrival_overlap_minutes"] for r in rows),default=None),
        prior_arrival_clocks_confirmed=sum(r["prior_arrival_clock_matches"] is True for r in rows),
        prior_arrival_clock_contradictions=sum(r["prior_arrival_clock_matches"] is False for r in rows),
        target_gate_clocks_confirmed=sum(r["target_gate_clock_ok"] is True for r in rows),
        linkage_eligibility_unchanged=True)
    (audit/"retrospective_anomaly_review_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
