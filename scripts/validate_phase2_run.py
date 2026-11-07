"""Repeat the full feature build and smoke-test all four April matrix views."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import psutil
import polars as pl
from flight_delay.features.builder import build
from flight_delay.features.config import FeatureConfig
from flight_delay.features.transforms import build_fold, matrix_summary
from flight_delay.features.validation import direct_oracle, scan
from flight_delay.features.views import VIEWS
from flight_delay.pipeline import write_json


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--features-dir",type=Path,default=Path("data/features"))
    parser.add_argument("--processed-dir",type=Path,default=Path("data/processed"))
    parser.add_argument("--skip-rerun",action="store_true")
    args=parser.parse_args()
    audit=args.features_dir/"audit"
    if not args.skip_rerun:
        before=json.loads((audit/"logical_hashes.json").read_text())
        counts=json.loads((audit/"phase2_summary.json").read_text())["counts"]
        first_execution=json.loads((audit/"execution.json").read_text())
        start=time.monotonic()
        result=build(args.processed_dir,args.features_dir,FeatureConfig.load(Path("configs/phase2.yaml")))
        after=json.loads((audit/"logical_hashes.json").read_text())
        proof=dict(passed=before==after and counts==result["counts"],
            identical_logical_hashes=before==after, identical_counts=counts==result["counts"],
            partitions=len(after), combined_logical_sha256=hashlib.sha256(json.dumps(after,sort_keys=True).encode()).hexdigest(),
            elapsed_seconds=time.monotonic()-start,initial_execution=first_execution)
        write_json(audit/"reproducibility.json",proof)
        if not proof["passed"]:
            raise ValueError("Full feature rerun was not reproducible")
    smoke=[]
    for view in VIEWS:
        start=time.monotonic()
        print(f"Full April matrix smoke: {view}",flush=True)
        bundle=build_fold("2025-04",view,features_dir=args.features_dir,processed_dir=args.processed_dir)
        row=matrix_summary(bundle,view)
        row.update(elapsed_seconds=time.monotonic()-start,
            process_peak_rss_bytes=getattr(psutil.Process().memory_info(),"peak_wset",psutil.Process().memory_info().rss))
        smoke.append(row)
        write_json(audit/"matrix_smoke.json",smoke)
        print(json.dumps(row),flush=True)
        del bundle
        gc.collect()
    if len({r["train_ids_sha256"] for r in smoke})!=1 or len({r["test_ids_sha256"] for r in smoke})!=1:
        raise ValueError("Feature configurations do not use identical samples")
    # The retained already-departed target must pass a direct self-outcome-free oracle.
    anomalous=scan(args.features_dir,"metadata").filter(pl.col("own_outcome_excluded")).collect(engine="streaming")
    if anomalous.height:
        sample=scan(args.features_dir,"full_features").join(anomalous.lazy(),on="target_record_id",how="inner").collect(engine="streaming")
        proof=direct_oracle(args.features_dir,sample)
    else:
        proof=dict(sample_rows=0,passed=True,scalar_feature_comparisons=0)
    write_json(audit/"self_outcome_oracle.json",proof)
    if not proof["passed"]:
        raise ValueError("Own-outcome exclusion did not pass independent verification")


if __name__=="__main__":
    main()
