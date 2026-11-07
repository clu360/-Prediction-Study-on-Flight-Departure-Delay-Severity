"""Rerun the real CLI and compare complete sorted logical output hashes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--raw-dir",type=Path,required=True)
    parser.add_argument("--processed-dir",type=Path,default=Path("data/processed"))
    args=parser.parse_args()
    audit=args.processed_dir/"audit"
    before=json.loads((audit/"logical_hashes.json").read_text())
    before_counts=json.loads((audit/"preprocessing_summary.json").read_text())["counts"]
    before_execution=json.loads((audit/"execution.json").read_text())
    signature=json.loads((audit/"run_signature.json").read_text())
    start=time.monotonic()
    result=subprocess.run([sys.executable,"-m","flight_delay.cli","preprocess","--raw-dir",str(args.raw_dir),
        "--out-dir",str(args.processed_dir),"--year","2025","--config","configs/preprocessing.yaml"],check=False)
    if result.returncode:
        return result.returncode
    after=json.loads((audit/"logical_hashes.json").read_text())
    after_counts=json.loads((audit/"preprocessing_summary.json").read_text())["counts"]
    checks=dict(same_run_signature=signature==json.loads((audit/"run_signature.json").read_text()),
        identical_all_partition_logical_hashes=before==after,identical_record_counts=before_counts==after_counts)
    proof=dict(checks=checks,passed=all(checks.values()),partition_count=len(after),
        run_key=signature["run_key"],elapsed_seconds=time.monotonic()-start,
        combined_logical_sha256=hashlib.sha256(json.dumps(after,sort_keys=True).encode()).hexdigest(),
        counts=after_counts,uncached_execution=before_execution)
    (audit/"reproducibility.json").write_text(json.dumps(proof,indent=2),encoding="utf-8")
    print(json.dumps(proof,indent=2))
    return 0 if proof["passed"] else 1


if __name__=="__main__":
    sys.exit(main())
