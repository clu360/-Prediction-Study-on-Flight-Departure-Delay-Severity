"""Independent second-engine full-batch Parquet readability and count validation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import pyarrow.dataset as ds


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--processed-dir",type=Path,default=Path("data/processed"))
    args=parser.parse_args()
    root=args.processed_dir
    summary=json.loads((root/"audit/preprocessing_summary.json").read_text())
    start=time.monotonic()
    result={}
    for table in ["flights_clean","flight_connections","modeling_cohort","history_departure_events","folds"]:
        dataset=ds.dataset(root/table,format="parquet")
        rows=0
        for batch in dataset.scanner(batch_size=32768,use_threads=False).to_batches():
            batch.validate(full=True)
            rows+=batch.num_rows
        result[table]=dict(rows=rows,expected_rows=summary["counts"][table],passed=rows==summary["counts"][table],
                           schema=str(dataset.schema))
        print(f"Arrow full-batch validation {table}: {rows:,}",flush=True)
    proof=dict(engine="pyarrow==22.0.0",tables=result,passed=all(t["passed"] for t in result.values()),
               elapsed_seconds=time.monotonic()-start)
    (root/"audit/arrow_validation.json").write_text(json.dumps(proof,indent=2),encoding="utf-8")
    if not proof["passed"]:
        raise ValueError("Independent Arrow row counts do not match")


if __name__=="__main__":
    main()
