"""Create the requested complete single-file CSV and Parquet feature tables."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import polars as pl
import pyarrow.parquet as pq
from flight_delay.features.validation import scan
from flight_delay.features.views import ALL


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--features-dir",type=Path,default=Path("data/features"))
    args=parser.parse_args()
    root=args.features_dir
    summary=json.loads((root/"audit/phase2_summary.json").read_text())
    if not summary["acceptance_passed"] or not (root/"audit/SUCCESS.json").exists():
        raise ValueError("Only validated complete features may be exported")
    source=scan(root,"full_features").select("target_record_id",*ALL)
    destination=root/"exports"
    destination.mkdir(exist_ok=True)
    expressions=[pl.len().alias("rows"),pl.struct(pl.all()).hash(seed=6740).sum().alias("row_hash_sum_1"),
                 pl.struct(pl.all()).hash(seed=2025).sum().alias("row_hash_sum_2")]
    expected=source.select(expressions).collect(engine="streaming").to_dicts()[0]
    result={}
    for format in ["parquet","csv"]:
        output=destination/f"full_features.{format}"
        pending=output.with_suffix(f".{format}.partial")
        print(f"Writing complete {format}: {expected['rows']:,} rows, ID + {len(ALL)} features",flush=True)
        if format=="parquet":
            source.sink_parquet(pending,compression="zstd",row_group_size=131072,maintain_order=True)
            exported=pl.scan_parquet(pending)
        else:
            source.sink_csv(pending,maintain_order=True)
            exported=pl.scan_csv(pending,schema=source.collect_schema())
        actual=exported.select(expressions).collect(engine="streaming").to_dicts()[0]
        if actual!=expected or exported.collect_schema()!=source.collect_schema():
            raise ValueError(f"Complete {format} export differs from validated feature table")
        if format=="parquet":
            rows=0
            for batch in pq.ParquetFile(pending).iter_batches(batch_size=65536):
                batch.validate(full=True); rows+=batch.num_rows
            if rows!=expected["rows"]:
                raise ValueError("Arrow full-Parquet row count mismatch")
        pending.replace(output)
        result[format]=dict(filename=output.name,rows=expected["rows"],columns=21,source_features=20,
            size_bytes=output.stat().st_size,sha256=sha256(output),all_values_verified=True)
        print(json.dumps(result[format]),flush=True)
    result["labels_location"]="targets/ (separate labels; never predictors)"
    result["metadata_location"]="metadata/ (separate identity, timestamps and audit flags)"
    (root/"audit/single_file_exports.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    (destination/"checksums.sha256").write_text("\n".join(
        f"{result[f]['sha256']}  {result[f]['filename']}" for f in ["csv","parquet"])+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
