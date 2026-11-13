"""CSV fallback checks do not relax official experiment eligibility."""

from pathlib import Path
import importlib.util
import json
import polars as pl
import pytest
from flight_delay.experiments.config import file_hash
from flight_delay.experiments.synthetic import make_fixture

spec = importlib.util.spec_from_file_location(
    "csv_readiness", Path(__file__).resolve().parents[1] / "scripts/check_phase3_csv.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def csv_fixture(tmp_path):
    make_fixture(tmp_path, 6)
    features = pl.read_parquet(
        list((tmp_path / "features/full_features").glob("year=*/month=*/*.parquet"))
    )
    labels = pl.read_parquet(
        list((tmp_path / "features/targets").glob("year=*/month=*/*.parquet"))
    )
    metadata = pl.read_parquet(
        list((tmp_path / "features/metadata").glob("year=*/month=*/*.parquet"))
    )
    cohort = labels.join(metadata, on="target_record_id", validate="1:1").with_columns(
        pl.col("prediction_time_utc")
        .dt.to_string("%Y-%m-%dT%H:%M:%S%:z")
        .alias("target_crs_dep_utc")
    )
    fp, cp = tmp_path / "features.csv", tmp_path / "cohort.csv"
    features.write_csv(fp)
    cohort.write_csv(cp)
    checksum = tmp_path / "checksums.sha256"
    checksum.write_text(f"{file_hash(fp)}  features.csv\n")
    validation = tmp_path / "cohort.validation.json"
    validation.write_text(
        json.dumps(
            dict(
                verified=True,
                rows=cohort.height,
                size_bytes=cp.stat().st_size,
                class_counts=cohort.group_by("target_class")
                .len()
                .sort("target_class")
                .to_dicts(),
            )
        )
    )
    return fp, cp, checksum, validation


def test_csv_export_full_ids_labels_and_chronological_subset(tmp_path):
    paths = csv_fixture(tmp_path)
    sample, report = module.verify_exports(*paths, rows_per_month=3)
    assert report["rows"] == 72 and report["full_id_alignment"]
    assert sample.height == 12
    assert set(sample["source_month"].to_list()) == {1, 2, 3, 4}
    assert all(
        i.endswith(("00000", "00001", "00002")) for i in sample["target_record_id"]
    )
    assert all(c in sample.columns for c in module.ALL)


def test_csv_checksum_corruption_refuses_model_input(tmp_path):
    fp, cp, checksum, validation = csv_fixture(tmp_path)
    checksum.write_text("0" * 64 + "  features.csv\n")
    with pytest.raises(ValueError, match="checksum"):
        module.verify_exports(fp, cp, checksum, validation, 3)


def test_csv_invalid_labels_fail_before_any_model_fit(tmp_path):
    fp, cp, checksum, validation = csv_fixture(tmp_path)
    cohort = pl.read_csv(cp).with_columns(pl.lit(2).alias("target_class"))
    cohort.write_csv(cp)
    meta = json.loads(validation.read_text())
    meta["size_bytes"] = cp.stat().st_size
    validation.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="thresholds"):
        module.verify_exports(fp, cp, checksum, validation, 3)
