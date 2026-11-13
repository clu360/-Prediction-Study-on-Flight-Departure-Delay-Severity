"""Verify complete CSV exports and produce a real-data chronological smoke run.

This is a readiness diagnostic, NOT official tuning or final-holdout evaluation.
It never alters the upstream exports or the official Phase III gate/selection.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
import joblib
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from flight_delay.experiments.config import Phase3Config, digest, file_hash
from flight_delay.experiments.evaluation import evaluate
from flight_delay.experiments.results import (
    prediction_frame,
    write_frame,
    write_json,
    write_model,
)
from flight_delay.experiments.runner import Budget, fit_estimator, predict_frozen
from flight_delay.features.transforms import FoldTransformer
from flight_delay.features.views import ALL, CATEGORICAL, VIEWS

ID = "target_record_id"


def csv_schema():
    integers = {
        "g1_day_of_week",
        "g1_month",
        "g3_airport_scheduled_volume_7d",
        "g3_airline_airport_count_28d",
    }
    return {
        ID: pl.String,
        **{
            c: pl.Int64
            if c in integers
            else (pl.String if c in CATEGORICAL else pl.Float64)
            for c in ALL
        },
    }


def chronological_sample(cohort, rows_per_month):
    if rows_per_month < 1:
        raise ValueError("Positive chronological subset size required")
    subset = (
        cohort.filter(pl.col("source_month").is_in([1, 2, 3, 4]))
        .sort(["source_month", "target_crs_dep_utc", ID])
        .group_by("source_month", maintain_order=True)
        .head(rows_per_month)
        .sort(ID)
    )
    if subset[ID].n_unique() != subset.height or set(
        subset["source_month"].to_list()
    ) != {1, 2, 3, 4}:
        raise ValueError("Missing months or duplicate subset IDs")
    return subset


def verify_exports(
    features_csv, cohort_csv, checksums, cohort_validation, rows_per_month
):
    print("Verifying CSV checksums", flush=True)
    expected = {
        line.split()[1]: line.split()[0]
        for line in Path(checksums).read_text().splitlines()
        if line.strip()
    }
    hashes = {"features": file_hash(features_csv), "cohort": file_hash(cohort_csv)}
    if hashes["features"] != expected[Path(features_csv).name]:
        raise ValueError("Feature CSV checksum disagrees with validated export")
    metadata = json.loads(Path(cohort_validation).read_text())
    if (
        not metadata.get("verified")
        or Path(cohort_csv).stat().st_size != metadata["size_bytes"]
    ):
        raise ValueError("Cohort CSV does not match export validation metadata")
    features = pl.scan_csv(features_csv, schema_overrides=csv_schema())
    if features.collect_schema().names() != [ID] + ALL:
        raise ValueError("CSV must have ID plus exactly the ordered 20 predictors")
    print("Reading complete cohort CSV and feature ID/month columns", flush=True)
    cohort = (
        pl.scan_csv(
            cohort_csv,
            schema_overrides={
                ID: pl.String,
                "source_month": pl.Int8,
                "target_class": pl.Int8,
                "target_dep_delay_min": pl.Float64,
            },
        )
        .select(
            ID,
            "source_month",
            "target_class",
            "target_dep_delay_min",
            "target_origin",
            "target_crs_dep_utc",
        )
        .with_columns(
            pl.col("target_crs_dep_utc").str.to_datetime(time_zone="UTC", strict=True)
        )
        .collect(engine="streaming")
    )
    keys = features.select(ID, "g1_month").collect(engine="streaming")
    n = cohort.height
    if (
        n != metadata["rows"]
        or keys.height != n
        or cohort[ID].null_count()
        or keys[ID].null_count()
    ):
        raise ValueError("Full CSV population count/null ID mismatch")
    if cohort[ID].n_unique() != n or keys[ID].n_unique() != n:
        raise ValueError("Duplicate CSV target IDs")
    matched = keys.join(
        cohort.select(ID, "source_month"), on=ID, how="inner", validate="1:1"
    )
    if (
        matched.height != n
        or matched.filter(pl.col("g1_month") != pl.col("source_month")).height
    ):
        raise ValueError("CSV feature/cohort ID or local-month alignment mismatch")
    invalid = cohort.filter(
        pl.col("target_class").is_null()
        | ~pl.col("target_class").is_in([0, 1, 2])
        | pl.col("target_dep_delay_min").is_null()
        | ~pl.col("target_dep_delay_min").is_finite()
        | (
            pl.col("target_class")
            != pl.when(pl.col("target_dep_delay_min") < 15)
            .then(0)
            .when(pl.col("target_dep_delay_min") < 60)
            .then(1)
            .otherwise(2)
        )
    )
    if invalid.height:
        raise ValueError("CSV labels violate signed departure-delay thresholds")
    counts = cohort.group_by("target_class").len().sort("target_class").to_dicts()
    if counts != metadata["class_counts"]:
        raise ValueError("CSV class supports differ from validated export")
    subset = chronological_sample(cohort, rows_per_month)
    month_counts = cohort.group_by("source_month").len().sort("source_month").to_dicts()
    del keys, matched, invalid, cohort
    gc.collect()
    sample = (
        features.filter(pl.col("g1_month").is_in([1, 2, 3, 4]))
        .join(subset.lazy(), on=ID, how="inner", validate="1:1")
        .sort(ID)
        .collect(engine="streaming")
    )
    if sample.height != subset.height or sample[ID].n_unique() != subset.height:
        raise ValueError("CSV smoke sample lost or duplicated target IDs")
    report = dict(
        rows=n,
        source_predictors=20,
        hashes=hashes,
        class_counts=counts,
        month_counts=month_counts,
        full_id_alignment=True,
        unique_ids=True,
        signed_label_thresholds=True,
        feature_export_checksum_verified=True,
        cohort_export_metadata_verified=True,
    )
    return sample, report


def run_models(cfg, sample, output, rows_per_month, verification):
    if output.exists():
        raise ValueError(
            "Use a fresh diagnostic output directory; existing outputs are preserved"
        )
    output.mkdir(parents=True)
    train = sample.filter(pl.col("source_month").is_in([1, 2, 3])).sort(ID)
    test = sample.filter(pl.col("source_month") == 4).sort(ID)
    if set(train[ID]) & set(test[ID]):
        raise ValueError("Chronological smoke IDs overlap")
    print(
        f"Chronological CSV smoke: {train.height:,} Jan-Mar train; {test.height:,} April score",
        flush=True,
    )
    budget = Budget(cfg)
    records = []
    for name in cfg["models"]:
        for view in ["independent"] if name == "baseline" else cfg["views"]:
            params = cfg.grids[name][
                0
            ]  # predeclared lexical-first configuration; NO selection using scores
            start = time.monotonic()
            if name == "baseline":
                transformer = None
                X_train, X_test = (
                    csr_matrix((train.height, 0), dtype=np.float32),
                    csr_matrix((test.height, 0), dtype=np.float32),
                )
                unknown = {}
            else:
                transformer = FoldTransformer(
                    VIEWS[view], scale=cfg["scale"], batch_rows=cfg["matrix_batch_rows"]
                ).fit(train)
                X_train, X_test = (
                    transformer.transform(train),
                    transformer.transform(test),
                )
                unknown = transformer.unknown_counts(test)
            preprocessing = time.monotonic() - start
            estimator, fit_seconds, messages = fit_estimator(
                cfg, name, params, X_train, train["target_class"].to_numpy(), budget
            )
            p, inference_seconds = predict_frozen(estimator, transformer, X_test)
            fit_id = digest(
                dict(
                    model=name,
                    view=view,
                    params=params,
                    model_hash=joblib.hash(estimator),
                    transformer_hash=joblib.hash(transformer),
                    train_ids=digest(train[ID].to_list()),
                )
            )
            predictions = prediction_frame(
                output.name,
                name,
                view,
                "csv_readiness_smoke",
                "2025-04",
                test[ID].to_numpy(),
                test["target_class"].to_numpy(),
                p,
                fit_id,
            )
            write_frame(output / f"predictions_{name}_{view}.parquet", predictions)
            write_frame(output / f"predictions_{name}_{view}.csv", predictions)
            write_model(
                output / f"model_{name}_{view}.joblib",
                dict(estimator=estimator, transformer=transformer),
            )
            records.append(
                dict(
                    model=name,
                    feature_view=view,
                    params=params,
                    model_fit_id=fit_id,
                    metrics=evaluate(test["target_class"].to_numpy(), p),
                    unknown_categories=unknown,
                    warnings=messages,
                    preprocessing_seconds=preprocessing,
                    fit_seconds=fit_seconds,
                    inference_seconds=inference_seconds,
                    train_shape=list(X_train.shape),
                    test_shape=list(X_test.shape),
                    peak_rss_bytes=budget.peak,
                )
            )
            print(
                f"Saved {name}/{view}: {predictions.height:,} probability vectors",
                flush=True,
            )
    report = dict(
        status="csv_models_verified",
        study_results=False,
        evaluation_stage="csv_readiness_smoke",
        scope="Readiness diagnostic only: not hyperparameter tuning, official backtest, or final evaluation",
        split="Earliest scheduled targets per local month; Jan-Mar train / April score",
        rows_per_month=rows_per_month,
        original_fold_manifest_verified=False,
        limitation="Full original Phase I/II audit and fold-manifest verification remain separate requirements for official experiments",
        train_rows=train.height,
        test_rows=test.height,
        model_view_combinations=len(records),
        train_ids_hash=digest(train[ID].to_list()),
        test_ids_hash=digest(test[ID].to_list()),
        csv_verification=verification,
        models=records,
    )
    write_json(output / "readiness.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase3.yaml"))
    parser.add_argument("--rows-per-month", type=int, default=2000)
    args = parser.parse_args()
    cfg = Phase3Config.load(args.config)
    sample, verification = verify_exports(
        cfg.root / "data/features/exports/full_features.csv",
        cfg.root / "data/processed/csv/modeling_cohort.csv",
        cfg.root / "data/features/exports/checksums.sha256",
        cfg.root / "data/processed/csv/modeling_cohort.validation.json",
        args.rows_per_month,
    )
    signature = digest(
        dict(
            input_hashes=verification["hashes"],
            rows_per_month=args.rows_per_month,
            script_hash=file_hash(Path(__file__)),
            grids=cfg.grids,
            seed=cfg["seed"],
        )
    )[:16]
    output = cfg.path("output_dir") / "csv_readiness" / signature
    if output.exists():
        print(
            json.dumps(
                dict(
                    status="existing_diagnostic_preserved", private_output=str(output)
                ),
                indent=2,
            )
        )
        return
    result = run_models(cfg, sample, output, args.rows_per_month, verification)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "study_results",
                    "train_rows",
                    "test_rows",
                    "model_view_combinations",
                ]
            },
            indent=2,
        )
    )
    print("Private output:", output)


if __name__ == "__main__":
    main()
