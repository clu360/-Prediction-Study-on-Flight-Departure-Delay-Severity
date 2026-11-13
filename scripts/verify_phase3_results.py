"""Independently reread complete held-out predictions and recompute all metrics."""

from pathlib import Path
import argparse
import gc
import time
import joblib
import numpy as np
import polars as pl

from flight_delay.experiments.config import Phase3Config
from flight_delay.experiments.evaluation import evaluate, airport_metrics
from flight_delay.experiments.inputs import (
    verified_gate,
    memberships,
    table_scan,
    origins_for,
)
from flight_delay.experiments.results import (
    ResultStore,
    read_json,
    write_json,
    PROB_COLUMNS,
)
from flight_delay.experiments.runner import context, ids_hash, validate_selection


def equivalent(expected, actual):
    if isinstance(expected, dict):
        return set(expected) == set(actual) and all(
            equivalent(expected[k], actual[k]) for k in expected
        )
    if isinstance(expected, list):
        return len(expected) == len(actual) and all(
            equivalent(a, b) for a, b in zip(expected, actual)
        )
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return bool(np.isclose(expected, actual, atol=1e-10, rtol=1e-10))
    return expected == actual


def verify(cfg):
    started = time.monotonic()
    gate = verified_gate(cfg, "frozen_holdout")
    store = ResultStore(cfg.path("output_dir"), context(cfg, gate), True)
    selection = read_json(store.root / "development/selected_configs.json")
    validate_selection(cfg, store, selection)
    train = memberships(cfg.path("processed_dir"), "2025-11", "train").sort(
        "target_record_id"
    )
    train_hash = ids_hash(train["target_record_id"].to_list())
    reference = {}
    total_rows = 0
    verified = []
    for item in selection["selected"].values():
        name, view = item["model"], item["feature_view"]
        fit = store.load(f"final/fits/{name}_{view}")
        if (
            fit is None
            or fit["train_ids_hash"] != train_hash
            or fit["train_months"] != [8, 9, 10]
        ):
            raise ValueError(
                "Final training provenance differs from exact August-October manifest"
            )
        state = joblib.load(store.root / f"final/frozen_model_{name}_{view}.joblib")
        if (
            joblib.hash(state["estimator"]) != fit["model_hash"]
            or joblib.hash(state["transformer"]) != fit["transformer_hash"]
        ):
            raise ValueError("Serialized frozen state differs from its fit checksum")
        del state
        gc.collect()
        frames = []
        origins = []
        for month in [11, 12]:
            period = f"2025-{month:02}"
            record = store.load(f"final/months/{name}_{view}_{period}")
            frame = pl.read_parquet(
                store.root / f"final/predictions/{name}_{view}_{period}.parquet"
            )
            expected_ids = memberships(cfg.path("processed_dir"), period, "test").sort(
                "target_record_id"
            )
            if (
                frame["target_record_id"].to_list()
                != expected_ids["target_record_id"].to_list()
            ):
                raise ValueError(
                    "Saved holdout IDs differ from original Phase I membership"
                )
            labels = (
                table_scan(cfg.path("features_dir"), "targets", [month])
                .select("target_record_id", "target_class")
                .collect(engine="streaming")
            )
            checked = frame.select("target_record_id", "y_true").join(
                labels, on="target_record_id", validate="1:1"
            )
            if checked.filter(pl.col("y_true") != pl.col("target_class")).height:
                raise ValueError("Saved labels differ from validated targets")
            probabilities = frame.select(PROB_COLUMNS).to_numpy()
            if not np.array_equal(
                probabilities.argmax(axis=1), frame["y_pred"].to_numpy()
            ):
                raise ValueError("Saved argmax classes are wrong")
            values = origins_for(cfg, frame["target_record_id"].to_numpy(), month)
            metrics = evaluate(frame["y_true"].to_numpy(), probabilities)
            airports = airport_metrics(
                frame["y_true"].to_numpy(),
                probabilities,
                values,
                cfg["evaluation_airports"],
            )
            if not equivalent(record["metrics"], metrics) or not equivalent(
                record["airport_metrics"], airports
            ):
                raise ValueError(
                    "Recomputed monthly/airport metrics differ from saved results"
                )
            if any(
                record[k] != fit[k]
                for k in ["model_fit_id", "model_hash", "transformer_hash"]
            ):
                raise ValueError(
                    "November and December did not use the identical frozen fit"
                )
            digest = ids_hash(frame["target_record_id"].to_list())
            if month in reference and reference[month] != digest:
                raise ValueError(
                    "Compared models/views have different holdout populations"
                )
            reference[month] = digest
            total_rows += frame.height
            frames.append(frame)
            origins.append(values)
        pooled = pl.concat(frames)
        if pooled["target_record_id"].n_unique() != pooled.height:
            raise ValueError("Duplicate pooled holdout IDs")
        record = store.load(f"final/pooled/{name}_{view}")
        probabilities = pooled.select(PROB_COLUMNS).to_numpy()
        if not equivalent(
            record["metrics"], evaluate(pooled["y_true"].to_numpy(), probabilities)
        ):
            raise ValueError("Pooled metrics were not recomputed from all predictions")
        if not equivalent(
            record["airport_metrics"],
            airport_metrics(
                pooled["y_true"].to_numpy(),
                probabilities,
                np.concatenate(origins),
                cfg["evaluation_airports"],
            ),
        ):
            raise ValueError("Pooled airport metrics differ")
        verified.append(
            dict(
                model=name,
                feature_view=view,
                pooled_rows=pooled.height,
                model_fit_id=fit["model_fit_id"],
            )
        )
        print(
            f"Independently verified {name}/{view}: {pooled.height:,} frozen-holdout rows",
            flush=True,
        )
        del frames, origins, pooled, probabilities
        gc.collect()
    result = dict(
        passed=True,
        run_id=store.run_id,
        models_verified=len(verified),
        prediction_rows=total_rows,
        original_train_ids_hash=train_hash,
        month_test_id_hashes=reference,
        checks=[
            "selected_config_evidence",
            "original_manifest_IDs",
            "one_to_one_target_labels",
            "class_order_and_simplex",
            "saved_argmax",
            "frozen_fit_checksums",
            "monthly_metrics",
            "pooled_metrics",
            "origin_airport_metrics",
            "causal_history_cutoffs",
        ],
        models=verified,
        verification_seconds=time.monotonic() - started,
    )
    write_json(store.root / "verification.json", result)
    write_json(cfg.path("reports_dir") / "result_verification.json", result)
    print("Independent full-result verification passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase3.yaml"))
    verify(Phase3Config.load(parser.parse_args().config))
