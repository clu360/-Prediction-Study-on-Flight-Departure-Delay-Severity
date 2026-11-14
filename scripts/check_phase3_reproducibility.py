"""Compare matching private development predictions; never fit or select models."""

from pathlib import Path
import argparse
import json
import re
import numpy as np
import polars as pl
from flight_delay.experiments.config import Phase3Config
from flight_delay.experiments.evaluation import check_probabilities
from flight_delay.experiments.results import (
    ResultStore,
    read_json,
    write_json,
    PROB_COLUMNS,
)


def compare(cfg, previous_run, current_run, *, strict=True):
    base = cfg.path("output_dir")
    stores = []
    for name in [previous_run, current_run]:
        if not re.fullmatch(r"phase3-[0-9a-f]{20}", name):
            raise ValueError("Expected an explicit valid private run ID")
        store = ResultStore(base, read_json(base / name / "context.json"), True)
        if store.run_id != name:
            raise ValueError("Run ID does not match its signed context")
        stores.append(store)
    old, new = stores
    for store in stores:
        if (
            store.context["class_order"] != [0, 1, 2]
            or store.context["resolved_config"]["seed"] != 6740
        ):
            raise ValueError("Expected the locked class order and seed")
    if any(
        old.context[key] != new.context[key]
        for key in ["versions", "history_policy", "data_kind"]
    ):
        raise ValueError(
            "Paired runs have different dependency versions or history policies"
        )
    if (
        old.run_id == new.run_id
        or old.context["input_digest"] != new.context["input_digest"]
    ):
        raise ValueError("Require distinct runs with the same validated input digest")
    records = []
    previous_paths = sorted((old.root / "development/units").glob("*.json"))
    for path in previous_paths:
        key = "development/units/" + path.stem
        previous, current = old.load(key), new.load(key)
        if current is None:
            continue
        for field in [
            "model",
            "feature_view",
            "scored_month",
            "params",
            "train_ids_hash",
            "test_ids_hash",
        ]:
            if previous[field] != current[field]:
                raise ValueError(f"Paired development fits differ in {field}")
        if (
            current["evaluation_stage"] != "development"
            or previous["evaluation_stage"] != "development"
        ):
            raise ValueError("Only development predictions may enter this comparison")
        arrays, frames = [], []
        for store, record in [(old, previous), (new, current)]:
            name = next(x for x in record["artifacts"] if x.endswith(".parquet"))
            frame = pl.read_parquet(
                store.root / name, columns=["target_record_id", "y_true", *PROB_COLUMNS]
            )
            frames.append(frame)
            arrays.append(frame.select(PROB_COLUMNS).to_numpy())
        if not frames[0]["target_record_id"].equals(
            frames[1]["target_record_id"]
        ) or not frames[0]["y_true"].equals(frames[1]["y_true"]):
            raise ValueError("Paired prediction rows have different IDs or labels")
        before, after = arrays
        check_probabilities(before, len(before))
        check_probabilities(after, len(after))
        if before.shape != after.shape:
            raise ValueError("Paired prediction arrays have different shapes")
        within = bool(np.allclose(before, after, atol=1e-7, rtol=0))
        if strict and not within:
            raise ValueError(
                "Repeated seeded development predictions differ beyond tolerance"
            )
        records.append(
            dict(
                model=current["model"],
                feature_view=current["feature_view"],
                scored_month=current["scored_month"],
                candidate_id=current["selected_config_id"],
                prediction_rows=len(after),
                max_absolute_difference=float(np.abs(before - after).max()),
                mean_absolute_difference=float(np.abs(before - after).mean()),
                argmax_changed_rows=int((before.argmax(1) != after.argmax(1)).sum()),
                within_declared_tolerance=within,
                ordered_probability_digest_matches=previous["probability_hash"]
                == current["probability_hash"],
            )
        )
    if not records:
        raise ValueError("No completed paired development predictions found")
    result = dict(
        passed=all(row["within_declared_tolerance"] for row in records),
        complete=len(records) == len(previous_paths),
        verification_only=True,
        used_for_selection=False,
        data_kind=new.context["data_kind"],
        previous_run=old.run_id,
        corrected_run=new.run_id,
        input_digest=new.context["input_digest"],
        class_order=[0, 1, 2],
        seed=6740,
        absolute_tolerance=1e-7,
        relative_tolerance=0,
        compared_fits=len(records),
        available_previous_fits=len(previous_paths),
        compared_prediction_rows=sum(x["prediction_rows"] for x in records),
        comparisons=records,
        strict_assertion=strict,
        fits_exceeding_tolerance=sum(
            not row["within_declared_tolerance"] for row in records
        ),
        scope="Complete-population development reproducibility only; interrupted fits never enter current selection or performance tables",
    )
    name = (
        "reproducibility_check.json"
        if strict
        else "cross_platform_reproducibility.json"
    )
    if not strict:
        result["scope"] = (
            "Cross-platform development diagnostic; numeric differences are reported as failures at the unchanged tolerance, not treated as a passing seeded-repeat gate. Metadata, IDs and labels must still match."
        )
    write_json(new.root / name, result)
    write_json(cfg.path("reports_dir") / name, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase3.yaml"))
    parser.add_argument("--previous-run", required=True)
    parser.add_argument("--current-run", required=True)
    parser.add_argument(
        "--diagnose-differences",
        action="store_true",
        help="Record all numeric differences without asserting a pass; structural mismatches still fail",
    )
    args = parser.parse_args()
    result = compare(
        Phase3Config.load(args.config),
        args.previous_run,
        args.current_run,
        strict=not args.diagnose_differences,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in [
                    "passed",
                    "complete",
                    "compared_fits",
                    "compared_prediction_rows",
                ]
            }
        )
    )
