"""Independent same-platform full-population repeats; never used for selection."""

from dataclasses import replace
from pathlib import Path
import argparse
import platform
import time

import run_phase3_batch as batch
from check_phase3_reproducibility import compare
from validate_phase3_pace import staged_repository
from flight_delay.experiments.config import Phase3Config, file_hash
from flight_delay.experiments.inputs import verified_gate, development_fold, origins_for
from flight_delay.experiments.results import ResultStore, exclusive, write_json
from flight_delay.experiments.runner import (
    Budget,
    baseline_development,
    evaluate_development,
    _unit_key,
)
from flight_delay.experiments.tuning import config_id


def repeat(cfg, *, xgboost_config_id=None, gate=None):
    gate = gate or verified_gate(cfg, "development")
    reference = batch.coordinator(cfg, gate)
    batch.unlocked(cfg, reference)
    plan, attempts = None, []
    if batch.manifest_path(cfg, "B").exists():
        plan = batch.load_manifest(cfg, "B", reference)
        task_id = next(
            task["task_id"]
            for task in plan["tasks"]
            if task["view"] == "G1" and task["fold"] == "2025-04"
        )
        attempts = batch.attempt_stores(cfg, "B", task_id, reference)
    parent = cfg.path("output_dir") / "verification_repeats"
    snapshot = ResultStore(parent, reference.context, True)
    requested = {model: cfg.grids[model][0] for model in cfg["models"]}
    if xgboost_config_id:
        choices = [
            params
            for params in cfg.grids["xgboost"]
            if config_id(params) == xgboost_config_id
        ]
        if len(choices) != 1:
            raise ValueError("Repeat XGBoost ID is not in the original declared grid")
        requested["xgboost"] = choices[0]
    replica = ResultStore(
        parent,
        dict(
            reference.context,
            verification_repeat="Independent full-population April/G1 fits, one per family; no selection",
            verification_parameters=requested,
            repeat_code_sha256=file_hash(Path(__file__)),
        ),
        True,
    )
    # First declared parameters are fixed independently of scores. The optional
    # XGBoost ID repeats the exact already-diagnosed cross-platform case.
    with exclusive(replica.root), exclusive(snapshot.root):
        for model, params in requested.items():
            view = "independent" if model == "baseline" else "G1"
            key = _unit_key(model, view, params, "2025-04")
            sources = [reference] if model == "baseline" else [reference, *attempts]
            source = next(
                (store for store in sources if store.load(key) is not None), None
            )
            if source is None:
                raise ValueError(
                    "Required official full-population reference fit is incomplete"
                )
            batch.import_record(
                source,
                snapshot,
                key,
                plan["manifest_hash"] if model != "baseline" and plan else None,
            )
        budget = Budget(cfg)
        baseline_key = _unit_key(
            "baseline", "independent", requested["baseline"], "2025-04"
        )
        if replica.load(baseline_key) is None:
            record, path = baseline_development(cfg, replica, "2025-04", budget, True)
            replica.save(baseline_key, record, [path])
        missing = [
            (model, params)
            for model, params in requested.items()
            if model != "baseline"
            and replica.load(_unit_key(model, "G1", params, "2025-04")) is None
        ]
        if missing:
            started = time.monotonic()
            with budget.measure():
                bundle = development_fold(cfg, "2025-04", "G1", replica.root)
            preprocessing = time.monotonic() - started
            origins = origins_for(cfg, bundle.test_ids, 4)
            for model, params in missing:
                batch.unlocked(cfg, reference)
                print(
                    f"Independent seeded repeat: {model}/G1/2025-04 on {len(bundle.y_train):,} training rows",
                    flush=True,
                )
                record, path = evaluate_development(
                    cfg,
                    replica,
                    bundle,
                    model,
                    "G1",
                    params,
                    "2025-04",
                    budget,
                    preprocessing,
                    origins,
                    save_predictions=True,
                    candidate=True,
                )
                replica.save(_unit_key(model, "G1", params, "2025-04"), record, [path])
    private_cfg = replace(
        cfg,
        values=dict(
            cfg.values,
            output_dir=str(Path(cfg["output_dir"]) / "verification_repeats"),
            reports_dir=str(Path(cfg["output_dir"]) / "verification_repeats/reports"),
        ),
    )
    result = compare(private_cfg, replica.run_id, snapshot.run_id)
    if not result["complete"] or result["compared_fits"] != 4:
        raise ValueError("All four independent family repeats must pass")
    result.update(
        scope="Same-platform independent repeats: one full-population April/G1 fit per family, all four compared; not a rerun of every candidate/view/fold",
        independent_repeat=True,
        same_platform=True,
        platform=platform.system(),
        additional_verification_fits=4,
        official_performance_fit_count_unchanged=True,
    )
    write_json(reference.root / "reproducibility_check.json", result)
    return result


def run_staged(cfg, xgboost_config_id):
    with staged_repository(cfg, cfg.root.parent) as local_cfg:
        shared = cfg.path("output_dir")
        target = local_cfg.root / cfg["output_dir"]
        if target.exists() or target.is_symlink():
            raise ValueError("Staging archive contains fitted output directories")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(shared, target_is_directory=True)
        assert local_cfg.path("output_dir") == shared
        result = repeat(local_cfg, xgboost_config_id=xgboost_config_id)
    write_json(cfg.path("reports_dir") / "reproducibility_check.json", result)
    print(
        {
            key: result[key]
            for key in [
                "passed",
                "complete",
                "compared_fits",
                "compared_prediction_rows",
            ]
        },
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase3_pace.yaml"))
    parser.add_argument("--xgboost-config-id")
    parser.add_argument(
        "--stage-inputs",
        action="store_true",
        help="Use verified node-local source/input copies on PACE",
    )
    args = parser.parse_args()
    cfg = Phase3Config.load(args.config)
    if args.stage_inputs:
        run_staged(cfg, args.xgboost_config_id)
    else:
        result = repeat(cfg, xgboost_config_id=args.xgboost_config_id)
        write_json(cfg.path("reports_dir") / "reproducibility_check.json", result)
        print(
            {
                key: result[key]
                for key in [
                    "passed",
                    "complete",
                    "compared_fits",
                    "compared_prediction_rows",
                ]
            },
            flush=True,
        )
