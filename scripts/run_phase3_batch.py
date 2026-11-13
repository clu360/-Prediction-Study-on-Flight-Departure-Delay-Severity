"""Isolated Slurm development workers; selection/final scoring use the locked runner."""

from pathlib import Path
import argparse
import gc
import os
import re
import shutil
import time

from flight_delay.experiments.config import Phase3Config, digest, file_hash
from flight_delay.experiments.inputs import verified_gate, development_fold, origins_for
from flight_delay.experiments.results import (
    ResultStore,
    exclusive,
    read_json,
    write_json,
    atomic,
)
from flight_delay.experiments.runner import (
    Budget,
    context,
    baseline_development,
    evaluate_development,
    _unit_key,
    tune,
)
from flight_delay.experiments.tuning import aggregate, rank


def coordinator(cfg, gate):
    return ResultStore(cfg.path("output_dir"), context(cfg, gate), True)


def unlocked(cfg, store):
    if (cfg.path("output_dir") / "PRIMARY_HOLDOUT_LOCK.json").exists() or (
        store.root / "final/selection_lock.json"
    ).exists():
        raise ValueError("Holdout execution has begun; batch development is locked")


def manifest_path(cfg, stage):
    return cfg.path("output_dir") / "batch" / f"{stage}_tasks.json"


def prepare(cfg, stage, *, gate=None):
    gate = gate or verified_gate(cfg, "development")
    store = coordinator(cfg, gate)
    unlocked(cfg, store)
    learned = [name for name in cfg["models"] if name != "baseline"]
    with exclusive(store.root):
        if stage == "B":
            budget = Budget(cfg)
            for fold in cfg["development_folds"]:
                key = _unit_key("baseline", "independent", {}, fold)
                if store.load(key) is None:
                    record, path = baseline_development(cfg, store, fold, budget, True)
                    store.save(key, record, [path])
            shortlists = None
            folds = cfg["screening_folds"]
        elif stage == "C":
            shortlists = {}
            for view in cfg["views"]:
                shortlists[view] = {}
                for name in learned:
                    candidates = []
                    for params in cfg.grids[name]:
                        records = [
                            store.load(_unit_key(name, view, params, fold))
                            for fold in cfg["screening_folds"]
                        ]
                        if any(record is None for record in records):
                            raise ValueError(
                                "Full screening must finish before shortlisting"
                            )
                        candidates.append(aggregate(records, cfg["screening_folds"]))
                    shortlists[view][name] = [
                        item["params"]
                        for item in rank(candidates, cfg["tie_tolerance"])[
                            : cfg["shortlist_size"]
                        ]
                    ]
            folds = [
                fold
                for fold in cfg["development_folds"]
                if fold not in cfg["screening_folds"]
            ]
        else:
            raise ValueError("Expected development stage B or C")
        tasks = []
        for view in cfg["views"]:
            for fold in folds:
                requested = (
                    {name: cfg.grids[name] for name in learned}
                    if shortlists is None
                    else shortlists[view]
                )
                tasks.append(
                    dict(task_id=len(tasks), view=view, fold=fold, requested=requested)
                )
        result = dict(
            stage=stage,
            run_id=store.run_id,
            run_signature=store.signature,
            config_hash=cfg.config_hash,
            worker_code_sha256=file_hash(Path(__file__)),
            tasks=tasks,
            fit_count=sum(
                len(params) for task in tasks for params in task["requested"].values()
            ),
        )
        result["manifest_hash"] = digest(result)
        path = manifest_path(cfg, stage)
        if path.exists() and read_json(path) != result:
            raise ValueError(
                "Refusing to replace a different frozen batch task manifest"
            )
        write_json(path, result)
    return result


def load_manifest(cfg, stage, store):
    value = read_json(manifest_path(cfg, stage))
    if (
        value["manifest_hash"]
        != digest({k: v for k, v in value.items() if k != "manifest_hash"})
        or value["run_signature"] != store.signature
        or value["config_hash"] != cfg.config_hash
        or value["worker_code_sha256"] != file_hash(Path(__file__))
    ):
        raise ValueError("Stale or corrupted batch manifest/source")
    return value


def task_parent(cfg, stage, task_id):
    base = cfg.path("output_dir").resolve()
    if os.name == "nt" and not str(base).startswith("\\\\?\\"):
        base = Path("\\\\?\\" + str(base))
    return base / "batch_workers" / stage / f"task-{task_id:02}"


def attempt_stores(cfg, stage, task_id, reference):
    return [
        ResultStore(path.parent, reference.context, True)
        for path in sorted(
            task_parent(cfg, stage, task_id).glob(f"attempt-*/{reference.run_id}")
        )
    ]


def import_record(source, destination, key, expected_manifest):
    record = source.load(key)
    if record is None:
        return False
    if record.get("batch_manifest_hash") != expected_manifest:
        raise ValueError("Worker result belongs to a different batch manifest")
    existing = destination.load(key)
    if existing is not None:
        if any(
            existing[field] != record[field]
            for field in [
                "params",
                "train_ids_hash",
                "test_ids_hash",
                "probability_hash",
            ]
        ):
            raise ValueError("Conflicting duplicate worker results")
        return True
    artifacts = []
    for relative in record["artifacts"]:
        original, target = source.root / relative, destination.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            try:
                os.link(original, target)
            except OSError:
                atomic(target, lambda temporary: shutil.copyfile(original, temporary))
        if file_hash(original) != file_hash(target):
            raise ValueError(
                "Worker prediction copy differs from its verified checksum"
            )
        artifacts.append(target)
    record = {
        k: v
        for k, v in record.items()
        if k not in {"artifacts", "run_signature", "record_hash"}
    }
    destination.save(key, record, artifacts)
    return True


def worker(cfg, stage, task_id, attempt, *, gate=None):
    gate = gate or verified_gate(cfg, "development")
    reference = coordinator(cfg, gate)
    unlocked(cfg, reference)
    manifest = load_manifest(cfg, stage, reference)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", attempt):
        raise ValueError("Expected a simple safe attempt label")
    task = manifest["tasks"][task_id]
    view, fold = task["view"], task["fold"]
    if (
        task["task_id"] != task_id
        or view not in cfg["views"]
        or fold not in cfg["development_folds"]
    ):
        raise ValueError("Batch task has invalid or held-out membership")
    requested = [
        (name, params)
        for name, choices in task["requested"].items()
        for params in choices
    ]
    if any(
        name == "baseline" or params not in cfg.grids[name]
        for name, params in requested
    ):
        raise ValueError("Batch task contains an undeclared candidate")
    store = ResultStore(
        task_parent(cfg, stage, task_id) / f"attempt-{attempt}", reference.context, True
    )
    with exclusive(store.root):
        for previous in attempt_stores(cfg, stage, task_id, reference):
            if previous.root == store.root:
                continue
            for name, params in requested:
                import_record(
                    previous,
                    store,
                    _unit_key(name, view, params, fold),
                    manifest["manifest_hash"],
                )
        missing = [
            (name, params)
            for name, params in requested
            if store.load(_unit_key(name, view, params, fold)) is None
        ]
        if not missing:
            return store
        budget = Budget(cfg)
        start = time.monotonic()
        with budget.measure():
            bundle = development_fold(cfg, fold, view, store.root)
        preprocessing = time.monotonic() - start
        origins = origins_for(cfg, bundle.test_ids, int(fold[-2:]))
        for name, params in missing:
            unlocked(cfg, reference)
            budget.check()
            print(
                f"Batch {stage}/{task_id}: {name}/{view}/{fold}, {len(bundle.y_train):,} training rows",
                flush=True,
            )
            record, path = evaluate_development(
                cfg,
                store,
                bundle,
                name,
                view,
                params,
                fold,
                budget,
                preprocessing,
                origins,
                save_predictions=True,
                candidate=True,
            )
            record["batch_manifest_hash"] = manifest["manifest_hash"]
            record["resources"]["batch_stage"] = stage
            record["resources"]["batch_task_id"] = task_id
            store.save(_unit_key(name, view, params, fold), record, [path])
            gc.collect()
    return store


def merge(cfg, stage, *, gate=None):
    gate = gate or verified_gate(cfg, "development")
    store = coordinator(cfg, gate)
    unlocked(cfg, store)
    manifest = load_manifest(cfg, stage, store)
    with exclusive(store.root):
        for task in manifest["tasks"]:
            attempts = attempt_stores(cfg, stage, task["task_id"], store)
            for name, choices in task["requested"].items():
                for params in choices:
                    key = _unit_key(name, task["view"], params, task["fold"])
                    found = False
                    for source in attempts:
                        found = (
                            import_record(source, store, key, manifest["manifest_hash"])
                            or found
                        )
                    if not found:
                        raise ValueError(
                            f"Missing completed worker result: {stage}/{task['task_id']}/{name}"
                        )
        write_json(
            store.root / f"batch/{stage}_merged.json",
            dict(
                complete=True,
                manifest_hash=manifest["manifest_hash"],
                fits=manifest["fit_count"],
            ),
        )
    return store


def select(cfg, *, gate=None):
    gate = gate or verified_gate(cfg, "development")
    merge(cfg, "C", gate=gate)
    # All candidates already exist: the original tuner alone produces selection.
    return tune(cfg, resume=True, gate=gate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "worker", "merge", "select"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["B", "C"], default="B")
    parser.add_argument("--task-id", type=int)
    parser.add_argument(
        "--attempt",
        default=os.environ.get("SLURM_JOB_ID", "local")
        + "-"
        + os.environ.get("SLURM_RESTART_COUNT", "0"),
    )
    args = parser.parse_args()
    cfg = Phase3Config.load(args.config)
    if args.command == "worker":
        if args.task_id is None:
            parser.error("worker requires --task-id")
        worker(cfg, args.stage, args.task_id, args.attempt)
    elif args.command == "prepare":
        result = prepare(cfg, args.stage)
        print(f"Prepared {len(result['tasks'])} tasks / {result['fit_count']} fits")
    elif args.command == "merge":
        merge(cfg, args.stage)
    else:
        select(cfg)
