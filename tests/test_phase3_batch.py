"""Batch scheduling preserves the original development and selection protocol."""

from dataclasses import replace
from pathlib import Path
import importlib.util
import pytest
from flight_delay.experiments import inputs, runner
from flight_delay.experiments.config import digest
from flight_delay.experiments.results import read_json, write_json

spec = importlib.util.spec_from_file_location(
    "phase3_batch", Path(__file__).resolve().parents[1] / "scripts/run_phase3_batch.py"
)
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


def test_batch_all_development_shards_merge_then_original_selector_without_refits(
    phase3_fixture, monkeypatch
):
    cfg, gate = phase3_fixture
    params = cfg.grids["logistic"][0]
    cfg = replace(
        cfg, grids=dict(cfg.grids, logistic=[dict(params, C=0.1), dict(params, C=1.0)])
    )
    original_scan = inputs.table_scan

    def guard(root, table, months):
        assert max(months) <= 10, "Batch development opened a holdout month"
        return original_scan(root, table, months)

    monkeypatch.setattr(inputs, "table_scan", guard)
    screening = batch.prepare(cfg, "B", gate=gate)
    assert len(screening["tasks"]) == 12 and screening["fit_count"] == 48
    roots = set()
    for task in screening["tasks"]:
        store = batch.worker(cfg, "B", task["task_id"], "test", gate=gate)
        roots.add(store.root)
    assert len(roots) == 12
    batch.merge(cfg, "B", gate=gate)
    finalists = batch.prepare(cfg, "C", gate=gate)
    assert len(finalists["tasks"]) == 16 and finalists["fit_count"] == 64
    assert {task["fold"] for task in finalists["tasks"]} == {
        "2025-05",
        "2025-06",
        "2025-08",
        "2025-09",
    }
    for task in finalists["tasks"]:
        batch.worker(cfg, "C", task["task_id"], "test", gate=gate)
    monkeypatch.setattr(
        runner,
        "fit_estimator",
        lambda *a, **k: pytest.fail("Selector refitted a completed candidate"),
    )
    store = batch.select(cfg, gate=gate)
    selected = read_json(store.root / "development/selected_configs.json")
    assert selected["complete"] and len(selected["selected"]) == 13
    assert len(list((store.root / "development/units").glob("*.json"))) == 119
    assert len(list((store.root / "development/selected").glob("*.json"))) == 91
    assert (
        len(list((store.root / "development/selected_predictions").glob("*.parquet")))
        == 91
    )
    assert not (cfg.path("output_dir") / "PRIMARY_HOLDOUT_LOCK.json").exists()


def test_batch_interrupted_attempt_reuses_verified_completed_fit(
    phase3_fixture, monkeypatch
):
    cfg, gate = phase3_fixture
    batch.prepare(cfg, "B", gate=gate)
    original = runner.fit_estimator
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(args[1])
        if len(calls) == 2:
            raise RuntimeError("simulated scheduler preemption")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "fit_estimator", interrupted)
    with pytest.raises(RuntimeError, match="preemption"):
        batch.worker(cfg, "B", 0, "old-attempt", gate=gate)
    calls.clear()

    def count(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "fit_estimator", count)
    result = batch.worker(cfg, "B", 0, "new-attempt", gate=gate)
    assert calls == ["random_forest", "xgboost"]
    assert len(list((result.root / "development/units").glob("*.json"))) == 3


def test_batch_refuses_holdout_tasks_and_existing_primary_lock(phase3_fixture):
    cfg, gate = phase3_fixture
    manifest = batch.prepare(cfg, "B", gate=gate)
    manifest["tasks"][0]["fold"] = "2025-11"
    manifest["manifest_hash"] = digest(
        {k: v for k, v in manifest.items() if k != "manifest_hash"}
    )
    write_json(batch.manifest_path(cfg, "B"), manifest)
    with pytest.raises(ValueError, match="held-out membership"):
        batch.worker(cfg, "B", 0, "bad", gate=gate)
    write_json(cfg.path("output_dir") / "PRIMARY_HOLDOUT_LOCK.json", {"locked": True})
    with pytest.raises(ValueError, match="locked"):
        batch.prepare(cfg, "B", gate=gate)


def test_slurm_final_scoring_requires_successful_selection_and_bounded_arrays(
    tmp_path, monkeypatch
):
    spec = importlib.util.spec_from_file_location(
        "phase3_slurm",
        Path(__file__).resolve().parents[1] / "scripts/submit_phase3_slurm.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def submit(command, **kwargs):
        calls.append(command)
        return str(10000 + len(calls)) + "\n"

    monkeypatch.setattr(module.subprocess, "check_output", submit)
    result = module.submit(
        tmp_path,
        Path("configs/phase3_pace.yaml"),
        "test-allocation",
        "cpu-gnr",
        "inferno",
        8,
    )
    stages = {stage["stage"]: stage for stage in result["stages"]}
    assert stages["frozen_final_verify"]["parent"] == "select"
    assert f"--dependency=afterok:{stages['select']['job_id']}" in calls[-1]
    assert f"--dependency=afterok:{stages['screen_B']['job_id']}" in calls[3]
    assert f"--dependency=afterok:{stages['finalists_C']['job_id']}" in calls[5]
    assert "--array=0-11%8" in calls[2] and "--array=0-15%8" in calls[4]
    assert all("--cpus-per-task=4" in command for command in calls)
