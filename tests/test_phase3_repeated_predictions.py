"""Strict repeat verification and explicitly failing cross-platform diagnostics."""

from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import os
import errno

import numpy as np
import polars as pl
import pytest

from flight_delay.experiments.results import ResultStore, PROB_COLUMNS


def script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paired(tmp_path, *, mismatch=None):
    cfg = SimpleNamespace(path=lambda key: tmp_path / key)
    stores = []
    for index in range(2):
        context = dict(
            class_order=[0, 1, 2],
            resolved_config=dict(seed=6740),
            versions=dict(fixture="test"),
            history_policy="causal",
            data_kind="synthetic_not_study",
            input_digest="same-fixture",
            independent_run=index,
        )
        store = ResultStore(cfg.path("output_dir"), context, True)
        p = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        if index and mismatch == "numeric":
            p[0] = [0.7, 0.2, 0.1]
        ids = ["fixture-a", "fixture-b", "fixture-c"]
        if index and mismatch == "IDs":
            ids[0] = "different-fixture"
        frame = pl.DataFrame(
            dict(
                target_record_id=ids,
                y_true=[0, 1, 2],
                **{name: p[:, i] for i, name in enumerate(PROB_COLUMNS)},
            )
        )
        path = store.root / "development/probe.parquet"
        path.parent.mkdir()
        frame.write_parquet(path)
        record = dict(
            model="xgboost",
            feature_view="G1",
            scored_month="2025-04",
            params={},
            train_ids_hash="training",
            test_ids_hash="testing",
            evaluation_stage="development",
            selected_config_id="probe",
            probability_hash=hashlib.sha256(p.tobytes()).hexdigest(),
        )
        store.save("development/units/probe", record, [path])
        stores.append(store)
    return cfg, stores


def test_numeric_diagnostic_reports_failure_without_weakening_strict_tolerance(
    tmp_path,
):
    compare = script("check_phase3_reproducibility").compare
    cfg, stores = paired(tmp_path, mismatch="numeric")
    with pytest.raises(ValueError, match="beyond tolerance"):
        compare(cfg, stores[0].run_id, stores[1].run_id)
    result = compare(cfg, stores[0].run_id, stores[1].run_id, strict=False)
    assert (
        result["complete"] and not result["passed"] and not result["strict_assertion"]
    )
    assert result["absolute_tolerance"] == 1e-7 and result["relative_tolerance"] == 0
    assert result["fits_exceeding_tolerance"] == 1
    assert result["comparisons"][0]["max_absolute_difference"] == pytest.approx(0.1)
    assert (cfg.path("reports_dir") / "cross_platform_reproducibility.json").exists()
    assert not (cfg.path("reports_dir") / "reproducibility_check.json").exists()


def test_diagnostic_mode_still_rejects_changed_prediction_IDs(tmp_path):
    cfg, stores = paired(tmp_path, mismatch="IDs")
    with pytest.raises(ValueError, match="different IDs or labels"):
        script("check_phase3_reproducibility").compare(
            cfg, stores[0].run_id, stores[1].run_id, strict=False
        )


def test_same_platform_repeat_fits_all_four_families_independently_and_resumes(
    phase3_fixture, monkeypatch
):
    from flight_delay.experiments import runner

    cfg, gate = phase3_fixture
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = script("repeat_phase3_seeded")
    module.batch.prepare(cfg, "B", gate=gate)
    module.batch.worker(cfg, "B", 0, "official-fixture", gate=gate)
    calls = []
    original = runner.fit_estimator
    original_link = os.link
    scope_root = cfg.root / "node_local_scope"

    def same_filesystem_only(source, target, **kwargs):
        if any(
            Path(source).is_relative_to(cfg.path(key))
            for key in ["features_dir", "processed_dir"]
        ) and not Path(target).is_relative_to(scope_root):
            raise OSError(
                errno.EXDEV, "Simulated node-local/shared filesystem boundary"
            )
        return original_link(source, target, **kwargs)

    monkeypatch.setattr(os, "link", same_filesystem_only)

    def fit(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "fit_estimator", fit)
    result = module.repeat(cfg, gate=gate, input_scope_root=scope_root)
    assert calls == ["baseline", "logistic", "random_forest", "xgboost"]
    assert result["passed"] and result["complete"] and result["compared_fits"] == 4
    assert result["independent_repeat"] and not result["used_for_selection"]
    assert result["additional_verification_fits"] == 4
    assert result["previous_run"] != result["corrected_run"]
    calls.clear()
    module.repeat(cfg, gate=gate, input_scope_root=scope_root)
    assert not calls, "Resumption refitted an already independently completed repeat"
    reference = module.batch.coordinator(cfg, gate)
    assert len(list((reference.root / "development/units").glob("*.json"))) == 7
    assert not (cfg.path("output_dir") / "PRIMARY_HOLDOUT_LOCK.json").exists()
