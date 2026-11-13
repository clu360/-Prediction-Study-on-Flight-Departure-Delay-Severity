from pathlib import Path
import json
import subprocess
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from flight_delay.experiments.config import (
    Phase3Config,
    DEV_FOLDS,
    analysis_fingerprint,
    code_fingerprints,
)
from flight_delay.experiments.inputs import public_error
from flight_delay.experiments.inputs import bounded_python
from flight_delay.experiments.results import ResultStore
from flight_delay.experiments.runner import Budget, fit_estimator
from flight_delay.experiments.tuning import aggregate, rank


def test_declared_grids_protocol_and_all_candidate_keys():
    cfg = Phase3Config.load()
    assert {m: len(g) for m, g in cfg.grids.items()} == {
        "baseline": 1,
        "logistic": 6,
        "random_forest": 8,
        "xgboost": 8,
    }
    assert cfg["development_folds"] == DEV_FOLDS
    assert cfg["final_train_months"] == [8, 9, 10] and cfg["final_eval_months"] == [
        11,
        12,
    ]
    assert cfg["seed"] == 6740 and cfg["outer_parallelism"] == 1
    assert cfg["threads"] <= 4


def test_screening_is_not_final_selection_and_tiebreaks_use_variance_then_runtime():
    def rows(p, scores):
        return [
            dict(
                model="logistic",
                feature_view="G1",
                params={"C": p},
                evaluation_stage="development",
                scored_month=f,
                metrics=dict(macro_f1=s, log_loss=1),
                resources=dict(fit_seconds=1),
                provenance_hash="x",
            )
            for f, s in zip(DEV_FOLDS, scores)
        ]

    a = rows(1, [0.8, 0.1, 0.1, 0.8, 0.1, 0.1, 0.8])
    b = rows(2, [0.7] * 7)
    screen = ["2025-04", "2025-07", "2025-10"]
    assert rank(
        [
            aggregate([r for r in a if r["scored_month"] in screen], screen),
            aggregate([r for r in b if r["scored_month"] in screen], screen),
        ]
    )[0]["params"] == {"C": 1}
    assert rank([aggregate(a, DEV_FOLDS), aggregate(b, DEV_FOLDS)])[0]["params"] == {
        "C": 2
    }
    x = aggregate(b, DEV_FOLDS)
    y = dict(
        x, params={"C": 3}, config_id="other", macro_f1_std=0.01, median_runtime=0.001
    )
    assert rank([y, x])[0]["params"] == {"C": 2}
    y.update(macro_f1_std=x["macro_f1_std"], median_runtime=0.1)
    assert rank([x, y])[0]["params"] == {"C": 3}


def test_nonconvergence_is_captured_not_hidden():
    cfg = Phase3Config.load()
    params = dict(cfg.grids["logistic"][0], max_iter=1, tol=1e-15)
    X = csr_matrix(np.random.default_rng(6740).normal(size=(90, 4)).astype(np.float32))
    _, _, messages = fit_estimator(
        cfg, "logistic", params, X, np.arange(90) % 3, Budget(cfg)
    )
    assert any("ConvergenceWarning" in m for m in messages)


def test_checkpoint_missing_artifact_and_public_error_redaction(tmp_path):
    store = ResultStore(tmp_path, {"context": 1})
    artifact = store.root / "private.bin"
    artifact.write_bytes(b"known")
    store.save("fit", {"state": "frozen"}, [artifact])
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="corrupted"):
        store.load("fit")
    cfg = Phase3Config.load()
    assert "Example" not in public_error(
        OSError("Cannot open C:\\Users\\Example\\fixture.parquet"), cfg
    )


def test_notebook_analysis_only_and_no_private_outputs():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / "notebooks/03_classical_models_analysis.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
    )
    assert (
        ".fit(" not in source
        and "fit_transform" not in source
        and "build_fold" not in source
    )
    assert "data/features" not in source and "model_outputs" not in source
    assert all(
        not c.get("outputs") and c.get("execution_count") is None
        for c in notebook["cells"]
        if c["cell_type"] == "code"
    )
    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "data/model_outputs/phase3/private.joblib",
            "data/model_outputs/phase3/predictions.parquet",
            ".env",
            ".venv",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0 and len(result.stdout.splitlines()) == 4


def test_upstream_timeout_is_bounded_and_does_not_leave_a_child_process():
    cfg = Phase3Config.load()
    with pytest.raises(subprocess.TimeoutExpired):
        bounded_python(cfg, ["-c", "import time; time.sleep(10)"], 0.1)


def test_notebook_outputs_and_markdown_do_not_invalidate_fitted_state(tmp_path):
    folder = tmp_path / "notebooks"
    folder.mkdir()
    path = folder / "03_classical_models_analysis.ipynb"
    notebook = {"cells": [{"cell_type": "code", "source": ["print(1)"], "outputs": []}]}
    path.write_text(json.dumps(notebook), encoding="utf-8")
    model_hashes = code_fingerprints(tmp_path)
    original_analysis = analysis_fingerprint(tmp_path)
    notebook["cells"][0]["outputs"] = [{"output_type": "stream", "text": "1"}]
    notebook["cells"].append({"cell_type": "markdown", "source": ["Verified findings"]})
    path.write_text(json.dumps(notebook), encoding="utf-8")
    assert code_fingerprints(tmp_path) == model_hashes
    assert analysis_fingerprint(tmp_path) == original_analysis
    notebook["cells"][0]["source"] = ["print(2)"]
    path.write_text(json.dumps(notebook), encoding="utf-8")
    assert analysis_fingerprint(tmp_path) != original_analysis


def test_fresh_seeded_classifiers_reproduce_probabilities():
    from flight_delay.models.registry import create, ordered_probabilities

    cfg = Phase3Config.load()
    X = csr_matrix(np.random.default_rng(6740).normal(size=(60, 6)).astype(np.float32))
    y = np.arange(60) % 3
    for name in cfg["models"]:
        params = dict(cfg.grids[name][0])
        if name in ["random_forest", "xgboost"]:
            params.update(n_estimators=3, max_depth=2)
        first = create(name, params).fit(X, y)
        second = create(name, params).fit(X, y)
        assert np.allclose(
            ordered_probabilities(first, X),
            ordered_probabilities(second, X),
            atol=1e-7,
            rtol=0,
        )


@pytest.mark.parametrize("name", ["baseline", "logistic"])
def test_reloaded_prediction_verifier_rejects_mismatched_outputs(phase3_fixture, name):
    import importlib.util
    import joblib
    import polars as pl
    from flight_delay.experiments.inputs import final_holdout_loader, holdout_month
    from flight_delay.experiments.results import prediction_frame
    from flight_delay.models.registry import create, ordered_probabilities

    spec = importlib.util.spec_from_file_location(
        "verify_phase3_results",
        Path(__file__).resolve().parents[1] / "scripts/verify_phase3_results.py",
    )
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    cfg, _ = phase3_fixture
    transformer, train, y, _ = final_holdout_loader(cfg, "G1")
    test, truth, ids, _ = holdout_month(cfg, 11, "G1", transformer)
    if name == "baseline":
        train = csr_matrix((len(y), 0), dtype=np.float32)
        test = csr_matrix((len(truth), 0), dtype=np.float32)
        transformer = None
    estimator = create(name, cfg.grids[name][0]).fit(train, y)
    state_path = cfg.root / "reloaded.joblib"
    joblib.dump(dict(estimator=estimator, transformer=transformer), state_path)
    state = joblib.load(state_path)
    frame = prediction_frame(
        "test",
        name,
        "G1",
        "frozen_holdout",
        "2025-11",
        ids,
        truth,
        ordered_probabilities(estimator, test),
        "fit",
    )
    cache = {}
    assert verifier.check_reloaded_predictions(cfg, state, frame, 11, cache) == len(
        truth
    )
    wrong = frame.with_columns(
        (pl.col("p_minimal") * 0.99).alias("p_minimal"),
        (pl.col("p_severe") + pl.col("p_minimal") * 0.01).alias("p_severe"),
    )
    with pytest.raises(ValueError, match="differ from saved"):
        verifier.check_reloaded_predictions(cfg, state, wrong, 11, cache)


@pytest.mark.parametrize("name", ["baseline", "logistic", "random_forest", "xgboost"])
def test_logical_model_hash_survives_serialization_and_prediction(name):
    import io
    import joblib
    from flight_delay.models.registry import create, ordered_probabilities
    from flight_delay.models.state import state_hash

    cfg = Phase3Config.load()
    params = dict(cfg.grids[name][0])
    if name in ["random_forest", "xgboost"]:
        params.update(n_estimators=3, max_depth=2)
    matrix = csr_matrix(
        np.random.default_rng(6740).normal(size=(60, 6)).astype(np.float32)
    )
    model = create(name, params).fit(matrix, np.arange(60) % 3)
    expected = ordered_probabilities(model, matrix)
    original = state_hash(model)
    for _ in range(3):
        buffer = io.BytesIO()
        joblib.dump(model, buffer, compress=3)
        buffer.seek(0)
        model = joblib.load(buffer)
        assert state_hash(model) == original
        assert np.allclose(
            ordered_probabilities(model, matrix), expected, atol=1e-8, rtol=1e-7
        )
        assert state_hash(model) == original
    # Changing a parameter or fitted numerical value must change the fingerprint.
    if name == "baseline":
        model.priors_[0] += 0.01
    elif name == "logistic":
        model.coef_[0, 0] += 0.01
    elif name == "random_forest":
        model.estimators_[0].tree_.threshold[0] += 0.01
    elif name == "xgboost":
        model.set_params(max_depth=params["max_depth"] + 1)
    assert state_hash(model) != original


def test_logical_array_hash_ignores_strides_and_tree_padding_but_checks_values():
    from flight_delay.models.state import state_hash

    view = np.arange(12, dtype=np.float32).reshape(3, 4)[:, 1:]
    assert state_hash(view) == state_hash(np.ascontiguousarray(view))
    dtype = np.dtype(
        {
            "names": ["flag", "weight"],
            "formats": ["u1", "f8"],
            "offsets": [0, 8],
            "itemsize": 24,
        }
    )
    first, second = np.zeros(3, dtype=dtype), np.ones(3, dtype=dtype)
    first.view(np.uint8)[:] = 0x55
    second.view(np.uint8)[:] = 0xAA
    for array in (first, second):
        array["flag"] = 1
        array["weight"] = [0.2, 0.3, 0.5]
    assert state_hash(first) == state_hash(second)
    second["weight"][0] += 0.01
    assert state_hash(first) != state_hash(second)
