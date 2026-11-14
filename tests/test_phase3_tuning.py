import pytest
from flight_delay.experiments.config import DEV_FOLDS
from flight_delay.experiments.tuning import aggregate, rank
from flight_delay.experiments.results import ResultStore, write_json, read_json


def records(params, score=0.4):
    return [
        dict(
            model="logistic",
            feature_view="G1",
            params=params,
            evaluation_stage="development",
            scored_month=f,
            metrics=dict(macro_f1=score, log_loss=1.0),
            resources=dict(fit_seconds=1.0),
            provenance_hash="fixed",
        )
        for f in DEV_FOLDS
    ]


def test_selection_equal_month_mean_variance_and_deterministic_ties():
    a, b = (
        aggregate(records({"C": 1}), DEV_FOLDS),
        aggregate(records({"C": 2}), DEV_FOLDS),
    )
    assert a["mean_macro_f1"] == pytest.approx(0.4) and a[
        "macro_f1_std"
    ] == pytest.approx(0)
    assert rank([b, a])[0]["params"] == {"C": 1}
    b["mean_macro_f1"] += 5e-7
    b["mean_log_loss"] = 0.5
    assert rank([a, b])[0]["params"] == {"C": 2}
    b["mean_macro_f1"] -= 2e-6
    assert rank([b, a])[0]["params"] == {"C": 1}


def test_selection_refuses_holdout_rows_incomplete_folds_and_changed_params():
    rows = records({"C": 1})
    with pytest.raises(ValueError):
        aggregate(rows[:-1], DEV_FOLDS)
    rows[-1]["scored_month"] = "2025-11"
    with pytest.raises(ValueError):
        aggregate(rows, DEV_FOLDS)
    rows = records({"C": 1})
    rows[-1]["params"] = {"C": 2}
    with pytest.raises(ValueError):
        aggregate(rows, DEV_FOLDS)


def test_checkpoint_resume_corruption_and_provenance(tmp_path):
    store = ResultStore(tmp_path, {"config": "x"})
    store.save("candidate", {"score": 0.4})
    assert (
        ResultStore(tmp_path, {"config": "x"}, True).load("candidate")["score"] == 0.4
    )
    path = store.root / "candidate.json"
    value = read_json(path)
    value["score"] = 0.8
    write_json(path, value)
    with pytest.raises(ValueError, match="Corrupted"):
        store.load("candidate")
    assert ResultStore(tmp_path, {"config": "y"}).run_id != store.run_id
