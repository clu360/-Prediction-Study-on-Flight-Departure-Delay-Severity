from dataclasses import replace
import joblib
import numpy as np
import polars as pl
import pytest
from flight_delay.experiments import runner, inputs
from flight_delay.experiments.results import read_json, PROB_COLUMNS
from flight_delay.features.transforms import FoldTransformer


def test_final_loader_aug_oct_only_frozen_unknowns_and_no_december_build_fold(
    phase3_fixture, monkeypatch
):
    cfg, _ = phase3_fixture
    calls = []
    original = FoldTransformer.fit

    def fit(self, frame):
        calls.append(frame["g1_month"].unique().sort().to_list())
        return original(self, frame)

    monkeypatch.setattr(FoldTransformer, "fit", fit)
    monkeypatch.setattr(
        inputs,
        "build_fold",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("Final must not call rolling fold loader")
        ),
    )
    transformer, X, y, ids = inputs.final_holdout_loader(cfg, "G1_G2_G3")
    assert calls == [[8, 9, 10]]
    before = joblib.hash(transformer)
    for month in [11, 12]:
        Xtest, yt, ids, unknown = inputs.holdout_month(
            cfg, month, "G1_G2_G3", transformer
        )
        assert unknown["g1_month"] == 12 and Xtest.shape[1] == X.shape[1]
    assert calls == [[8, 9, 10]] and joblib.hash(transformer) == before


def test_end_to_end_development_sentinels_frozen_final_resume_pooling_and_redaction(
    phase3_fixture, monkeypatch
):
    cfg, gate = phase3_fixture
    original = inputs.table_scan

    def dev_guard(root, table, months):
        if table in ["targets", "full_features", "metadata"]:
            assert max(months) <= 10, "Tuner opened holdout-only rows"
        return original(root, table, months)

    monkeypatch.setattr(inputs, "table_scan", dev_guard)
    store = runner.tune(cfg, gate=gate)
    monkeypatch.setattr(inputs, "table_scan", original)
    selected = read_json(store.root / "development/selected_configs.json")
    assert selected["complete"] and len(selected["selected"]) == 13
    assert len(list((store.root / "development/units").glob("*.json"))) == 91
    assert (
        len(list((store.root / "development/selected_predictions").glob("*.parquet")))
        == 91
    )
    # Change every final-month label AFTER tuning. Selection bytes remain frozen.
    selection_hash = selected["selection_hash"]
    for month in [11, 12]:
        path = next(
            (cfg.path("features_dir") / "targets/year=2025" / f"month={month}").glob(
                "*.parquet"
            )
        )
        pl.read_parquet(path).with_columns(
            pl.lit(2).alias("target_class")
        ).write_parquet(path)
    fits = []
    orig_fit = runner.fit_estimator

    def fit(*args, **kwargs):
        # Label 2 in holdout must not collapse the training population.
        assert set(np.asarray(args[4]).tolist()) == {0, 1, 2}
        model, seconds, messages = orig_fit(*args, **kwargs)
        fits.append(model)
        return model, seconds, messages

    monkeypatch.setattr(runner, "fit_estimator", fit)
    runner.frozen_final(
        cfg, store.root / "development/selected_configs.json", gate=gate
    )
    assert len(fits) == 13 and len({id(m) for m in fits}) == 13
    assert (
        read_json(store.root / "development/selected_configs.json")["selection_hash"]
        == selection_hash
    )
    for item in selected["selected"].values():
        name, view = item["model"], item["feature_view"]
        a = store.load(f"final/months/{name}_{view}_2025-11")
        b = store.load(f"final/months/{name}_{view}_2025-12")
        assert (
            a["model_fit_id"] == b["model_fit_id"]
            and a["model_hash"] == b["model_hash"]
            and a["transformer_hash"] == b["transformer_hash"]
        )
        assert a["train_ids_hash"] == b["train_ids_hash"]
        pooled = store.load(f"final/pooled/{name}_{view}")
        assert pooled["metrics"]["n"] == 24
        assert sum(pooled["metrics"]["support"]) == 24
    saved = pl.read_parquet(store.root / "final/predictions.parquet")
    assert saved.height == 13 * 24
    assert (
        saved.select(
            "model", "feature_view", "scored_month", "target_record_id"
        ).n_unique()
        == saved.height
    )
    assert np.array_equal(
        saved["y_pred"].to_numpy(), saved.select(PROB_COLUMNS).to_numpy().argmax(axis=1)
    )
    runner.frozen_final(
        cfg, store.root / "development/selected_configs.json", True, gate=gate
    )
    assert len(fits) == 13, "Resume refit a frozen model"
    assert runner.report(cfg, store)["completion_status"] == "complete"
    with pytest.raises(ValueError, match="locked"):
        runner.tune(cfg, True, gate=gate)
    public = "".join(
        p.read_text(encoding="utf-8") for p in cfg.path("reports_dir").glob("*.json")
    )
    assert "SYNTHETIC-2025-" not in public
    assert (
        "p_minimal"
        not in pl.read_csv(cfg.path("reports_dir") / "metrics_summary.csv").columns
    )


def test_strict_history_cutoff_is_checked_in_phase3(phase3_fixture):
    cfg, _ = phase3_fixture
    path = next(
        (cfg.path("features_dir") / "metadata/year=2025/month=04").glob("*.parquet")
    )
    meta = pl.read_parquet(path).with_columns(
        pl.col("prediction_time_utc").alias("history_max_event_utc")
    )
    meta.write_parquet(path)
    with pytest.raises(ValueError, match="causal"):
        inputs.origins_for(cfg, meta["target_record_id"].to_numpy(), 4)


def test_each_candidate_has_seven_fresh_fits_and_distinct_transformer_objects(
    phase3_fixture, monkeypatch
):
    cfg, gate = phase3_fixture
    params = cfg.grids["logistic"][0]
    cfg = replace(
        cfg,
        values=dict(cfg.values, models=["logistic"], views=["G1"]),
        grids=dict(cfg.grids, logistic=[dict(params, C=0.1), dict(params, C=1.0)]),
    )
    estimators = []
    transformers = []
    original_create = runner.create
    original_copy = runner.deepcopy

    def create(*args, **kwargs):
        model = original_create(*args, **kwargs)
        estimators.append(model)
        return model

    def copy(value):
        result = original_copy(value)
        if isinstance(result, FoldTransformer):
            transformers.append(result)
        return result

    monkeypatch.setattr(runner, "create", create)
    monkeypatch.setattr(runner, "deepcopy", copy)
    store = runner.tune(cfg, gate=gate)
    assert (
        len(estimators) == len(transformers) == 14
    )  # 2 x 7; cached probabilities avoid redundant refits
    assert (
        len({id(x) for x in estimators}) == 14
        and len({id(x) for x in transformers}) == 14
    )
    units = [
        store.load(str(p.relative_to(store.root).with_suffix("")).replace("\\", "/"))
        for p in (store.root / "development/units").glob("*.json")
    ]
    assert len(units) == 14
    for C in [0.1, 1.0]:
        assert {r["scored_month"] for r in units if r["params"]["C"] == C} == set(
            cfg["development_folds"]
        )
