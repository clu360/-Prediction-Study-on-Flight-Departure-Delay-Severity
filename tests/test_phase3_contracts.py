import numpy as np
import polars as pl
import pytest
from scipy.sparse import csr_matrix
from flight_delay.experiments.config import Phase3Config, DEV_FOLDS
from flight_delay.experiments.inputs import development_fold, checked_rows, memberships
from flight_delay.features.views import VIEWS, ALL
from flight_delay.models.registry import (
    create,
    ordered_probabilities,
    check_training_labels,
)


def test_exact_four_allowlisted_views_and_thresholds():
    from flight_delay.normalize import severity

    assert list(VIEWS) == ["G1", "G1_G2", "G1_G3", "G1_G2_G3"]
    assert [len(v) for v in VIEWS.values()] == [9, 14, 15, 20]
    assert len(ALL) == len(set(ALL)) == 20
    assert all(c.startswith(("g1_", "g2_", "g3_")) for c in ALL)
    assert pl.DataFrame({"delay": [-5.0, 14.9, 15.0, 59.9, 60.0]}).select(
        severity(pl.col("delay"))
    ).to_series().to_list() == [0, 0, 1, 1, 2]


def test_factories_unfitted_sparse_probability_order_and_missing_class():
    cfg = Phase3Config.load()
    X = csr_matrix(np.arange(18, dtype=np.float32).reshape(6, 3))
    y = np.array([0, 1, 2, 0, 1, 2])
    for name in cfg["models"]:
        params = dict(cfg.grids[name][0])
        if name == "random_forest":
            params.update(n_estimators=2, max_depth=2)
        if name == "xgboost":
            params.update(n_estimators=2, max_depth=2)
        model = create(name, params)
        assert not hasattr(model, "classes_")
        model.fit(X, y)
        p = ordered_probabilities(model, X)
        assert p.shape == (6, 3) and np.allclose(p.sum(axis=1), 1, atol=1e-6)
    baseline = create("baseline", {}).fit(X[:2], [0, 0])
    assert np.array_equal(
        ordered_probabilities(baseline, X), np.tile([1, 0, 0], (6, 1))
    )
    for name in ["logistic", "random_forest"]:
        model = create(name, cfg.grids[name][0]).fit(X[:4], [0, 2, 0, 2])
        assert np.all(ordered_probabilities(model, X)[:, 1] == 0)
    with pytest.raises(ValueError, match="all three"):
        check_training_labels("xgboost", [0, 1])
    with pytest.raises(ValueError):
        create("logistic", dict(cfg.grids["logistic"][0], early_stopping=True))


def test_development_reads_no_holdout_targets_and_aligns_all_views(
    phase3_fixture, monkeypatch
):
    cfg, _ = phase3_fixture
    original = pl.scan_parquet

    def guarded(paths, *args, **kwargs):
        assert "month=11" not in str(paths) and "month=12" not in str(paths)
        return original(paths, *args, **kwargs)

    monkeypatch.setattr(pl, "scan_parquet", guarded)
    reference = None
    for view in VIEWS:
        bundle = development_fold(cfg, "2025-10", view, cfg.root / "private")
        assert bundle.X_train.dtype == np.float32 and bundle.X_train.format == "csr"
        assert len(bundle.train_ids) == 36 and len(bundle.test_ids) == 12
        assert bundle.fitted_transformer.test_unknown_counts_["g1_month"] == 12
        if reference:
            assert np.array_equal(reference[0], bundle.train_ids) and np.array_equal(
                reference[1], bundle.test_ids
            )
        reference = (bundle.train_ids, bundle.test_ids)
    with pytest.raises(ValueError, match="holdout"):
        development_fold(cfg, "2025-12", "G1", cfg.root / "private")


def test_exact_rolling_folds_and_duplicate_missing_extra_ids_fail(phase3_fixture):
    cfg, _ = phase3_fixture
    for fold in DEV_FOLDS:
        train = memberships(cfg.path("processed_dir"), fold, "train")
        test = memberships(cfg.path("processed_dir"), fold, "test")
        assert set(train["target_record_id"]).isdisjoint(test["target_record_id"])
        month = int(fold[-2:])
        assert {int(i.split("-")[2]) for i in train["target_record_id"]} == set(
            range(month - 3, month)
        )
    ids = memberships(cfg.path("processed_dir"), "2025-11", "train")
    with pytest.raises(ValueError):
        checked_rows(cfg, pl.concat([ids, ids.head(1)]), [8, 9, 10], "G1")
    with pytest.raises(ValueError):
        checked_rows(cfg, ids.head(ids.height - 1), [8, 9, 10], "G1")


def test_foreign_holdout_partition_in_scope_is_rejected_before_scanning(phase3_fixture):
    cfg, _ = phase3_fixture
    root = cfg.root / "private"
    path = (
        root
        / "input_scopes/2025-10/features/targets/year=2025/month=11/foreign.parquet"
    )
    path.parent.mkdir(parents=True)
    pl.DataFrame({"target_record_id": ["foreign"], "target_class": [2]}).write_parquet(
        path
    )
    with pytest.raises(ValueError, match="Unexpected/stale"):
        development_fold(cfg, "2025-10", "G1", root)
