import numpy as np
import polars as pl
import pytest

pytest.importorskip("torch")

from flight_delay.models import mlp, source_mlp
from flight_delay.models.state import state_hash


def test_source_adapter_units_missing_and_unseen_categories():
    train = pl.DataFrame({"g1_crs_elapsed_min": [60, None], "g1_origin": ["ATL", None]})
    adapter = source_mlp.SourceAdapter.fit(train.columns, train)
    test = pl.DataFrame({"g1_crs_elapsed_min": [120, 0, None], "g1_origin": ["ATL", "LAX", None]})
    before = state_hash(adapter)
    np.testing.assert_equal(adapter.transform(test), [[2, 0, 2], [0, 0, 0], [0, 1, 1]])
    assert state_hash(adapter) == before
    assert adapter.cardinality == 3


def test_dense_embedding_fit_reuses_locked_training_loop():
    train = pl.DataFrame({"g1_crs_elapsed_min": [60, 120, None], "g1_origin": ["ATL", "LAX", None]})
    adapter = source_mlp.SourceAdapter.fit(train.columns, train)
    matrix = adapter.transform(train)
    candidate = dict(widths=[8, 4], learning_rate=0.001, dropout=0.3)
    settings = dict(threads=1, batch_size=2, weight_decay=0.0001, patience=2)
    kwargs = dict(model_factory=adapter.factory(), weighted=True)
    args = (matrix, np.array([0, 1, 2]), candidate, settings, 6740, "cpu", 2)
    first, _, _ = mlp.train(*args, **kwargs)
    second, _, _ = mlp.train(*args, **kwargs)
    assert mlp.state_sha(first) == mlp.state_sha(second)
    p = mlp.predict(first, matrix, 2, "cpu")
    assert p.shape == (3, 3)
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-6)
