import numpy as np
import pytest
from scipy.sparse import csr_matrix
from flight_delay.models.baseline import (
    PriorBaseline,
    inverse_cdf,
    sampled_classes,
    uniform_draws,
)
from flight_delay.models.state import state_hash
from flight_delay.experiments.evaluation import evaluate, airport_metrics
from flight_delay.experiments.results import prediction_frame


def test_cdf_boundaries_zero_mass_and_invalid_uniforms():
    p = np.tile([0.2, 0.3, 0.5], (5, 1))
    assert inverse_cdf(p, [0, 0.199, 0.2, 0.5, 0.999]).tolist() == [0, 0, 1, 2, 2]
    assert inverse_cdf(
        np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), [0, 0.9, 0.1]
    ).tolist() == [2, 0, 1]
    for invalid in [[-0.1], [1], [np.nan], [np.inf], [[0.2]]]:
        with pytest.raises(ValueError):
            inverse_cdf(np.array([[0.2, 0.3, 0.5]]), invalid)


def test_sampling_is_seeded_order_and_batch_invariant_without_state_mutation():
    ids = np.array([f"case-{i}" for i in range(1000)])
    X = csr_matrix((1000, 0))
    model = PriorBaseline().fit(X, [0] * 700 + [1] * 200 + [2] * 100)
    before = state_hash(model)
    p = model.predict_proba(X)
    predicted = model.predict(X, ids=ids)
    assert state_hash(model) == before
    assert np.array_equal(predicted, model.predict(X, ids=ids))
    assert np.array_equal(predicted[::-1], sampled_classes(p[::-1], ids[::-1]))
    assert np.array_equal(
        predicted,
        np.concatenate(
            [sampled_classes(p[:213], ids[:213]), sampled_classes(p[213:], ids[213:])]
        ),
    )
    assert not np.array_equal(predicted, sampled_classes(p, ids, seed=6741))
    assert np.all((uniform_draws(ids) >= 0) & (uniform_draws(ids) < 1))


def test_sampled_frequencies_match_historical_priors_and_ignore_outcomes():
    ids = [f"independent-{i}" for i in range(100000)]
    p = np.tile([0.7, 0.2, 0.1], (len(ids), 1))
    predicted = sampled_classes(p, ids)
    assert np.allclose(
        np.bincount(predicted, minlength=3) / len(ids), p[0], atol=0.005, rtol=0
    )
    assert set(predicted) == {0, 1, 2}
    frame = prediction_frame(
        "test",
        "baseline",
        "independent",
        "development",
        "2025-04",
        ids,
        np.zeros(len(ids)),
        p,
        "fit",
    )
    changed = prediction_frame(
        "test",
        "baseline",
        "independent",
        "development",
        "2025-04",
        ids,
        np.full(len(ids), 2),
        p,
        "fit",
    )
    assert frame["y_pred"].equals(changed["y_pred"])


def test_classification_metrics_use_sample_but_probability_scores_use_priors():
    y = np.array([0, 1, 2])
    p = np.tile([0.7, 0.2, 0.1], (3, 1))
    prior = evaluate(y, p)
    sampled = evaluate(y, p, predicted=y)
    assert sampled["accuracy"] == sampled["macro_f1"] == 1
    assert sampled["argmax_proportions"] == [1, 0, 0]
    assert sampled["predicted_proportions"] == [1 / 3] * 3
    for key in ["log_loss", "brier", "auc", "calibration", "mean_probabilities"]:
        assert sampled[key] == prior[key]
    slices = airport_metrics(y, p, ["ATL", "DFW", "ATL"], ["ATL", "DFW"], predicted=y)
    assert slices["ATL"]["accuracy"] == slices["DFW"]["accuracy"] == 1
    for invalid in [[0, 1], [0, 1, 3], [0, np.nan, 2]]:
        with pytest.raises(ValueError):
            evaluate(y, p, predicted=invalid)
