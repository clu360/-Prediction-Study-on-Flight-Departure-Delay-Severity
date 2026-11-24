import numpy as np
import pytest
from scipy.sparse import csr_matrix

from flight_delay.experiments.phase4_final import risk_metrics, xgb_config
from flight_delay.experiments.evaluation import check_probabilities


def test_xgb_locked_manifest():
    config = xgb_config()
    assert config["params"]["n_estimators"] == 400
    assert config["params"]["objective"] == "multi:softprob"
    assert (config["seed"], config["threads"]) == (6740, 4)


def test_mlp_weights_and_frozen_deterministic_probabilities():
    pytest.importorskip("torch")
    from flight_delay.models.mlp import predict, state_sha, train, training_weights

    y = np.array([0, 0, 0, 1, 1, 2])
    X = csr_matrix(np.eye(len(y), dtype=np.float32))
    np.testing.assert_allclose(training_weights(y), [2/3, 1, 2])
    with pytest.raises(ValueError, match="missing a class"):
        training_weights(y[y != 2])
    candidate = dict(widths=[8, 4], learning_rate=0.001, dropout=0.1)
    settings = dict(threads=1, batch_size=2, weight_decay=0.0001, patience=1)
    first, epoch, _ = train(X, y, candidate, settings, 6740, "cpu", 2, weighted=True)
    second, _, _ = train(X, y, candidate, settings, 6740, "cpu", 2, weighted=True)
    assert epoch == 2
    frozen = state_sha(first)
    p = predict(first, X, 2, "cpu")
    check_probabilities(p, len(y))
    np.testing.assert_array_equal(p, predict(second, X, 2, "cpu"))
    assert state_sha(first) == frozen == state_sha(second)
    metrics = risk_metrics(y, p)
    assert metrics["observed_proportions"][2] == pytest.approx(1/6)
    assert metrics["severe_pr_auc_ap"] is not None

