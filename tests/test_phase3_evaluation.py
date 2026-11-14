import numpy as np
import pytest
from sklearn.metrics import log_loss
from flight_delay.experiments.evaluation import (
    evaluate,
    check_probabilities,
    airport_metrics,
)


def test_hand_calculated_probability_metrics_and_class_order():
    y = np.array([0, 1, 2])
    p = np.eye(3) * 0.7 + 0.1
    r = evaluate(y, p)
    assert r["accuracy"] == r["macro_f1"] == r["severe_recall"] == 1
    assert r["brier"] == pytest.approx(0.06)
    assert r["log_loss"] == pytest.approx(-np.log(0.8))
    assert r["log_loss"] == pytest.approx(log_loss(y, p, labels=[0, 1, 2]))
    assert r["confusion_matrix"] == np.eye(3, dtype=int).tolist()
    assert sum(c["n"] for c in r["calibration"]) == 9
    assert len(r["calibration"]) == 30


@pytest.mark.parametrize(
    "p",
    [
        np.array([[np.nan, 0, 1]]),
        np.array([[0.5, 0.5, 0.5]]),
        np.array([[-0.1, 0.1, 1]]),
        np.array([[1, 0]]),
    ],
)
def test_invalid_probabilities_fail(p):
    with pytest.raises(ValueError):
        check_probabilities(p, 1)


def test_missing_class_auc_is_explicit_na_and_zero_division_is_zero():
    r = evaluate([0, 0], np.array([[0.6, 0.2, 0.2], [0.7, 0.2, 0.1]]))
    assert r["auc"] == [None, None, None] and r["macro_auc"] is None
    assert r["recall"][2] == 0


def test_pooled_metrics_recompute_and_airport_uses_origin():
    y = np.array([0, 1, 2, 2])
    p = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]], dtype=float)
    pooled = evaluate(y, p)
    a, b = evaluate(y[:1], p[:1]), evaluate(y[1:], p[1:])
    assert pooled["macro_f1"] != pytest.approx((a["macro_f1"] + b["macro_f1"]) / 2)
    slices = airport_metrics(y, p, ["ATL", "DFW", "ATL", "DEN"], ["ATL", "DFW"])
    assert slices["ATL"]["n"] == 2 and slices["ATL"]["support"] == [1, 0, 1]
    assert slices["DFW"]["n"] == 1
