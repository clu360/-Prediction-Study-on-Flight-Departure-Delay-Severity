import pytest

from flight_delay.experiments.phase4_inputs import BASE, ID
from flight_delay.models.state import state_hash
from test_phase4_alignment import phase4_inputs  # noqa: F401


def test_locked_development_and_holdout_windows(phase4_inputs):
    inputs = phase4_inputs
    april_gradient = inputs.rows(4, "train", months=[1, 2])
    april_stop = inputs.rows(4, "train", months=[3])
    april_scored = inputs.rows(4, "test")
    assert set(april_gradient["g1_month"]) == {1, 2}
    assert set(april_stop["g1_month"]) == {3}
    assert set(april_scored["g1_month"]) == {4}

    train = inputs.rows(11, "train")
    assert set(train["g1_month"]) == {8, 9, 10}
    with pytest.raises(ValueError, match="cannot enter training"):
        inputs.rows(12, "train")
    with pytest.raises(ValueError, match="outside"):
        inputs.rows(11, "train", months=[11])
    transformation = inputs.transformer(BASE, train)
    fitted_state = state_hash(transformation)
    for month in [11, 12]:
        evaluation = inputs.rows(month, "test")
        assert not set(train[ID]) & set(evaluation[ID])
        assert transformation.transform(evaluation).shape[0] == evaluation.height
        assert transformation.unknown_counts(evaluation)["g1_month"] == evaluation.height
        assert state_hash(transformation) == fitted_state

