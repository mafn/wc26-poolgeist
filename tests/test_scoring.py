import pytest

from poolgeist.scoring import score_prediction, tip_points


def test_required_tip_points_examples():
    assert tip_points(1, 2, 2, 1) == 0
    assert tip_points(1, 0, 2, 1) == 3
    assert tip_points(0, 0, 1, 1) == 3
    assert tip_points(1, 2, 1, 2) == 4
    assert tip_points(2, 0, 3, 1) == 3
    assert tip_points(2, 0, 3, 0) == 2
    assert tip_points(2, 2, 1, 1) == 3
    assert tip_points(2, 2, 2, 2) == 4


def test_score_prediction_wrapper_and_negative_goals():
    assert score_prediction(2, 1, 2, 1) == 4
    with pytest.raises(ValueError):
        score_prediction(-1, 0, 1, 0)
