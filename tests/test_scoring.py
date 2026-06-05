from poolgeist.scoring import score_prediction


def test_exact_score_gets_exact_points():
    assert score_prediction(2, 1, 2, 1) == 4


def test_correct_goal_difference_gets_difference_points():
    assert score_prediction(3, 1, 2, 0) == 3


def test_correct_tendency_gets_tendency_points():
    assert score_prediction(2, 0, 1, 0) == 3


def test_negative_goals_are_rejected():
    try:
        score_prediction(-1, 0, 1, 0)
    except ValueError:
        return
    raise AssertionError("negative goals should raise")
