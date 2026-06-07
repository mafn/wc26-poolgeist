import numpy as np
import pytest

from poolgeist.config import ScoringConfig, load_scoring_config
from poolgeist.models.base import matrix_to_signal
from poolgeist.models.ensemble import ModelCouncil
from poolgeist.scoring import score_prediction, tip_points


class StaticDrawModel:
    default_weight = 1.0

    def predict_match(self, home_team: str, away_team: str):
        matrix = np.zeros((3, 3), dtype=float)
        matrix[1, 1] = 1.0
        return matrix_to_signal(
            matrix,
            model_name="static_draw",
            model_weight=1.0,
            home_team=home_team,
            away_team=away_team,
        )


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


def test_custom_draw_tendency_config():
    config = ScoringConfig(correct_draw_tendency=2)
    assert tip_points(0, 0, 1, 1, config) == 2
    assert tip_points(2, 2, 1, 1, config) == 2
    assert tip_points(2, 2, 2, 2, config) == 4


def test_load_scoring_config_from_nested_yaml(tmp_path):
    config_path = tmp_path / "poolgeist.yaml"
    config_path.write_text(
        "scoring:\n  correct_draw_tendency: 2\n  correct_goal_difference: 3\n",
        encoding="utf-8",
    )

    config = load_scoring_config(config_path)

    assert config.correct_draw_tendency == 2
    assert tip_points(0, 0, 1, 1, config) == 2


def test_model_council_uses_explicit_scoring_config_for_ev():
    prediction = ModelCouncil(
        models=[StaticDrawModel()],
        scoring_config=ScoringConfig(correct_draw_tendency=2),
    ).predict_match("Neutral A", "Neutral B")

    draw_candidate = prediction.score_ev_table[
        (prediction.score_ev_table["pred_home_goals"] == 0)
        & (prediction.score_ev_table["pred_away_goals"] == 0)
    ].iloc[0]

    assert draw_candidate.expected_points == 2
