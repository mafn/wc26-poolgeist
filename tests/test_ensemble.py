import pytest

from poolgeist.models.ensemble import ModelCouncil, blend_signals, normalize_weights
from poolgeist.models.poisson import PoissonGoalsModel


def test_ensemble_weights_normalize():
    signals = [
        PoissonGoalsModel(home_xg=1.2, away_xg=1.1).predict_match("A", "B"),
        PoissonGoalsModel(home_xg=1.0, away_xg=1.4).predict_match("A", "B"),
    ]
    weights = normalize_weights(signals)
    assert sum(weights) == pytest.approx(1.0)


def test_blend_signal_sums_to_one():
    signal = blend_signals(
        [
            PoissonGoalsModel(home_xg=1.2, away_xg=1.1).predict_match("A", "B"),
            PoissonGoalsModel(home_xg=1.0, away_xg=1.4).predict_match("A", "B"),
        ]
    )
    assert sum(signal.tendency_probs.values()) == pytest.approx(1.0)
    assert signal.score_matrix.sum() == pytest.approx(1.0)


def test_model_council_recommendations_exist():
    prediction = ModelCouncil().predict_match("Neutral A", "Neutral B")
    assert prediction.recommendation_classes["safest"]
    assert prediction.chaos_index >= 0
