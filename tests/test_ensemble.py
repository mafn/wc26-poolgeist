from dataclasses import replace

import numpy as np
import pytest

from poolgeist.config import EnsembleConfig, ModelWeightsConfig
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


def _only_weight(enabled_name: str, value: float = 1.0) -> ModelWeightsConfig:
    weights = ModelWeightsConfig(
        poisson=0.0,
        dixon_coles=0.0,
        negative_binomial=0.0,
        correlated_goals=0.0,
        skellam=0.0,
        market=0.0,
        team_cards=0.0,
        news=0.0,
        bayesian_update=0.0,
        temperature=0.0,
        ising=0.0,
        spin_flip=0.0,
        octopus=0.0,
    )
    return replace(weights, **{enabled_name: value})


def test_model_council_applies_configured_zero_weight_ablation():
    config = EnsembleConfig(weights=_only_weight("poisson"))
    prediction = ModelCouncil(config=config).predict_match("Neutral A", "Neutral B")

    assert [signal.model_name for signal in prediction.component_signals] == ["independent_poisson"]
    assert prediction.signal.metadata["component_weights"] == {"independent_poisson": 1.0}


def test_model_council_configured_weights_change_blend():
    poisson_only = ModelCouncil(
        config=EnsembleConfig(weights=_only_weight("poisson"))
    ).predict_match("Neutral A", "Neutral B")
    octopus_only = ModelCouncil(
        config=EnsembleConfig(weights=_only_weight("octopus"))
    ).predict_match("Neutral A", "Neutral B")

    assert not np.allclose(poisson_only.signal.score_matrix, octopus_only.signal.score_matrix)
    assert octopus_only.signal.metadata["component_weights"] == {"paul_octopus_oracle": 1.0}
