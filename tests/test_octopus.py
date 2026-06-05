import pytest

from poolgeist.models.octopus import RandomOctopusOracle


def test_octopus_deterministic_with_seed():
    first = RandomOctopusOracle(seed=2026).predict_match("Neutral A", "Neutral B")
    second = RandomOctopusOracle(seed=2026).predict_match("Neutral A", "Neutral B")
    assert first.score_matrix == pytest.approx(second.score_matrix)


def test_octopus_default_weight_stays_low():
    assert RandomOctopusOracle.default_weight <= 0.02
