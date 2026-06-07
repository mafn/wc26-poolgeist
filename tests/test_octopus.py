import pytest

from poolgeist.models.octopus import PaulOctopusOracle, RandomOctopusOracle


def test_octopus_deterministic_with_seed():
    first = PaulOctopusOracle(seed=2026).predict_match("Neutral A", "Neutral B")
    second = PaulOctopusOracle(seed=2026).predict_match("Neutral A", "Neutral B")
    assert first.score_matrix == pytest.approx(second.score_matrix)


def test_octopus_default_weight_stays_low():
    assert PaulOctopusOracle.default_weight <= 0.02


def test_backward_compatible_alias():
    assert RandomOctopusOracle is PaulOctopusOracle


def test_octopus_box_choice_is_deterministic():
    oracle = PaulOctopusOracle(seed=42)
    choice1 = oracle.oracle_box_choice("Neutral A", "Neutral B")
    choice2 = oracle.oracle_box_choice("Neutral A", "Neutral B")
    assert choice1 == choice2
    assert choice1 in {"home", "away"}


def test_octopus_model_name_is_paul():
    signal = PaulOctopusOracle(seed=2026).predict_match("Neutral A", "Neutral B")
    assert signal.model_name == "paul_octopus_oracle"
    assert "oracle_chose_home" in signal.metadata
    assert "paul_accuracy" in signal.metadata


def test_octopus_matrix_sums_to_one():
    signal = PaulOctopusOracle(seed=2026).predict_match("Neutral A", "Neutral B")
    assert signal.score_matrix.sum() == pytest.approx(1.0)
