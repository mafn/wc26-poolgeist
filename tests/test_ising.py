import math

import pytest

from poolgeist.models.ising import IsingPressureModel
from poolgeist.models.poisson import PoissonGoalsModel
from poolgeist.models.spin_flip import SpinFlipVolatilityModel


def test_ising_outputs_valid_probabilities():
    signal = IsingPressureModel(home_pressure=0.1, away_pressure=-0.1).predict_match("A", "B")
    assert sum(signal.tendency_probs.values()) == pytest.approx(1.0)
    assert all(value >= 0 for value in signal.tendency_probs.values())
    assert "pressure_modifier" in signal.metadata


def test_no_nans_in_basic_model_outputs():
    models = [IsingPressureModel(), SpinFlipVolatilityModel(), PoissonGoalsModel()]
    for model in models:
        signal = model.predict_match("Neutral A", "Neutral B")
        assert all(math.isfinite(value) for value in signal.tendency_probs.values())
