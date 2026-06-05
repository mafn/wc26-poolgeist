import math

import pytest

from poolgeist.models.correlated_goals import CorrelatedGoalsModel
from poolgeist.models.dixon_coles import DixonColesModel
from poolgeist.models.market import MarketPriorModel
from poolgeist.models.negative_binomial import NegativeBinomialGoalsModel
from poolgeist.models.news import NewsSignalModel
from poolgeist.models.skellam import SkellamGoalDifferenceModel
from poolgeist.models.temperature import TemperatureChaosModel


def test_model_signal_schema_validity_for_required_members():
    for cls in [
        DixonColesModel,
        NegativeBinomialGoalsModel,
        CorrelatedGoalsModel,
        SkellamGoalDifferenceModel,
        MarketPriorModel,
        NewsSignalModel,
        TemperatureChaosModel,
    ]:
        signal = cls().predict_match("Neutral A", "Neutral B")
        assert signal.score_matrix.sum() == pytest.approx(1.0)
        assert all(math.isfinite(v) for v in signal.tendency_probs.values())
