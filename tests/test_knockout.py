import numpy as np

from poolgeist.models.poisson import PoissonGoalsModel
from poolgeist.simulation.knockout import simulate_knockout_match
from poolgeist.simulation.penalties import shootout_win_probability


def test_knockout_flow_decides_advancement():
    result = simulate_knockout_match(
        "A",
        "B",
        PoissonGoalsModel(home_xg=0.05, away_xg=0.05, max_goals=1),
        generator=np.random.default_rng(7),
    )
    assert result.advanced_team in {"A", "B"}
    assert result.decided_by in {"normal_time", "extra_time", "penalties"}


def test_penalty_probability_clamped():
    assert 0.38 <= shootout_win_probability(manual_modifier=99) <= 0.62
