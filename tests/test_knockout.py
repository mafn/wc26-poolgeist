import numpy as np

from poolgeist.models.poisson import PoissonGoalsModel
from poolgeist.simulation.knockout import simulate_knockout_match
from poolgeist.simulation.penalties import (
    shootout_win_probability,
    shootout_win_probability_from_tendency,
)


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


def test_penalty_probability_uses_tendency_strength():
    assert shootout_win_probability_from_tendency({"home": 0.4, "draw": 0.2, "away": 0.4}) == 0.5
    assert shootout_win_probability_from_tendency({"home": 0.7, "draw": 0.1, "away": 0.2}) > 0.5
    assert shootout_win_probability_from_tendency({"home": 0.2, "draw": 0.1, "away": 0.7}) < 0.5


def test_knockout_penalties_exact_scores():
    # Find a seed that goes to penalties
    found = False
    for seed in range(50):
        result = simulate_knockout_match(
            "A",
            "B",
            PoissonGoalsModel(home_xg=0.1, away_xg=0.1, max_goals=2),
            generator=np.random.default_rng(seed),
        )
        if result.decided_by == "penalties":
            found = True
            assert result.penalties_home_goals > 0 or result.penalties_away_goals > 0
            assert result.penalties_home_goals != result.penalties_away_goals
            assert (
                result.final_home_goals
                == result.home_goals + result.extra_time_home_goals + result.penalties_home_goals
            )
            assert (
                result.final_away_goals
                == result.away_goals + result.extra_time_away_goals + result.penalties_away_goals
            )
            # Verify one team won
            if result.advanced_team == "A":
                assert result.final_home_goals > result.final_away_goals
            else:
                assert result.final_away_goals > result.final_home_goals
            break
    assert found, "Should find a shootout match in 50 random trials"


def test_knockout_ev_matrix_optimization():
    from poolgeist.models.ensemble import ModelCouncil

    prediction = ModelCouncil().predict_match("Neutral A", "Neutral B", is_knockout=True)

    # Diagonal (draws) in the score matrix must have 0 probability
    diag = np.diag(prediction.signal.score_matrix)
    assert np.allclose(diag, 0.0)

    # EV table must contain valid predictions
    assert not prediction.score_ev_table.empty

    # Recommendations must not contain draws
    for rec_score in prediction.recommendation_classes.values():
        home_t, away_t = map(int, rec_score.split("-"))
        assert home_t != away_t


def test_knockout_ev_uses_blended_strength_for_shootout_probability():
    from poolgeist.models.ensemble import ModelCouncil

    prediction = ModelCouncil(models=[PoissonGoalsModel(home_xg=2.4, away_xg=0.4)]).predict_match(
        "Neutral A", "Neutral B", is_knockout=True
    )

    assert prediction.signal.metadata["knockout_home_shootout_probability"] > 0.5
