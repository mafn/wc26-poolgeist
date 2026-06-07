from poolgeist.models.poisson import PoissonGoalsModel
from poolgeist.simulation.tournament import monte_carlo_champions, simulate_world_cup_2026


def test_tournament_orchestrator():
    # 48 teams
    teams = [f"Team {i}" for i in range(48)]
    model = PoissonGoalsModel(home_xg=1.2, away_xg=1.1)

    # Test single simulation
    champion = simulate_world_cup_2026(teams, model)
    assert champion in teams

    # Test monte carlo
    results = monte_carlo_champions(teams, model, n_simulations=5, seed=2026)
    assert len(results) > 0
    assert results.iloc[0]["champion_probability"] > 0
