"""Monte Carlo tournament simulation summaries."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd

from poolgeist.models.base import MatchModel
from poolgeist.simulation.knockout import simulate_knockout_match


def simulate_simple_knockout(
    teams: Sequence[str], model: MatchModel, *, seed: int | None = 2026
) -> pd.DataFrame:
    """Simulate a simple neutral knockout bracket and return champion counts."""

    generator = np.random.default_rng(seed)
    current = list(teams)
    while len(current) > 1:
        next_round = []
        for i in range(0, len(current), 2):
            next_round.append(
                simulate_knockout_match(
                    current[i], current[i + 1], model, generator=generator
                ).advanced_team
            )
        current = next_round
    return pd.DataFrame([{"team": current[0], "champion_count": 1}])


def monte_carlo_champions(
    teams: Sequence[str], model: MatchModel, *, n_simulations: int = 5_000, seed: int | None = 2026
) -> pd.DataFrame:
    """Run repeated simple brackets for quick-mode smoke tests and demos."""

    counts: Counter[str] = Counter()
    rng = np.random.default_rng(seed)
    for _ in range(n_simulations):
        order = list(teams)
        rng.shuffle(order)
        winner = simulate_simple_knockout(order, model, seed=int(rng.integers(0, 2**32 - 1))).iloc[
            0
        ]["team"]
        counts[str(winner)] += 1
    return pd.DataFrame(
        [
            {"team": team, "champion_probability": count / n_simulations}
            for team, count in counts.items()
        ]
    ).sort_values("champion_probability", ascending=False)
