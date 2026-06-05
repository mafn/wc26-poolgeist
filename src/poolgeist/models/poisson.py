"""Neutral Poisson goal model."""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from poolgeist.models.base import ModelSignal


class PoissonGoalsModel:
    """Predict outcomes from neutral expected goals."""

    default_weight = 0.5

    def __init__(self, *, home_xg: float = 1.35, away_xg: float = 1.15, max_goals: int = 10):
        if home_xg <= 0 or away_xg <= 0:
            raise ValueError("Expected goals must be positive.")
        self.home_xg = home_xg
        self.away_xg = away_xg
        self.max_goals = max_goals

    def predict_match(self, home_team: str, away_team: str) -> ModelSignal:
        """Predict a match by integrating a truncated score grid."""

        del home_team, away_team
        goals = np.arange(self.max_goals + 1)
        home_probs = poisson.pmf(goals, self.home_xg)
        away_probs = poisson.pmf(goals, self.away_xg)
        matrix = np.outer(home_probs, away_probs)
        return ModelSignal.from_weights(
            {
                "home": float(np.tril(matrix, k=-1).sum()),
                "draw": float(np.trace(matrix)),
                "away": float(np.triu(matrix, k=1).sum()),
            },
            weight=self.default_weight,
            source="poisson_goals",
            metadata={"home_xg": self.home_xg, "away_xg": self.away_xg},
        )
