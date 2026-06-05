"""Independent Poisson score model."""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from poolgeist.models.base import matrix_to_signal
from poolgeist.schemas import ModelSignal


class PoissonGoalsModel:
    """Baseline score model using configurable expected goals."""

    default_weight = 0.32

    def __init__(self, *, home_xg: float = 1.35, away_xg: float = 1.15, max_goals: int = 10):
        if home_xg <= 0 or away_xg <= 0:
            raise ValueError("Expected goals must be positive.")
        self.home_xg = home_xg
        self.away_xg = away_xg
        self.max_goals = max_goals

    def score_matrix(self) -> np.ndarray:
        """Return a normalized score probability matrix."""

        goals = np.arange(self.max_goals + 1)
        return np.outer(poisson.pmf(goals, self.home_xg), poisson.pmf(goals, self.away_xg))

    def predict_match(self, home_team: str, away_team: str) -> ModelSignal:
        """Predict a match by integrating a truncated Poisson score grid."""

        return matrix_to_signal(
            self.score_matrix(),
            model_name="independent_poisson",
            model_weight=self.default_weight,
            home_team=home_team,
            away_team=away_team,
            explanations=["Independent Poisson baseline with neutral configurable xG inputs."],
            metadata={"home_xg": self.home_xg, "away_xg": self.away_xg},
        )
