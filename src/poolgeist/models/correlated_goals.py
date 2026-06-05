"""Correlated bivariate score layer for open games."""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from poolgeist.models.base import matrix_to_signal
from poolgeist.schemas import ModelSignal


class CorrelatedGoalsModel:
    """Correlated bivariate score layer for open games."""

    default_weight = 0.08

    def __init__(
        self,
        *,
        home_xg: float = 1.35,
        away_xg: float = 1.15,
        max_goals: int = 10,
        rho: float = -0.08,
    ):
        self.home_xg = home_xg
        self.away_xg = away_xg
        self.max_goals = max_goals
        self.rho = rho

    def predict_match(self, home_team: str, away_team: str) -> ModelSignal:
        """Return a valid neutral score signal."""

        goals = np.arange(self.max_goals + 1)
        home = poisson.pmf(goals, self.home_xg)
        away = poisson.pmf(goals, self.away_xg)
        matrix = np.outer(home, away)
        matrix[1:, 1:] *= 1.15
        return matrix_to_signal(
            matrix,
            model_name="correlated_goals",
            model_weight=self.default_weight,
            home_team=home_team,
            away_team=away_team,
            explanations=["Correlated bivariate score layer for open games."],
            warnings=[],
        )
