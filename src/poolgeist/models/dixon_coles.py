"""Low-score Dixon-Coles correction model."""

from __future__ import annotations

import numpy as np

from poolgeist.models.base import (
    adjust_xg_with_modifiers,
    independent_poisson_matrix,
    matrix_to_signal,
)
from poolgeist.schemas import ModelSignal


class DixonColesModel:
    """Low-score Dixon-Coles correction model."""

    default_weight = 0.12

    def __init__(
        self,
        *,
        home_xg: float = 1.35,
        away_xg: float = 1.15,
        max_goals: int = 10,
        rho: float = -0.08,
    ):
        if home_xg <= 0 or away_xg <= 0:
            raise ValueError("Expected goals must be positive.")
        self.home_xg = home_xg
        self.away_xg = away_xg
        self.max_goals = max_goals
        self.rho = rho

    def score_matrix(self, home_xg: float, away_xg: float) -> np.ndarray:
        """Return a Dixon-Coles corrected finite score matrix."""

        matrix = independent_poisson_matrix(home_xg, away_xg, self.max_goals)
        if self.max_goals >= 1:
            matrix[0, 0] *= max(0.0, 1.0 - home_xg * away_xg * self.rho)
            matrix[0, 1] *= max(0.0, 1.0 + home_xg * self.rho)
            matrix[1, 0] *= max(0.0, 1.0 + away_xg * self.rho)
            matrix[1, 1] *= max(0.0, 1.0 - self.rho)
        return matrix

    def predict_match(self, home_team: str, away_team: str) -> ModelSignal:
        """Return a dynamically calibrated Dixon-Coles score signal."""

        home_xg, away_xg = adjust_xg_with_modifiers(
            home_team,
            away_team,
            self.home_xg,
            self.away_xg,
            getattr(self, "team_modifiers", None),
        )

        matrix = self.score_matrix(home_xg, away_xg)
        return matrix_to_signal(
            matrix,
            model_name="dixon_coles",
            model_weight=self.default_weight,
            home_team=home_team,
            away_team=away_team,
            explanations=["Dixon-Coles low-score interaction calibrated from matchup xG and rho."],
            warnings=[],
            metadata={"home_xg": home_xg, "away_xg": away_xg, "rho": self.rho},
        )
