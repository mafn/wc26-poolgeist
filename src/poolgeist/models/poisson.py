"""Independent Poisson score model."""

from __future__ import annotations

import numpy as np

from poolgeist.models.base import (
    adjust_xg_with_modifiers,
    independent_poisson_matrix,
    matrix_to_signal,
)
from poolgeist.schemas import ModelSignal


class PoissonGoalsModel:
    """Calibrated independent Poisson score model."""

    default_weight = 0.32

    def __init__(self, *, home_xg: float = 1.35, away_xg: float = 1.15, max_goals: int = 10):
        if home_xg <= 0 or away_xg <= 0:
            raise ValueError("Expected goals must be positive.")
        self.home_xg = home_xg
        self.away_xg = away_xg
        self.max_goals = max_goals

    def score_matrix(
        self, home_xg: float | None = None, away_xg: float | None = None
    ) -> np.ndarray:
        """Return an exact finite score probability matrix."""

        h_xg = home_xg if home_xg is not None else self.home_xg
        a_xg = away_xg if away_xg is not None else self.away_xg
        return independent_poisson_matrix(h_xg, a_xg, self.max_goals)

    def predict_match(self, home_team: str, away_team: str) -> ModelSignal:
        """Predict a match by integrating a truncated Poisson score grid."""

        home_xg, away_xg = adjust_xg_with_modifiers(
            home_team,
            away_team,
            self.home_xg,
            self.away_xg,
            getattr(self, "team_modifiers", None),
        )

        return matrix_to_signal(
            self.score_matrix(home_xg, away_xg),
            model_name="independent_poisson",
            model_weight=self.default_weight,
            home_team=home_team,
            away_team=away_team,
            explanations=[
                "Independent Poisson matrix calibrated from matchup modifiers with tail mass "
                "folded into the final goal bucket."
            ],
            metadata={"home_xg": home_xg, "away_xg": away_xg},
        )
