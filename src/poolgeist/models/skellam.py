"""Skellam-inspired goal-difference signal."""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson, skellam

from poolgeist.models.base import adjust_xg_with_modifiers, matrix_to_signal
from poolgeist.schemas import ModelSignal


class SkellamGoalDifferenceModel:
    """Skellam-inspired goal-difference signal."""

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

        home_xg, away_xg = adjust_xg_with_modifiers(
            home_team,
            away_team,
            self.home_xg,
            self.away_xg,
            getattr(self, "team_modifiers", None),
        )

        goals = np.arange(self.max_goals + 1)
        home = poisson.pmf(goals, home_xg)
        away = poisson.pmf(goals, away_xg)
        matrix = np.outer(home, away)
        diffs = goals[:, None] - goals[None, :]
        for diff in np.unique(diffs):
            mass = float(max(skellam.pmf(int(diff), home_xg, away_xg), 1e-6))
            matrix[diffs == diff] *= mass
        return matrix_to_signal(
            matrix,
            model_name="skellam_goal_difference",
            model_weight=self.default_weight,
            home_team=home_team,
            away_team=away_team,
            explanations=["Skellam-inspired goal-difference signal."],
            warnings=[],
        )
