"""Low-score Dixon-Coles correction model."""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from poolgeist.models.base import matrix_to_signal
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
        self.home_xg = home_xg
        self.away_xg = away_xg
        self.max_goals = max_goals
        self.rho = rho

    def predict_match(self, home_team: str, away_team: str) -> ModelSignal:
        """Return a valid neutral score signal."""

        home_xg = self.home_xg
        away_xg = self.away_xg
        if getattr(self, "team_modifiers", None):
            home_mods = self.team_modifiers.get(home_team, {})
            away_mods = self.team_modifiers.get(away_team, {})
            home_attack = home_mods.get("attack_modifier", 0.0)
            away_defense = away_mods.get("defense_modifier", 0.0)
            home_xg = max(0.01, home_xg + home_attack + away_defense)

            away_attack = away_mods.get("attack_modifier", 0.0)
            home_defense = home_mods.get("defense_modifier", 0.0)
            away_xg = max(0.01, away_xg + away_attack + home_defense)

        goals = np.arange(self.max_goals + 1)
        home = poisson.pmf(goals, home_xg)
        away = poisson.pmf(goals, away_xg)
        matrix = np.outer(home, away)
        matrix[0, 0] *= 1.0 - self.rho
        matrix[1, 1] *= 1.0 - self.rho
        matrix[1, 0] *= 1.0 + self.rho
        matrix[0, 1] *= 1.0 + self.rho
        return matrix_to_signal(
            matrix,
            model_name="dixon_coles",
            model_weight=self.default_weight,
            home_team=home_team,
            away_team=away_team,
            explanations=["Low-score Dixon-Coles correction model."],
            warnings=[],
        )
