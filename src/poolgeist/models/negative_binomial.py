"""Gamma-Poisson overdispersed goal model."""

from __future__ import annotations

import numpy as np
from scipy.stats import nbinom

from poolgeist.models.base import matrix_to_signal
from poolgeist.schemas import ModelSignal


class NegativeBinomialGoalsModel:
    """Gamma-Poisson overdispersed goal model."""

    default_weight = 0.1

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
        r = 2.5
        hp = r / (r + home_xg)
        ap = r / (r + away_xg)
        matrix = np.outer(nbinom.pmf(goals, r, hp), nbinom.pmf(goals, r, ap))
        return matrix_to_signal(
            matrix,
            model_name="negative_binomial",
            model_weight=self.default_weight,
            home_team=home_team,
            away_team=away_team,
            explanations=["Gamma-Poisson overdispersed goal model."],
            warnings=[],
        )
