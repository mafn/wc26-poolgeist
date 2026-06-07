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

    def score_matrix(
        self, home_xg: float | None = None, away_xg: float | None = None
    ) -> np.ndarray:
        """Return a normalized score probability matrix."""

        h_xg = home_xg if home_xg is not None else self.home_xg
        a_xg = away_xg if away_xg is not None else self.away_xg
        goals = np.arange(self.max_goals + 1)
        return np.outer(poisson.pmf(goals, h_xg), poisson.pmf(goals, a_xg))

    def predict_match(self, home_team: str, away_team: str) -> ModelSignal:
        """Predict a match by integrating a truncated Poisson score grid."""

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

        return matrix_to_signal(
            self.score_matrix(home_xg, away_xg),
            model_name="independent_poisson",
            model_weight=self.default_weight,
            home_team=home_team,
            away_team=away_team,
            explanations=["Independent Poisson baseline with neutral configurable xG inputs."],
            metadata={"home_xg": home_xg, "away_xg": away_xg},
        )
