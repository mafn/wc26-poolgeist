"""Spin-flip volatility model for favourite-control-to-chaos states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from poolgeist.models.base import matrix_to_signal
from poolgeist.schemas import ModelSignal


@dataclass(frozen=True)
class SpinFlipOutput:
    """Volatility modifiers estimated by the spin-flip model."""

    flip_probability: float
    volatility_modifier: float
    upset_modifier: float
    late_goal_modifier: float


class SpinFlipVolatilityModel:
    """Estimate match-state flip probability from pressure and vulnerability features."""

    default_weight = 0.01

    def __init__(
        self,
        *,
        volatility: float = 0.2,
        early_goal_sensitivity: float = 0.2,
        red_card_vulnerability: float = 0.1,
        discipline_risk: float = 0.1,
        high_line_vulnerability: float = 0.1,
        underdog_transition_threat: float = 0.1,
        crowd_host_pressure: float = 0.0,
        knockout_pressure: float = 0.0,
        max_goals: int = 10,
    ):
        self.features = [
            volatility,
            early_goal_sensitivity,
            red_card_vulnerability,
            discipline_risk,
            high_line_vulnerability,
            underdog_transition_threat,
            crowd_host_pressure,
            knockout_pressure,
        ]
        if any(not 0 <= feature <= 1 for feature in self.features):
            raise ValueError("spin-flip features must be between 0 and 1")
        self.max_goals = max_goals

    def estimate(self, features: list[float] | None = None) -> SpinFlipOutput:
        """Estimate flip, upset, volatility, and late-goal modifiers."""

        feats = features if features is not None else self.features
        score = float(np.mean(feats))
        flip = float(np.clip(0.08 + 0.55 * score, 0.02, 0.70))
        return SpinFlipOutput(flip, 1.0 + flip, 0.5 * flip, 0.35 * flip)

    def predict_match(self, home_team: str, away_team: str) -> ModelSignal:
        """Return a valid high-uncertainty score signal."""

        feats = list(self.features)
        if getattr(self, "team_modifiers", None):
            home_mods = self.team_modifiers.get(home_team, {})
            away_mods = self.team_modifiers.get(away_team, {})
            avg_chaos = 0.5 * (
                home_mods.get("chaos_modifier", 0.0) + away_mods.get("chaos_modifier", 0.0)
            )
            feats[0] = float(np.clip(feats[0] + avg_chaos, 0.0, 1.0))

        out = self.estimate(features=feats)
        goals = np.arange(self.max_goals + 1)
        matrix = np.exp(-0.45 * (goals[:, None] + goals[None, :]))
        high_total = goals[:, None] + goals[None, :] >= 3
        matrix[high_total] *= out.volatility_modifier
        matrix[goals[:, None] < goals[None, :]] *= 1.0 + out.upset_modifier
        return matrix_to_signal(
            matrix,
            model_name="spin_flip",
            model_weight=self.default_weight,
            home_team=home_team,
            away_team=away_team,
            explanations=[
                "Spin-flip estimates probability that control breaks into a volatile state."
            ],
            metadata=out.__dict__,
        )
