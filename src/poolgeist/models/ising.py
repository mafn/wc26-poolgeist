"""Ising-style tournament pressure abstraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from poolgeist.models.base import matrix_to_signal
from poolgeist.schemas import ModelSignal


@dataclass(frozen=True)
class IsingPressureOutput:
    """Explainable pressure modifiers for home/draw/away tendencies."""

    adjusted_tendency_probs: dict[str, float]
    pressure_modifier: float
    chaos_modifier: float
    draw_modifier: float
    attacking_risk_modifier: float
    explanations: list[str]


class IsingPressureModel:
    """Small-weight model exposing incentive pressure without dominating the council."""

    default_weight = 0.02

    def __init__(
        self,
        *,
        strength_difference: float = 0.0,
        group_incentive_pressure: float = 0.0,
        need_goal_difference: float = 0.0,
        already_qualified_rotation: float = 0.0,
        must_win_state: float = 0.0,
        fatigue_rest_asymmetry: float = 0.0,
        home_host_pressure: float = 0.0,
        match_importance: float = 0.5,
        temperature: float = 1.0,
        home_pressure: float | None = None,
        away_pressure: float | None = None,
        max_goals: int = 10,
    ):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if home_pressure is not None or away_pressure is not None:
            strength_difference += (home_pressure or 0.0) - (away_pressure or 0.0)
        self.strength_difference = strength_difference
        self.group_incentive_pressure = group_incentive_pressure
        self.need_goal_difference = need_goal_difference
        self.already_qualified_rotation = already_qualified_rotation
        self.must_win_state = must_win_state
        self.fatigue_rest_asymmetry = fatigue_rest_asymmetry
        self.home_host_pressure = home_host_pressure
        self.match_importance = match_importance
        self.temperature = temperature
        self.max_goals = max_goals

    def pressure_output(self) -> IsingPressureOutput:
        """Return Ising-inspired tendency probabilities and modifiers."""

        field = (
            self.strength_difference
            + 0.25 * self.group_incentive_pressure
            + 0.20 * self.need_goal_difference
            - 0.20 * self.already_qualified_rotation
            + 0.25 * self.must_win_state
            + 0.15 * self.fatigue_rest_asymmetry
            + 0.15 * self.home_host_pressure
        ) / self.temperature
        pressure = float(np.tanh(field))
        chaos = float(
            np.clip(0.12 + 0.18 * self.match_importance + 0.12 * (self.temperature - 1), 0, 0.55)
        )
        draw_mod = float(
            np.clip(0.30 - 0.10 * abs(pressure) - 0.08 * self.need_goal_difference, 0.15, 0.40)
        )
        home = (1.0 - draw_mod) * (1.0 + pressure) / 2.0
        away = (1.0 - draw_mod) - home
        return IsingPressureOutput(
            adjusted_tendency_probs={"home": float(home), "draw": draw_mod, "away": float(away)},
            pressure_modifier=pressure,
            chaos_modifier=chaos,
            draw_modifier=draw_mod,
            attacking_risk_modifier=float(
                np.clip(self.need_goal_difference + self.must_win_state, -1, 1)
            ),
            explanations=[
                "Ising states map tendency to +1 home, 0 draw, -1 away using incentive pressure."
            ],
        )

    def predict_match(self, home_team: str, away_team: str) -> ModelSignal:
        """Convert tendency pressure into a diffuse valid score matrix."""

        out = self.pressure_output()
        goals = np.arange(self.max_goals + 1)
        base = np.exp(-0.55 * (goals[:, None] + goals[None, :]))
        matrix = np.zeros_like(base, dtype=float)
        matrix[goals[:, None] > goals[None, :]] = (
            base[goals[:, None] > goals[None, :]] * out.adjusted_tendency_probs["home"]
        )
        matrix[goals[:, None] == goals[None, :]] = (
            base[goals[:, None] == goals[None, :]] * out.adjusted_tendency_probs["draw"]
        )
        matrix[goals[:, None] < goals[None, :]] = (
            base[goals[:, None] < goals[None, :]] * out.adjusted_tendency_probs["away"]
        )
        return matrix_to_signal(
            matrix,
            model_name="ising_pressure",
            model_weight=self.default_weight,
            home_team=home_team,
            away_team=away_team,
            explanations=out.explanations,
            metadata=out.__dict__,
        )
