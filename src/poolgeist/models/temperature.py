"""Temperature scenario model."""

from __future__ import annotations

import numpy as np

from poolgeist.models.base import (
    adjust_xg_with_modifiers,
    matchup_modifiers,
    matrix_to_signal,
)
from poolgeist.schemas import ModelSignal


class TemperatureChaosModel:
    """Temperature scenario model."""

    default_weight = 0.03

    def __init__(
        self,
        *,
        home_xg: float = 1.35,
        away_xg: float = 1.15,
        max_goals: int = 10,
        temperature_celsius: float = 26.0,
        humidity: float = 0.55,
        altitude_meters: float = 0.0,
        quadrature_points: int = 5,
    ):
        if home_xg <= 0 or away_xg <= 0:
            raise ValueError("Expected goals must be positive.")
        if not 0 <= humidity <= 1:
            raise ValueError("humidity must be between 0 and 1")
        if quadrature_points < 3:
            raise ValueError("quadrature_points must be at least 3")
        self.home_xg = home_xg
        self.away_xg = away_xg
        self.max_goals = max_goals
        self.temperature_celsius = temperature_celsius
        self.humidity = humidity
        self.altitude_meters = altitude_meters
        self.quadrature_points = quadrature_points

    def score_matrix(
        self, home_xg: float, away_xg: float, chaos: float, tempo: float
    ) -> np.ndarray:
        """Mix Poisson rates through lognormal weather shocks."""

        stress = self.thermal_stress()
        pace = float(np.exp(-0.10 * stress + 0.18 * tempo))
        sigma = float(np.clip(0.05 + 0.16 * stress + 0.55 * max(chaos, 0.0), 0.03, 0.45))
        common_sigma = 0.45 * sigma
        idiosyncratic_sigma = np.sqrt(max(sigma**2 - common_sigma**2, 0.0))
        nodes, weights = np.polynomial.hermite.hermgauss(self.quadrature_points)
        normal_weights = weights / np.sqrt(np.pi)

        shocks = np.sqrt(2.0) * nodes
        common_grid = shocks[:, np.newaxis, np.newaxis]
        home_grid = shocks[np.newaxis, :, np.newaxis]
        away_grid = shocks[np.newaxis, np.newaxis, :]

        home_rates = (
            home_xg
            * pace
            * np.exp(common_sigma * common_grid + idiosyncratic_sigma * home_grid - 0.5 * sigma**2)
        )
        away_rates = (
            away_xg
            * pace
            * np.exp(common_sigma * common_grid + idiosyncratic_sigma * away_grid - 0.5 * sigma**2)
        )

        goals = np.arange(self.max_goals + 1)
        from scipy.stats import poisson

        # home_probs/away_probs shape: (max_goals+1, Q, Q, Q)
        home_probs = poisson.pmf(
            goals[:, np.newaxis, np.newaxis, np.newaxis], home_rates[np.newaxis, ...]
        )
        away_probs = poisson.pmf(
            goals[:, np.newaxis, np.newaxis, np.newaxis], away_rates[np.newaxis, ...]
        )

        if self.max_goals > 0:
            home_probs[-1, ...] = poisson.sf(self.max_goals - 1, home_rates)
            away_probs[-1, ...] = poisson.sf(self.max_goals - 1, away_rates)

        # Ensure normalization across the goal axis
        home_probs /= home_probs.sum(axis=0, keepdims=True)
        away_probs /= away_probs.sum(axis=0, keepdims=True)

        # outer_probs shape: (goals, goals, Q, Q, Q)
        outer_probs = home_probs[:, np.newaxis, :, :, :] * away_probs[np.newaxis, :, :, :, :]

        # weight_grid shape: (Q, Q, Q)
        weight_grid = (
            normal_weights[:, np.newaxis, np.newaxis]
            * normal_weights[np.newaxis, :, np.newaxis]
            * normal_weights[np.newaxis, np.newaxis, :]
        )

        # Sum over the three quadrature axes
        matrix = np.sum(outer_probs * weight_grid[np.newaxis, np.newaxis, :, :, :], axis=(2, 3, 4))

        total = goals[:, None] + goals[None, :]
        late_fatigue_tail = 1.0 + 0.04 * stress * np.clip(total - 3, 0, None)
        return matrix * late_fatigue_tail

    def thermal_stress(self) -> float:
        """Return a bounded match-condition stress index."""

        heat = max(0.0, (self.temperature_celsius - 22.0) / 16.0)
        humidity = max(0.0, self.humidity - 0.45) * 0.9
        altitude = max(0.0, self.altitude_meters) / 2800.0
        return float(np.clip(heat + humidity + altitude, 0.0, 1.8))

    def predict_match(self, home_team: str, away_team: str) -> ModelSignal:
        """Return a weather-chaos score signal."""

        home_xg, away_xg = adjust_xg_with_modifiers(
            home_team,
            away_team,
            self.home_xg,
            self.away_xg,
            getattr(self, "team_modifiers", None),
        )
        modifiers = matchup_modifiers(home_team, away_team, getattr(self, "team_modifiers", None))

        matrix = self.score_matrix(home_xg, away_xg, modifiers["chaos"], modifiers["tempo"])
        return matrix_to_signal(
            matrix,
            model_name="temperature_chaos",
            model_weight=self.default_weight,
            home_team=home_team,
            away_team=away_team,
            explanations=[
                "Temperature chaos model mixes Poisson rates through lognormal thermal shocks."
            ],
            warnings=[],
            metadata={
                "home_xg": home_xg,
                "away_xg": away_xg,
                "thermal_stress": self.thermal_stress(),
                "temperature_celsius": self.temperature_celsius,
                "humidity": self.humidity,
                "altitude_meters": self.altitude_meters,
                "chaos_modifier": modifiers["chaos"],
                "tempo_modifier": modifiers["tempo"],
            },
        )
