"""Placeholder Ising-inspired pressure model.

This is a neutral toy model that converts abstract pressure terms into a valid
home/draw/away probability vector. It does not encode team-specific priors.
"""

from __future__ import annotations

import numpy as np

from poolgeist.models.base import ModelSignal


class IsingPressureModel:
    """Neutral Ising-style model using temperature and pressure imbalance."""

    default_weight = 0.15

    def __init__(
        self, *, temperature: float = 1.0, home_pressure: float = 0.0, away_pressure: float = 0.0
    ):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        self.home_pressure = home_pressure
        self.away_pressure = away_pressure

    def predict_match(self, home_team: str, away_team: str) -> ModelSignal:
        """Predict a match with neutral pressure dynamics."""

        del home_team, away_team
        imbalance = (self.home_pressure - self.away_pressure) / self.temperature
        home = np.exp(imbalance)
        away = np.exp(-imbalance)
        draw = np.exp(-abs(imbalance) * 0.5)
        return ModelSignal.from_weights(
            {"home": float(home), "draw": float(draw), "away": float(away)},
            weight=self.default_weight,
            source="ising_pressure",
            metadata={"temperature": self.temperature},
        )
