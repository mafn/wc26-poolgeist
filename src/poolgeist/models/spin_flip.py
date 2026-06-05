"""Placeholder spin-flip volatility model."""

from __future__ import annotations

from poolgeist.models.base import ModelSignal


class SpinFlipVolatilityModel:
    """Neutral model that shifts mass toward draws as volatility increases."""

    default_weight = 0.1

    def __init__(self, *, volatility: float = 0.2):
        if not 0 <= volatility <= 1:
            raise ValueError("volatility must be between 0 and 1")
        self.volatility = volatility

    def predict_match(self, home_team: str, away_team: str) -> ModelSignal:
        """Predict a match without team-specific priors."""

        del home_team, away_team
        draw = 0.28 + 0.2 * self.volatility
        side = (1.0 - draw) / 2.0
        return ModelSignal(
            home=side,
            draw=draw,
            away=side,
            weight=self.default_weight,
            source="spin_flip_volatility",
            metadata={"volatility": self.volatility},
        )
