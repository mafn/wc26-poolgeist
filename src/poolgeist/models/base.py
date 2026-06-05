"""Base model protocols and signal containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from poolgeist.utils.probability import assert_probability_vector, normalize


@dataclass(frozen=True)
class ModelSignal:
    """Outcome probabilities produced by a model."""

    home: float
    draw: float
    away: float
    weight: float = 1.0
    source: str = "model"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert_probability_vector(self.probabilities)
        if self.weight < 0:
            raise ValueError("Model weight must be non-negative.")

    @property
    def probabilities(self) -> dict[str, float]:
        """Return probabilities as a mapping."""

        return {"home": self.home, "draw": self.draw, "away": self.away}

    @classmethod
    def from_weights(
        cls,
        weights: dict[str, float],
        *,
        weight: float = 1.0,
        source: str = "model",
        metadata: dict[str, Any] | None = None,
    ) -> ModelSignal:
        """Build a signal from unnormalized outcome weights."""

        probs = normalize(weights)
        return cls(
            home=probs["home"],
            draw=probs["draw"],
            away=probs["away"],
            weight=weight,
            source=source,
            metadata=metadata or {},
        )


class MatchModel(Protocol):
    """Protocol implemented by match-outcome models."""

    default_weight: float

    def predict_match(self, home_team: str, away_team: str) -> ModelSignal:
        """Predict a single match."""
