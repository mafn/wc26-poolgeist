"""Single-match simulation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from poolgeist.models.base import ModelSignal


@dataclass(frozen=True)
class MatchResult:
    """Simulated match result."""

    home_team: str
    away_team: str
    outcome: str


def simulate_outcome(
    home_team: str,
    away_team: str,
    signal: ModelSignal,
    generator: np.random.Generator | None = None,
) -> MatchResult:
    """Sample one home/draw/away outcome from a model signal."""

    generator = generator or np.random.default_rng()
    outcomes = np.array(["home", "draw", "away"])
    sampled = generator.choice(outcomes, p=[signal.home, signal.draw, signal.away])
    return MatchResult(home_team=home_team, away_team=away_team, outcome=str(sampled))
