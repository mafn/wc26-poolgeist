"""Match simulation primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from poolgeist.schemas import ModelSignal


@dataclass(frozen=True)
class MatchResult:
    """A simulated match result."""

    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    winner: str | None = None


def simulate_score(signal: ModelSignal, generator: np.random.Generator) -> tuple[int, int]:
    """Sample an exact score from a model signal."""

    flat_index = int(generator.choice(signal.score_matrix.size, p=signal.score_matrix.ravel()))
    return tuple(int(v) for v in np.unravel_index(flat_index, signal.score_matrix.shape))


def simulate_outcome(
    home_team: str,
    away_team: str,
    signal: ModelSignal,
    generator: np.random.Generator,
) -> MatchResult:
    """Simulate a 90-minute match outcome."""

    home_goals, away_goals = simulate_score(signal, generator)
    winner = (
        home_team if home_goals > away_goals else away_team if away_goals > home_goals else None
    )
    return MatchResult(home_team, away_team, home_goals, away_goals, winner)
