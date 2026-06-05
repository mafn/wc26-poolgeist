"""Penalty shootout model."""

from __future__ import annotations

import numpy as np


def shootout_win_probability(
    penalty_skill: float = 0.5,
    goalkeeper_quality: float = 0.5,
    mentality: float = 0.5,
    fatigue: float = 0.5,
    pressure: float = 0.5,
    manual_modifier: float = 0.0,
    *,
    allow_strong_modifier: bool = False,
) -> float:
    """Estimate shootout win probability, clamped unless explicitly overridden."""

    probability = (
        0.5
        + 0.08 * (penalty_skill - 0.5)
        + 0.06 * (goalkeeper_quality - 0.5)
        + 0.04 * (mentality - 0.5)
        - 0.04 * (fatigue - 0.5)
        - 0.03 * (pressure - 0.5)
        + manual_modifier
    )
    lo, hi = (0.30, 0.70) if allow_strong_modifier else (0.38, 0.62)
    return float(np.clip(probability, lo, hi))


def simulate_shootout(
    home_team: str, away_team: str, generator: np.random.Generator, *, home_probability: float = 0.5
) -> str:
    """Simulate advancement by penalties only; shootout goals are not normal goals."""

    return str(
        generator.choice([home_team, away_team], p=[home_probability, 1.0 - home_probability])
    )
