"""Minimal knockout helpers."""

from __future__ import annotations

import numpy as np

from poolgeist.models.base import ModelSignal


def choose_winner(
    home_team: str,
    away_team: str,
    signal: ModelSignal,
    *,
    generator: np.random.Generator | None = None,
) -> str:
    """Choose a knockout winner, splitting draw probability equally before penalties."""

    gen = generator or np.random.default_rng()
    home_probability = float(np.clip(signal.home + signal.draw / 2.0, 0.0, 1.0))
    return str(
        gen.choice([home_team, away_team], p=[home_probability, 1.0 - home_probability])
    )
