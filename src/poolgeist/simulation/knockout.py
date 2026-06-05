"""Minimal knockout helpers."""

from __future__ import annotations

import numpy as np

from poolgeist.models.base import ModelSignal


def choose_winner(
    home_team: str,
    away_team: str,
    signal: ModelSignal,
    *,
    seed: int | None = None,
) -> str:
    """Choose a knockout winner, splitting draw probability equally before penalties."""

    generator = np.random.default_rng(seed)
    home_probability = signal.home + signal.draw / 2.0
    return str(
        generator.choice([home_team, away_team], p=[home_probability, 1.0 - home_probability])
    )
