"""Penalty shootout helpers."""

from __future__ import annotations

import numpy as np


def simulate_penalty_winner(team_a: str, team_b: str, *, seed: int | None = None) -> str:
    """Choose a neutral penalty winner."""

    generator = np.random.default_rng(seed)
    return str(generator.choice([team_a, team_b]))
