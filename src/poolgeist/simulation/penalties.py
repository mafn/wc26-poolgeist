"""Penalty shootout helpers."""

from __future__ import annotations

import numpy as np


def simulate_penalty_winner(
    team_a: str,
    team_b: str,
    *,
    generator: np.random.Generator | None = None,
) -> str:
    """Choose a neutral penalty winner."""

    gen = generator or np.random.default_rng()
    return str(gen.choice([team_a, team_b]))
