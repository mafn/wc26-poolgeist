"""Seeded Random Octopus Oracle.

A deliberately tiny-weight novelty model. It is deterministic when seeded and
is not intended to be predictive or prescriptive.
"""

from __future__ import annotations

import hashlib

import numpy as np

from poolgeist.models.base import ModelSignal


class RandomOctopusOracle:
    """Random low-weight oracle for playful ensemble diversity."""

    default_weight = 0.01

    def __init__(self, *, seed: int | None = None, concentration: float = 10.0):
        if concentration <= 0:
            raise ValueError("concentration must be positive")
        self.seed = seed
        self.concentration = concentration

    def predict_match(self, home_team: str, away_team: str) -> ModelSignal:
        """Return a seeded random, valid probability signal."""

        key = f"{self.seed}:{home_team}:{away_team}"
        derived_seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % (
            2**32
        )
        generator = np.random.default_rng(derived_seed)
        probs = generator.dirichlet(np.full(3, self.concentration))
        return ModelSignal(
            home=float(probs[0]),
            draw=float(probs[1]),
            away=float(probs[2]),
            weight=self.default_weight,
            source="random_octopus_oracle",
            metadata={"seed": self.seed, "novelty_only": True},
        )
