"""Seeded Random Octopus Oracle sanity-check model."""

from __future__ import annotations

import hashlib
import warnings

import numpy as np

from poolgeist.models.base import matrix_to_signal
from poolgeist.schemas import ModelSignal


class RandomOctopusOracle:
    """Seeded random low-weight oracle; never intended to dominate predictions."""

    default_weight = 0.005

    def __init__(
        self, *, seed: int | None = None, concentration: float = 10.0, max_goals: int = 10
    ):
        if concentration <= 0:
            raise ValueError("concentration must be positive")
        self.seed = seed
        self.concentration = concentration
        self.max_goals = max_goals

    def predict_match(self, home_team: str, away_team: str) -> ModelSignal:
        """Return a deterministic-by-seed valid probability matrix."""

        key = f"{self.seed}:{home_team}:{away_team}"
        derived_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (2**32)
        generator = np.random.default_rng(derived_seed)
        matrix = generator.dirichlet(
            np.full((self.max_goals + 1) ** 2, self.concentration)
        ).reshape(self.max_goals + 1, self.max_goals + 1)
        return matrix_to_signal(
            matrix,
            model_name="random_octopus_oracle",
            model_weight=self.default_weight,
            home_team=home_team,
            away_team=away_team,
            explanations=["Seeded random oracle chaos pick for ablation sanity checks only."],
            metadata={
                "seed": self.seed,
                "novelty_only": True,
                "oracle_chaos_pick": self.oracle_chaos_pick(home_team, away_team),
            },
        )

    def oracle_chaos_pick(self, home_team: str, away_team: str) -> str:
        """Return one deterministic novelty pick label."""

        key = f"pick:{self.seed}:{home_team}:{away_team}"
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")
        rng = np.random.default_rng(seed)
        return f"{int(rng.integers(0, 4))}-{int(rng.integers(0, 4))}"


def warn_if_octopus_wins(octopus_score: float, serious_score: float) -> None:
    """Warn if backtests show the novelty oracle beating serious models."""

    if octopus_score > serious_score:
        warnings.warn(
            "Your serious model is not beating the octopus. Check calibration and assumptions.",
            stacklevel=2,
        )
