"""Paul the Octopus Oracle — inspired by the Oberhausen Sea Life Centre legend.

Paul (2008–2010) famously chose between two boxes — each decorated with a
team flag — to predict match outcomes at Euro 2008 and the 2010 World Cup.
He achieved 12 correct predictions out of 14 (≈ 85.7%).

This model recreates Paul's ritual as a statistical mechanism:
1. A seeded RNG deterministically "chooses a box" (home or away) with
   probability driven by a logistic function of the strength gap.
2. The chosen side receives a concentrated Poisson-based score matrix
   tilted toward that team winning.
3. The unchosen side contributes a diffuse upset prior, reflecting Paul's
   rare incorrect picks.
4. The two matrices are mixed using Paul's historical accuracy as the
   blend weight, producing a final score distribution.

Reference: https://de.wikipedia.org/wiki/Paul_(Krake)
"""

from __future__ import annotations

import hashlib
import warnings

import numpy as np

from poolgeist.models.base import (
    adjust_xg_with_modifiers,
    independent_poisson_matrix,
    matchup_modifiers,
    matrix_to_signal,
)
from poolgeist.schemas import ModelSignal

# Paul's empirical accuracy: 12 correct out of 14 predictions.
PAUL_HISTORICAL_ACCURACY = 12.0 / 14.0


class PaulOctopusOracle:
    """Deterministic oracle modelling Paul the Octopus's two-box choice ritual.

    The oracle is designed as a low-weight novelty/ablation model.  It should
    never dominate the council; if backtest scores beat serious models, that
    signals a calibration problem in the main council.
    """

    default_weight = 0.005

    def __init__(
        self,
        *,
        seed: int | None = None,
        home_xg: float = 1.35,
        away_xg: float = 1.15,
        paul_accuracy: float = PAUL_HISTORICAL_ACCURACY,
        tentacle_sensitivity: float = 0.85,
        max_goals: int = 10,
    ):
        if home_xg <= 0 or away_xg <= 0:
            raise ValueError("Expected goals must be positive.")
        if not 0.5 <= paul_accuracy <= 1.0:
            raise ValueError("paul_accuracy must be between 0.5 and 1.0")
        if tentacle_sensitivity <= 0:
            raise ValueError("tentacle_sensitivity must be positive")
        self.seed = seed
        self.home_xg = home_xg
        self.away_xg = away_xg
        self.paul_accuracy = paul_accuracy
        self.tentacle_sensitivity = tentacle_sensitivity
        self.max_goals = max_goals

    def _matchup_seed(self, home_team: str, away_team: str, *, salt: str = "") -> int:
        """Derive a deterministic 32-bit seed from the matchup and instance seed."""

        key = f"{salt}{self.seed}:{home_team}:{away_team}"
        return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (2**32)

    def _box_choice_probability(self, strength_gap: float) -> float:
        """Return probability that Paul opens the *home* box.

        Uses a logistic link centred at 0.5 so that equally-matched teams
        get a coin-flip, while a large positive gap (home stronger) drives
        the probability toward 1.
        """

        return float(1.0 / (1.0 + np.exp(-self.tentacle_sensitivity * strength_gap)))

    def _chosen_matrix(
        self,
        home_xg: float,
        away_xg: float,
        chose_home: bool,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Build a concentrated score matrix tilted toward the chosen side."""

        if self.max_goals == 0:
            return np.array([[1.0]])

        if chose_home:
            boosted_home = home_xg * 1.18
            dampened_away = away_xg * 0.82
        else:
            boosted_home = home_xg * 0.82
            dampened_away = away_xg * 1.18

        matrix = independent_poisson_matrix(boosted_home, dampened_away, self.max_goals)

        # Apply a small Dirichlet jitter to avoid exact Poisson rigidity while
        # keeping the concentration high enough to preserve the shape.
        alpha = np.maximum(matrix.ravel() * 120.0, 0.01)
        return rng.dirichlet(alpha).reshape(self.max_goals + 1, self.max_goals + 1)

    def _upset_matrix(
        self,
        home_xg: float,
        away_xg: float,
        chose_home: bool,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Build a diffuse matrix representing Paul's rare wrong predictions."""

        if self.max_goals == 0:
            return np.array([[1.0]])

        # Flip the boost: Paul picked the wrong box.
        if chose_home:
            upset_home = home_xg * 0.78
            upset_away = away_xg * 1.25
        else:
            upset_home = home_xg * 1.25
            upset_away = away_xg * 0.78

        matrix = independent_poisson_matrix(upset_home, upset_away, self.max_goals)

        # Much lower concentration → more diffuse, chaotic scorelines
        alpha = np.maximum(matrix.ravel() * 25.0, 0.01)
        return rng.dirichlet(alpha).reshape(self.max_goals + 1, self.max_goals + 1)

    def predict_match(self, home_team: str, away_team: str) -> ModelSignal:
        """Simulate Paul's box-choice ritual and return a score distribution."""

        rng = np.random.default_rng(self._matchup_seed(home_team, away_team))

        home_xg, away_xg = adjust_xg_with_modifiers(
            home_team,
            away_team,
            self.home_xg,
            self.away_xg,
            getattr(self, "team_modifiers", None),
        )
        mods = matchup_modifiers(home_team, away_team, getattr(self, "team_modifiers", None))

        # Step 1: Paul approaches the two boxes.
        home_box_prob = self._box_choice_probability(mods["strength_gap"])
        chose_home = bool(rng.random() < home_box_prob)

        # Step 2: Build the "Paul was right" and "Paul was wrong" matrices.
        chosen_matrix = self._chosen_matrix(home_xg, away_xg, chose_home, rng)
        upset_matrix = self._upset_matrix(home_xg, away_xg, chose_home, rng)

        # Step 3: Blend with Paul's historical accuracy as the mixture weight.
        matrix = self.paul_accuracy * chosen_matrix + (1.0 - self.paul_accuracy) * upset_matrix

        # Pick the oracle's headline scoreline from the final distribution.
        pick = self._pick_from_matrix(matrix, rng)

        return matrix_to_signal(
            matrix,
            model_name="paul_octopus_oracle",
            model_weight=self.default_weight,
            home_team=home_team,
            away_team=away_team,
            explanations=[
                "Paul the Octopus oracle: deterministic two-box choice ritual "
                "inspired by Paul (2008–2010, Oberhausen Sea Life Centre). "
                f"Paul chose {'home' if chose_home else 'away'} box with "
                f"p={home_box_prob:.3f} for home."
            ],
            metadata={
                "seed": self.seed,
                "novelty_only": True,
                "oracle_chose_home": chose_home,
                "home_box_probability": home_box_prob,
                "paul_accuracy": self.paul_accuracy,
                "oracle_headline_pick": pick,
                "home_xg": home_xg,
                "away_xg": away_xg,
            },
        )

    def oracle_box_choice(self, home_team: str, away_team: str) -> str:
        """Return which box Paul opens: 'home' or 'away'."""

        rng = np.random.default_rng(self._matchup_seed(home_team, away_team))
        mods = matchup_modifiers(home_team, away_team, getattr(self, "team_modifiers", None))
        home_prob = self._box_choice_probability(mods["strength_gap"])
        return "home" if rng.random() < home_prob else "away"

    def _pick_from_matrix(self, matrix: np.ndarray, generator: np.random.Generator) -> str:
        index = int(generator.choice(matrix.size, p=matrix.ravel()))
        home_goals, away_goals = divmod(index, self.max_goals + 1)
        return f"{home_goals}-{away_goals}"


# Keep the backward-compatible alias so existing imports don't break.
RandomOctopusOracle = PaulOctopusOracle


def warn_if_octopus_wins(octopus_score: float, serious_score: float) -> None:
    """Warn if backtests show the novelty oracle beating serious models."""

    if octopus_score > serious_score:
        warnings.warn(
            "Your serious model is not beating the octopus. Check calibration and assumptions.",
            stacklevel=2,
        )
