"""Spin-flip volatility model using Glauber dynamics for match-state transitions.

Maps a football match onto a lattice of minute-by-minute "control spins":
+1 = favourite in control, −1 = underdog in control (chaos / upset state).

Each spin can flip according to a Glauber acceptance probability determined
by the local field (team quality, tactical features) and a temperature
parameter (match volatility).  After running a full Monte Carlo sweep, the
magnetization profile quantifies how often control is lost — directly
translating into upset probability, late-goal surges, and volatility
modifiers applied to an underlying Poisson score matrix.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from poolgeist.models.base import (
    adjust_xg_with_modifiers,
    independent_poisson_matrix,
    matchup_modifiers,
    matrix_to_signal,
)
from poolgeist.schemas import ModelSignal


@dataclass(frozen=True)
class SpinFlipOutput:
    """Volatility modifiers estimated by the spin-flip Glauber dynamics."""

    flip_probability: float
    volatility_modifier: float
    upset_modifier: float
    late_goal_modifier: float
    magnetization: float
    chaos_fraction: float


class SpinFlipVolatilityModel:
    """Estimate match volatility via Glauber spin-flip dynamics on a 90-spin lattice."""

    default_weight = 0.01

    def __init__(
        self,
        *,
        volatility: float = 0.2,
        early_goal_sensitivity: float = 0.2,
        red_card_vulnerability: float = 0.1,
        discipline_risk: float = 0.1,
        high_line_vulnerability: float = 0.1,
        underdog_transition_threat: float = 0.1,
        crowd_host_pressure: float = 0.0,
        knockout_pressure: float = 0.0,
        home_xg: float = 1.35,
        away_xg: float = 1.15,
        max_goals: int = 10,
        n_spins: int = 90,
        mc_sweeps: int = 8,
        seed: int | None = None,
    ):
        self.features = [
            volatility,
            early_goal_sensitivity,
            red_card_vulnerability,
            discipline_risk,
            high_line_vulnerability,
            underdog_transition_threat,
            crowd_host_pressure,
            knockout_pressure,
        ]
        if any(not 0 <= feature <= 1 for feature in self.features):
            raise ValueError("spin-flip features must be between 0 and 1")
        if home_xg <= 0 or away_xg <= 0:
            raise ValueError("Expected goals must be positive.")
        if n_spins < 1:
            raise ValueError("n_spins must be at least 1")
        if mc_sweeps < 1:
            raise ValueError("mc_sweeps must be at least 1")
        self.home_xg = home_xg
        self.away_xg = away_xg
        self.max_goals = max_goals
        self.n_spins = n_spins
        self.mc_sweeps = mc_sweeps
        self.seed = seed

    def _derive_rng(self, home_team: str, away_team: str) -> np.random.Generator:
        """Derive a deterministic RNG from the matchup and instance seed."""

        key = f"spinflip:{self.seed}:{home_team}:{away_team}"
        derived = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (2**32)
        return np.random.default_rng(derived)

    def estimate(
        self,
        features: list[float] | None = None,
        strength_gap: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> SpinFlipOutput:
        """Run Glauber dynamics and return volatility modifiers.

        Parameters
        ----------
        features:
            Override feature vector (default: self.features).
        strength_gap:
            Signed quality difference (home − away); acts as external field.
        rng:
            Pre-seeded generator for reproducibility.
        """

        feats = features if features is not None else self.features
        if rng is None:
            rng = np.random.default_rng(42)

        # Mean feature score → base temperature.  Higher features ⇒ hotter
        # (more volatile), making spin flips more likely.
        mean_feat = float(np.mean(feats))
        temperature = max(0.15, 0.5 + 1.5 * mean_feat)

        # External field: positive favours +1 (home/favourite stays in control).
        h_field = float(np.clip(strength_gap * 0.6, -2.0, 2.0))

        # Nearest-neighbour coupling: consecutive minutes influence each other.
        coupling = float(np.clip(0.3 + 0.5 * (1.0 - mean_feat), 0.1, 0.9))

        # Initialise lattice: favourite starts in control (+1).
        spins = np.ones(self.n_spins, dtype=float)
        beta = 1.0 / temperature

        # Glauber dynamics: sweep the lattice multiple times.
        for _ in range(self.mc_sweeps):
            for i in rng.permutation(self.n_spins):
                # Local field from neighbours (1D periodic boundary).
                left = spins[(i - 1) % self.n_spins]
                right = spins[(i + 1) % self.n_spins]
                local_field = coupling * (left + right) + h_field

                # Glauber acceptance: P(spin_i = +1).
                p_up = 1.0 / (1.0 + np.exp(-2.0 * beta * local_field))
                spins[i] = 1.0 if rng.random() < p_up else -1.0

        # Analyse the final configuration.
        magnetization = float(np.mean(spins))
        chaos_fraction = float(np.mean(spins < 0))

        # Derive modifiers.
        flip_prob = float(np.clip(0.08 + 0.62 * chaos_fraction, 0.02, 0.70))
        volatility_mod = 1.0 + flip_prob
        upset_mod = float(np.clip(0.5 * chaos_fraction + 0.15 * mean_feat, 0.0, 0.8))

        # Late-goal modifier: second-half spins (minute 45–90) losing control.
        second_half = spins[self.n_spins // 2 :]
        late_chaos = float(np.mean(second_half < 0))
        late_goal_mod = float(np.clip(0.25 * late_chaos + 0.10 * mean_feat, 0.0, 0.5))

        return SpinFlipOutput(
            flip_probability=flip_prob,
            volatility_modifier=volatility_mod,
            upset_modifier=upset_mod,
            late_goal_modifier=late_goal_mod,
            magnetization=magnetization,
            chaos_fraction=chaos_fraction,
        )

    def predict_match(self, home_team: str, away_team: str) -> ModelSignal:
        """Return a volatility-adjusted score signal driven by Glauber spin dynamics."""

        rng = self._derive_rng(home_team, away_team)

        home_xg, away_xg = adjust_xg_with_modifiers(
            home_team,
            away_team,
            self.home_xg,
            self.away_xg,
            getattr(self, "team_modifiers", None),
        )
        mods = matchup_modifiers(home_team, away_team, getattr(self, "team_modifiers", None))

        # Inject chaos modifier from team cards into the volatility feature.
        feats = list(self.features)
        feats[0] = float(np.clip(feats[0] + mods["chaos"], 0.0, 1.0))

        out = self.estimate(features=feats, strength_gap=mods["strength_gap"], rng=rng)

        # Build score matrix: start from calibrated Poisson, then warp.
        matrix = independent_poisson_matrix(home_xg, away_xg, self.max_goals)
        goals = np.arange(self.max_goals + 1)
        total = goals[:, None] + goals[None, :]
        diff = goals[:, None] - goals[None, :]

        # Volatility boost for high-scoring outcomes.
        high_total = total >= 3
        matrix[high_total] *= out.volatility_modifier

        # Late-goal tail: extra mass on outcomes with many total goals.
        late_tail = 1.0 + out.late_goal_modifier * np.clip(total - 4, 0.0, None) / max(
            1.0, self.max_goals
        )
        matrix *= late_tail

        # Upset boost: away wins get extra mass proportional to upset modifier.
        away_winning = diff < 0
        matrix[away_winning] *= 1.0 + out.upset_modifier

        # Draw damping: when chaos is high, draws become less likely.
        draw_mask = diff == 0
        matrix[draw_mask] *= max(0.5, 1.0 - 0.3 * out.chaos_fraction)

        return matrix_to_signal(
            matrix,
            model_name="spin_flip",
            model_weight=self.default_weight,
            home_team=home_team,
            away_team=away_team,
            explanations=[
                "Glauber spin-flip dynamics on a 90-minute lattice estimate "
                "match volatility: each spin represents a minute of favourite "
                "control (+1) or chaos (−1)."
            ],
            metadata={
                "flip_probability": out.flip_probability,
                "volatility_modifier": out.volatility_modifier,
                "upset_modifier": out.upset_modifier,
                "late_goal_modifier": out.late_goal_modifier,
                "magnetization": out.magnetization,
                "chaos_fraction": out.chaos_fraction,
                "home_xg": home_xg,
                "away_xg": away_xg,
                "strength_gap": mods["strength_gap"],
            },
        )
