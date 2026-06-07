"""Base model protocols and signal construction helpers."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from scipy.stats import poisson

from poolgeist.schemas import ModelSignal
from poolgeist.utils.probability import normalize


def matrix_to_signal(
    score_matrix: np.ndarray,
    *,
    model_name: str,
    model_weight: float,
    home_team: str,
    away_team: str,
    explanations: list[str] | None = None,
    warnings: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ModelSignal:
    """Convert a valid score matrix into a full :class:`ModelSignal`."""

    matrix = np.asarray(score_matrix, dtype=float)
    if np.isnan(matrix).any() or not np.isfinite(matrix).all() or (matrix < -1e-9).any():
        raise ValueError("score_matrix must contain finite non-negative probabilities")
    matrix = np.maximum(matrix, 0.0)
    total = float(matrix.sum())
    if total <= 0:
        raise ValueError("score_matrix must contain positive mass")
    matrix = matrix / total
    rows, cols = matrix.shape
    home_goals = np.arange(rows)[:, None]
    away_goals = np.arange(cols)[None, :]
    home_prob = float(matrix[home_goals > away_goals].sum())
    draw_prob = float(np.trace(matrix))
    away_prob = float(matrix[home_goals < away_goals].sum())
    total_goals = home_goals + away_goals
    diffs = home_goals - away_goals
    goal_diff_distribution = {
        int(diff): float(matrix[diffs == diff].sum()) for diff in range(-(cols - 1), rows)
    }
    tendency = normalize({"home": home_prob, "draw": draw_prob, "away": away_prob})
    favourite_clean_sheet = max(float(matrix[:, 0].sum()), float(matrix[0, :].sum()))
    entropy_denominator = float(np.log(matrix.size))
    entropy = (
        0.0
        if entropy_denominator <= 0
        else -float(np.sum(matrix * np.log(np.clip(matrix, 1e-12, 1.0)))) / entropy_denominator
    )
    return ModelSignal(
        model_name=model_name,
        model_weight=model_weight,
        home_team=home_team,
        away_team=away_team,
        expected_goals_home=float((matrix * home_goals).sum()),
        expected_goals_away=float((matrix * away_goals).sum()),
        score_matrix=matrix,
        tendency_probs=tendency,
        btts_prob=float(matrix[1:, 1:].sum()),
        over_2_5_prob=float(matrix[total_goals > 2.5].sum()),
        over_3_5_prob=float(matrix[total_goals > 3.5].sum()),
        draw_prob=float(draw_prob),
        upset_prob=float(min(home_prob, away_prob)),
        clean_sheet_home_prob=float(matrix[:, 0].sum()),
        clean_sheet_away_prob=float(matrix[0, :].sum()),
        goal_diff_distribution=goal_diff_distribution,
        uncertainty=entropy,
        explanations=explanations or [],
        warnings=warnings or [],
        metadata={"favourite_clean_sheet_probability": favourite_clean_sheet, **(metadata or {})},
    )


class MatchModel(Protocol):
    """Protocol implemented by match probability models."""

    default_weight: float

    def predict_match(self, home_team: str, away_team: str) -> ModelSignal:
        """Predict a single match."""


def adjust_xg_with_modifiers(
    home_team: str,
    away_team: str,
    base_home_xg: float,
    base_away_xg: float,
    team_modifiers: dict[str, dict[str, Any]] | None,
) -> tuple[float, float]:
    """Adjust expected goals dynamically using team modifiers."""

    if not team_modifiers:
        return base_home_xg, base_away_xg

    home_mods = team_modifiers.get(home_team, {})
    away_mods = team_modifiers.get(away_team, {})

    home_attack = _modifier(home_mods, "attack_modifier")
    away_defense = _modifier(away_mods, "defense_modifier")
    home_rating = _strength_modifier(home_mods)
    away_rating = _strength_modifier(away_mods)
    tempo = 0.5 * (_modifier(home_mods, "tempo_modifier") + _modifier(away_mods, "tempo_modifier"))

    adjusted_home_xg = max(
        0.01,
        base_home_xg + home_attack + away_defense + 0.18 * (home_rating - away_rating) + tempo,
    )

    away_attack = _modifier(away_mods, "attack_modifier")
    home_defense = _modifier(home_mods, "defense_modifier")
    adjusted_away_xg = max(
        0.01,
        base_away_xg + away_attack + home_defense + 0.18 * (away_rating - home_rating) + tempo,
    )

    return adjusted_home_xg, adjusted_away_xg


def poisson_goal_probs(expected_goals: float, max_goals: int) -> np.ndarray:
    """Return an exact finite Poisson grid with tail mass folded into the final bin."""

    if expected_goals <= 0:
        raise ValueError("expected_goals must be positive")
    if max_goals < 0:
        raise ValueError("max_goals must be non-negative")

    goals = np.arange(max_goals + 1)
    probabilities = poisson.pmf(goals, expected_goals)
    if max_goals > 0:
        probabilities[-1] = poisson.sf(max_goals - 1, expected_goals)
    return probabilities / probabilities.sum()


def independent_poisson_matrix(
    home_xg: float,
    away_xg: float,
    max_goals: int,
) -> np.ndarray:
    """Return an exact independent score matrix for finite Monte Carlo grids."""

    return np.outer(poisson_goal_probs(home_xg, max_goals), poisson_goal_probs(away_xg, max_goals))


def matchup_modifiers(
    home_team: str,
    away_team: str,
    team_modifiers: dict[str, dict[str, Any]] | None,
) -> dict[str, float]:
    """Extract bounded tactical modifiers used by chaos-side models."""

    if not team_modifiers:
        return {
            "home_strength": 0.0,
            "away_strength": 0.0,
            "strength_gap": 0.0,
            "tempo": 0.0,
            "chaos": 0.0,
            "home_chaos": 0.0,
            "away_chaos": 0.0,
        }

    home_mods = team_modifiers.get(home_team, {})
    away_mods = team_modifiers.get(away_team, {})
    home_strength = (
        _modifier(home_mods, "attack_modifier")
        - _modifier(home_mods, "defense_modifier")
        + _strength_modifier(home_mods)
    )
    away_strength = (
        _modifier(away_mods, "attack_modifier")
        - _modifier(away_mods, "defense_modifier")
        + _strength_modifier(away_mods)
    )
    home_chaos = _modifier(home_mods, "chaos_modifier")
    away_chaos = _modifier(away_mods, "chaos_modifier")
    return {
        "home_strength": float(home_strength),
        "away_strength": float(away_strength),
        "strength_gap": float(home_strength - away_strength),
        "tempo": float(
            0.5 * (_modifier(home_mods, "tempo_modifier") + _modifier(away_mods, "tempo_modifier"))
        ),
        "chaos": float(0.5 * (home_chaos + away_chaos)),
        "home_chaos": float(home_chaos),
        "away_chaos": float(away_chaos),
    }


def _modifier(modifiers: dict[str, Any], key: str) -> float:
    value = modifiers.get(key, 0.0)
    if isinstance(value, bool):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _strength_modifier(modifiers: dict[str, Any]) -> float:
    if "strength_modifier" in modifiers:
        return _modifier(modifiers, "strength_modifier")
    if "rating_modifier" in modifiers:
        return _modifier(modifiers, "rating_modifier")
    if "base_rating" in modifiers:
        return _modifier(modifiers, "base_rating") - 0.5
    if "rating" in modifiers:
        return _modifier(modifiers, "rating") - 0.5
    return 0.0
