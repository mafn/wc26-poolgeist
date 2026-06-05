"""Base model protocols and signal construction helpers."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

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
    entropy = -float(np.sum(matrix * np.log(np.clip(matrix, 1e-12, 1.0)))) / float(
        np.log(matrix.size)
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
