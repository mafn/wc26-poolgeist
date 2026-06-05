"""Opponent-pool popularity estimation."""

from __future__ import annotations

import numpy as np


def estimate_pick_popularity(
    pred_home_goals: int,
    pred_away_goals: int,
    *,
    favourite_strength: float = 0.5,
    brand_popularity: float = 0.0,
    public_hype: float = 0.0,
    manual_override: float | None = None,
) -> float:
    """Estimate how common a candidate exact score may be in a neutral office pool."""

    if manual_override is not None:
        return float(np.clip(manual_override, 0.0, 1.0))
    common_scores = {(1, 0), (2, 1), (1, 1), (0, 0), (2, 0), (0, 1), (1, 2)}
    common_bonus = 0.28 if (pred_home_goals, pred_away_goals) in common_scores else 0.08
    favourite_bonus = 0.18 * favourite_strength if pred_home_goals > pred_away_goals else 0.08
    return float(
        np.clip(
            common_bonus + favourite_bonus + 0.12 * brand_popularity + 0.10 * public_hype,
            0.02,
            0.95,
        )
    )


def strategic_value(
    expected_points: float,
    estimated_popularity: float,
    *,
    lambda_uniqueness: float = 0.4,
    lambda_chaos: float = 0.0,
    chaos_index_value: float = 0.0,
    lambda_variance: float = 0.0,
    downside_risk: float = 0.0,
) -> float:
    """Compute risk-adjusted strategic value."""

    uniqueness = 1.0 - estimated_popularity
    return float(
        expected_points
        + lambda_uniqueness * uniqueness
        + lambda_chaos * chaos_index_value
        - lambda_variance * downside_risk
    )
