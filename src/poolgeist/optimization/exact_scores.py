"""Exact-score expected-value optimization."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from poolgeist.config import ScoringConfig, StrategyConfig
from poolgeist.optimization.popularity import estimate_pick_popularity, strategic_value
from poolgeist.schemas import ModelSignal
from poolgeist.scoring import outcome, tip_points


def candidate_scores(
    custom_candidates: Iterable[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Generate default exact-score candidates plus configurable custom candidates."""

    scores = {(home, away) for home in range(6) for away in range(6)}
    scores.update({(6, 0), (6, 1), (7, 0), (0, 6), (1, 6), (0, 7)})
    if custom_candidates:
        scores.update(custom_candidates)
    return sorted(scores)


def expected_points_for_score(
    pred_home_goals: int,
    pred_away_goals: int,
    signal: ModelSignal,
    scoring_config: ScoringConfig,
) -> float:
    """Compute expected office-pool points for a candidate exact score."""

    value = 0.0
    for actual_home in range(signal.score_matrix.shape[0]):
        for actual_away in range(signal.score_matrix.shape[1]):
            value += float(signal.score_matrix[actual_home, actual_away]) * tip_points(
                pred_home_goals, pred_away_goals, actual_home, actual_away, scoring_config
            )
    return value


def candidate_score_ev_table(
    signal: ModelSignal,
    scoring_config: ScoringConfig,
    strategy_config: StrategyConfig,
    chaos_index_value: float,
    disagreement_index: float,
) -> pd.DataFrame:
    """Return EV, popularity, uniqueness, strategic value, and chaos fields for candidates."""

    rows = []
    max_home, max_away = signal.score_matrix.shape
    for home, away in candidate_scores():
        exact = (
            float(signal.score_matrix[home, away]) if home < max_home and away < max_away else 0.0
        )
        tendency = signal.tendency_probs[{1: "home", 0: "draw", -1: "away"}[outcome(home, away)]]
        diff = signal.goal_diff_distribution.get(home - away, 0.0)
        ev = expected_points_for_score(home, away, signal, scoring_config)
        popularity = estimate_pick_popularity(
            home,
            away,
            favourite_strength=max(signal.tendency_probs["home"], signal.tendency_probs["away"]),
        )
        strategic = strategic_value(
            ev,
            popularity,
            lambda_uniqueness=strategy_config.lambda_uniqueness,
            lambda_chaos=strategy_config.lambda_chaos,
            chaos_index_value=chaos_index_value,
            lambda_variance=strategy_config.lambda_variance,
            downside_risk=max(0.0, 1.0 - ev / 4.0),
        )
        rows.append(
            {
                "pred_home_goals": home,
                "pred_away_goals": away,
                "exact_score_probability": exact,
                "tendency_probability": tendency,
                "goal_difference_probability": diff,
                "expected_points": ev,
                "estimated_popularity": popularity,
                "uniqueness": 1.0 - popularity,
                "strategic_value": strategic,
                "chaos_index": chaos_index_value,
                "disagreement_index": disagreement_index,
                "chaos_value": strategic + chaos_index_value,
                "disagreement_value": strategic + disagreement_index,
            }
        )
    return pd.DataFrame(rows)
