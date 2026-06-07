"""Exact-score expected-value optimization."""

from __future__ import annotations

import functools
from collections import Counter
from collections.abc import Iterable

import numpy as np
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
    # Add common high-scoring shootout outcomes (penalties, sudden death, lopsided scores)

    scores.update(
        {
            (6, 5),
            (5, 6),
            (7, 6),
            (6, 7),
            (8, 7),
            (7, 8),
            (6, 2),
            (2, 6),
            (6, 3),
            (3, 6),
            (6, 4),
            (4, 6),
            (7, 3),
            (3, 7),
            (7, 4),
            (4, 7),
            (7, 5),
            (5, 7),
            (8, 5),
            (5, 8),
            (8, 6),
            (6, 8),
        }
    )
    if custom_candidates:
        scores.update(custom_candidates)
    return sorted(scores)


@functools.lru_cache(maxsize=1)
def _get_static_shootout_probs() -> tuple[
    dict[tuple[int, int], float], dict[tuple[int, int], float]
]:
    """Pre-calculate static shootout score distributions for home and away wins."""
    from poolgeist.simulation.penalties import simulate_shootout_match_score

    rng = np.random.default_rng(2026)
    home_counts = Counter()
    away_counts = Counter()
    n_sims = 5000
    for _ in range(n_sims):
        home_counts[simulate_shootout_match_score(True, rng)] += 1
        away_counts[simulate_shootout_match_score(False, rng)] += 1

    home_probs = {score: count / n_sims for score, count in home_counts.items()}
    away_probs = {score: count / n_sims for score, count in away_counts.items()}
    return home_probs, away_probs


def _get_shootout_probs(home_prob_win: float) -> dict[tuple[int, int], float]:
    """Get shootout score probabilities by blending pre-calculated static distributions."""
    home_probs, away_probs = _get_static_shootout_probs()
    all_scores = set(home_probs.keys()) | set(away_probs.keys())
    return {
        score: home_prob_win * home_probs.get(score, 0.0)
        + (1.0 - home_prob_win) * away_probs.get(score, 0.0)
        for score in all_scores
    }


@functools.lru_cache(maxsize=1)
def _get_et_matrix() -> np.ndarray:
    """Pre-calculate the static extra-time goals distribution matrix."""
    from poolgeist.models.poisson import PoissonGoalsModel

    return (
        PoissonGoalsModel(home_xg=0.35, away_xg=0.35, max_goals=3)
        .predict_match("home", "away")
        .score_matrix
    )


def adjust_matrix_for_knockout(
    score_matrix: np.ndarray,
    home_prob_win: float = 0.5,
) -> np.ndarray:
    """Adjust a 90-minute score matrix for extra time and penalty shootouts."""
    et_matrix = _get_et_matrix()

    shootout_probs = _get_shootout_probs(home_prob_win)

    # New score matrix size: maximum possible goals is 90min max + ET max + shootout max
    max_90_goals = score_matrix.shape[0] - 1
    max_et_goals = et_matrix.shape[0] - 1
    max_shootout_goals = max(max(h, a) for h, a in shootout_probs)
    max_final_goals = max_90_goals + max_et_goals + max_shootout_goals

    ko_matrix = np.zeros((max_final_goals + 1, max_final_goals + 1), dtype=float)

    for h in range(score_matrix.shape[0]):
        for a in range(score_matrix.shape[1]):
            p_90 = score_matrix[h, a]
            if p_90 == 0:
                continue

            if h != a:
                # Decided in normal time
                ko_matrix[h, a] += p_90
            else:
                # Tied, goes to extra time
                for eth in range(et_matrix.shape[0]):
                    for eta in range(et_matrix.shape[1]):
                        p_et = et_matrix[eth, eta]
                        p_total = p_90 * p_et
                        if p_total == 0:
                            continue

                        if eth != eta:
                            # Decided in extra time
                            ko_matrix[h + eth, a + eta] += p_total
                        else:
                            # Still tied, goes to penalties
                            base_h = h + eth
                            base_a = a + eta
                            for (pen_h, pen_a), p_pen in shootout_probs.items():
                                ko_matrix[base_h + pen_h, base_a + pen_a] += p_total * p_pen

    # Normalize
    total = ko_matrix.sum()
    if total > 0:
        ko_matrix /= total

    return ko_matrix


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
