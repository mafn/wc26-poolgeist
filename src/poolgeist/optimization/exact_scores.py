"""Exact-score recommendation helpers."""

from __future__ import annotations

from scipy.stats import poisson


def most_likely_neutral_score(
    home_xg: float = 1.35, away_xg: float = 1.15, max_goals: int = 6
) -> tuple[int, int]:
    """Return the most likely score under independent Poisson goal assumptions."""

    best_score = (0, 0)
    best_probability = -1.0
    for home in range(max_goals + 1):
        for away in range(max_goals + 1):
            probability = float(poisson.pmf(home, home_xg) * poisson.pmf(away, away_xg))
            if probability > best_probability:
                best_probability = probability
                best_score = (home, away)
    return best_score
