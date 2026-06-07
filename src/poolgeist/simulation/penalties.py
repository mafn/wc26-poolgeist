"""Penalty shootout model."""

from __future__ import annotations

import numpy as np


def shootout_win_probability(
    penalty_skill: float = 0.5,
    goalkeeper_quality: float = 0.5,
    mentality: float = 0.5,
    fatigue: float = 0.5,
    pressure: float = 0.5,
    manual_modifier: float = 0.0,
    *,
    allow_strong_modifier: bool = False,
) -> float:
    """Estimate shootout win probability, clamped unless explicitly overridden."""

    probability = (
        0.5
        + 0.08 * (penalty_skill - 0.5)
        + 0.06 * (goalkeeper_quality - 0.5)
        + 0.04 * (mentality - 0.5)
        - 0.04 * (fatigue - 0.5)
        - 0.03 * (pressure - 0.5)
        + manual_modifier
    )
    lo, hi = (0.30, 0.70) if allow_strong_modifier else (0.38, 0.62)
    return float(np.clip(probability, lo, hi))


def simulate_shootout_match_score(
    winner_is_home: bool, generator: np.random.Generator
) -> tuple[int, int]:
    """Simulate a penalty shootout kick-by-kick until a winner is decided.

    Uses rejection sampling to ensure the simulated score aligns with the predetermined winner.
    """
    while True:
        home_kicks = []
        away_kicks = []
        # First 5 rounds
        for _ in range(5):
            # Home kicks
            home_kicks.append(generator.random() < 0.75)
            home_score = sum(home_kicks)
            away_score = sum(away_kicks)
            home_remaining = 5 - len(home_kicks)
            away_remaining = 5 - len(away_kicks)
            if home_score > away_score + away_remaining:
                break
            if away_score > home_score + home_remaining:
                break

            # Away kicks
            away_kicks.append(generator.random() < 0.75)
            home_score = sum(home_kicks)
            away_score = sum(away_kicks)
            home_remaining = 5 - len(home_kicks)
            away_remaining = 5 - len(away_kicks)
            if home_score > away_score + away_remaining:
                break
            if away_score > home_score + home_remaining:
                break
        else:
            home_score = sum(home_kicks)
            away_score = sum(away_kicks)
            if home_score == away_score:
                while True:
                    home_ok = generator.random() < 0.75
                    away_ok = generator.random() < 0.75
                    if home_ok:
                        home_score += 1
                    if away_ok:
                        away_score += 1
                    if home_ok != away_ok:
                        break

        simulated_home_wins = home_score > away_score
        if simulated_home_wins == winner_is_home:
            return home_score, away_score


def simulate_shootout(
    home_team: str, away_team: str, generator: np.random.Generator, *, home_probability: float = 0.5
) -> tuple[str, tuple[int, int]]:
    """Simulate advancement by penalties; shootout goals are not normal goals."""

    winner = str(
        generator.choice([home_team, away_team], p=[home_probability, 1.0 - home_probability])
    )
    winner_is_home = winner == home_team
    pen_score = simulate_shootout_match_score(winner_is_home, generator)
    return winner, pen_score
