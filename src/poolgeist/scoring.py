"""Neutral Kicktipp-style scoring for exact football scores.

The defaults are intentionally generic and configurable. They are inspired by common
office-pool rules: exact score, correct goal difference, correct tendency/outcome,
and optional goal-count bonuses.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreRules:
    """Configurable points for an office-pool score prediction."""

    exact: int = 4
    goal_difference: int = 3
    tendency: int = 2
    goals_one_team: int = 1
    goals_both_teams: int = 2


def outcome(home_goals: int, away_goals: int) -> int:
    """Return 1 for home win, 0 for draw, -1 for away win."""

    return (home_goals > away_goals) - (home_goals < away_goals)


def score_prediction(
    predicted_home: int,
    predicted_away: int,
    actual_home: int,
    actual_away: int,
    rules: ScoreRules | None = None,
) -> int:
    """Score a predicted exact score against an actual result."""

    rules = rules or ScoreRules()
    values = [predicted_home, predicted_away, actual_home, actual_away]
    if any(value < 0 for value in values):
        raise ValueError("Goals must be non-negative integers.")

    points = 0
    if (predicted_home, predicted_away) == (actual_home, actual_away):
        points = rules.exact
    else:
        predicted_outcome = outcome(predicted_home, predicted_away)
        actual_outcome = outcome(actual_home, actual_away)
        if predicted_outcome == actual_outcome:
            if predicted_home - predicted_away == actual_home - actual_away:
                points = max(points, rules.goal_difference)
            else:
                points = max(points, rules.tendency)

    points = 0
    predicted_outcome = outcome(predicted_home, predicted_away)
    actual_outcome = outcome(actual_home, actual_away)
    if predicted_outcome == actual_outcome:
        if predicted_home - predicted_away == actual_home - actual_away:
            points = max(points, rules.goal_difference)
        else:
            points = max(points, rules.tendency)

    correct_goal_count = int(predicted_home == actual_home) + int(predicted_away == actual_away)
    if correct_goal_count == 2:
        points += rules.goals_both_teams
    elif correct_goal_count == 1:
        points += rules.goals_one_team
    return points
