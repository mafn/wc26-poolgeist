"""Neutral Kicktipp-style scoring for exact football scores."""

from __future__ import annotations

from dataclasses import dataclass

from poolgeist.config import ScoringConfig


@dataclass(frozen=True)
class ScoreRules:
    """Backward-compatible scoring alias."""

    exact: int = 4
    goal_difference: int = 3
    tendency: int = 2
    goals_one_team: int = 0
    goals_both_teams: int = 0


def outcome(home_goals: int, away_goals: int) -> int:
    """Return 1 for home win, 0 for draw, -1 for away win."""

    return (home_goals > away_goals) - (home_goals < away_goals)


def tip_points(
    pred_home_goals: int,
    pred_away_goals: int,
    actual_home_goals: int,
    actual_away_goals: int,
    scoring_config: ScoringConfig | None = None,
) -> int:
    """Score a prediction using neutral Kicktipp-style 9er rules."""

    config = scoring_config or ScoringConfig()
    values = [pred_home_goals, pred_away_goals, actual_home_goals, actual_away_goals]
    if any(not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("Goals must be non-negative integers.")
    if (pred_home_goals, pred_away_goals) == (actual_home_goals, actual_away_goals):
        return (
            config.exact_draw_score
            if actual_home_goals == actual_away_goals
            else config.exact_winning_score
        )
    predicted_outcome = outcome(pred_home_goals, pred_away_goals)
    actual_outcome = outcome(actual_home_goals, actual_away_goals)
    if predicted_outcome != actual_outcome:
        return config.wrong_tendency
    if actual_outcome == 0:
        return config.correct_draw_tendency
    if pred_home_goals - pred_away_goals == actual_home_goals - actual_away_goals:
        return config.correct_goal_difference
    return config.correct_winner_only


def score_prediction(
    predicted_home: int,
    predicted_away: int,
    actual_home: int,
    actual_away: int,
    rules: ScoreRules | None = None,
) -> int:
    """Backward-compatible wrapper around :func:`tip_points`."""

    if rules is None:
        return tip_points(predicted_home, predicted_away, actual_home, actual_away)
    config = ScoringConfig(
        exact_winning_score=rules.exact,
        exact_draw_score=rules.exact,
        correct_goal_difference=rules.goal_difference,
        correct_draw_tendency=max(3, rules.goal_difference)
        if rules.tendency == 2
        else rules.tendency,
        correct_winner_only=rules.tendency,
    )
    return tip_points(predicted_home, predicted_away, actual_home, actual_away, config)
