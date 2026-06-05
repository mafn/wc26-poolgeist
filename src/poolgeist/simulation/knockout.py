"""Knockout simulation with extra time and penalties."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from poolgeist.models.base import MatchModel
from poolgeist.models.poisson import PoissonGoalsModel
from poolgeist.schemas import ModelSignal
from poolgeist.simulation.match import MatchResult, simulate_score
from poolgeist.simulation.penalties import shootout_win_probability, simulate_shootout


@dataclass(frozen=True)
class KnockoutResult(MatchResult):
    """Knockout result including advancement method."""

    advanced_team: str = ""
    decided_by: str = "normal_time"
    extra_time_home_goals: int = 0
    extra_time_away_goals: int = 0


def choose_winner(
    home_team: str,
    away_team: str,
    signal: ModelSignal,
    *,
    generator: np.random.Generator | None = None,
) -> str:
    """Choose a knockout winner, splitting draw probability into penalties."""

    gen = generator or np.random.default_rng(2026)
    home_probability = float(
        np.clip(signal.tendency_probs["home"] + signal.tendency_probs["draw"] / 2.0, 0.0, 1.0)
    )
    return str(gen.choice([home_team, away_team], p=[home_probability, 1.0 - home_probability]))


def simulate_knockout_match(
    home_team: str,
    away_team: str,
    model: MatchModel,
    *,
    generator: np.random.Generator | None = None,
) -> KnockoutResult:
    """Simulate 90 minutes, reduced-xG extra time, then penalties if still tied."""

    gen = generator or np.random.default_rng(2026)
    signal = model.predict_match(home_team, away_team)
    home_goals, away_goals = simulate_score(signal, gen)
    if home_goals > away_goals:
        return KnockoutResult(home_team, away_team, home_goals, away_goals, home_team, home_team)
    if away_goals > home_goals:
        return KnockoutResult(home_team, away_team, home_goals, away_goals, away_team, away_team)
    et_signal = PoissonGoalsModel(home_xg=0.35, away_xg=0.35, max_goals=3).predict_match(
        home_team, away_team
    )
    et_home, et_away = simulate_score(et_signal, gen)
    if et_home > et_away:
        return KnockoutResult(
            home_team,
            away_team,
            home_goals,
            away_goals,
            home_team,
            home_team,
            "extra_time",
            et_home,
            et_away,
        )
    if et_away > et_home:
        return KnockoutResult(
            home_team,
            away_team,
            home_goals,
            away_goals,
            away_team,
            away_team,
            "extra_time",
            et_home,
            et_away,
        )
    advanced = simulate_shootout(
        home_team, away_team, gen, home_probability=shootout_win_probability()
    )
    return KnockoutResult(
        home_team,
        away_team,
        home_goals,
        away_goals,
        advanced,
        advanced,
        "penalties",
        et_home,
        et_away,
    )
