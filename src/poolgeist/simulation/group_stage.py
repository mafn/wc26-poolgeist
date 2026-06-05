"""Group-stage simulation and ranking."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd

from poolgeist.models.base import MatchModel
from poolgeist.schemas import Fixture
from poolgeist.simulation.match import MatchResult, simulate_outcome


def simulate_group_fixtures(
    fixtures: Iterable[Fixture], model: MatchModel, *, seed: int | None = None
) -> list[MatchResult]:
    """Simulate a collection of group fixtures."""

    generator = np.random.default_rng(seed)
    return [
        simulate_outcome(
            f.home_team, f.away_team, model.predict_match(f.home_team, f.away_team), generator
        )
        for f in fixtures
    ]


def rank_group(
    results: Iterable[MatchResult], *, ratings: dict[str, float] | None = None
) -> pd.DataFrame:
    """Rank a group by points, goal difference, goals for, then rating/name fallback."""

    table = defaultdict(
        lambda: {"team": "", "points": 0, "goal_difference": 0, "goals_for": 0, "goals_against": 0}
    )
    for result in results:
        for team in (result.home_team, result.away_team):
            table[team]["team"] = team
        hg, ag = result.home_goals, result.away_goals
        table[result.home_team]["goals_for"] += hg
        table[result.home_team]["goals_against"] += ag
        table[result.away_team]["goals_for"] += ag
        table[result.away_team]["goals_against"] += hg
        table[result.home_team]["goal_difference"] += hg - ag
        table[result.away_team]["goal_difference"] += ag - hg
        if hg > ag:
            table[result.home_team]["points"] += 3
        elif ag > hg:
            table[result.away_team]["points"] += 3
        else:
            table[result.home_team]["points"] += 1
            table[result.away_team]["points"] += 1
    frame = pd.DataFrame(table.values())
    if frame.empty:
        return frame
    frame["rating_tiebreaker"] = frame["team"].map(ratings or {}).fillna(0.0)
    return frame.sort_values(
        ["points", "goal_difference", "goals_for", "rating_tiebreaker", "team"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
