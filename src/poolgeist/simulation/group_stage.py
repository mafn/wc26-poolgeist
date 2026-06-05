"""Minimal group-stage simulation primitives."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

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
            fixture.home_team,
            fixture.away_team,
            model.predict_match(fixture.home_team, fixture.away_team),
            generator,
        )
        for fixture in fixtures
    ]
