"""Lightweight domain schemas used by simulations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Team:
    """A neutral team record."""

    team_id: str
    name: str
    group: str | None = None


@dataclass(frozen=True)
class Fixture:
    """A scheduled match between two teams."""

    match_id: str
    home_team: str
    away_team: str
    stage: str = "group"
