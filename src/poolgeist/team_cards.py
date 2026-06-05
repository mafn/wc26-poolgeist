"""Neutral team-card helpers for demo inputs."""

from __future__ import annotations

import pandas as pd

REQUIRED_TEAM_CARD_COLUMNS = ["team_id", "team_name", "neutral_strength", "notes"]


def validate_team_cards(cards: pd.DataFrame) -> pd.DataFrame:
    """Validate and return team cards with neutral strength clipped to [0, 1]."""

    missing = set(REQUIRED_TEAM_CARD_COLUMNS) - set(cards.columns)
    if missing:
        msg = f"Missing team-card columns: {sorted(missing)}"
        raise ValueError(msg)
    result = cards.copy()
    result["neutral_strength"] = result["neutral_strength"].clip(0.0, 1.0)
    return result
