"""Neutral team-card helpers for demo inputs."""

from __future__ import annotations

import pandas as pd

REQUIRED_TEAM_CARD_COLUMNS = ["team_id", "team_name", "neutral_strength", "notes"]


def validate_team_cards(cards: pd.DataFrame) -> pd.DataFrame:
    """Validate and return team cards with neutral strength clipped to [0, 1]."""

    from poolgeist.utils.validation import require_columns

    require_columns(set(cards.columns), set(REQUIRED_TEAM_CARD_COLUMNS))
    result = cards.copy()
    result["neutral_strength"] = pd.to_numeric(result["neutral_strength"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    return result
