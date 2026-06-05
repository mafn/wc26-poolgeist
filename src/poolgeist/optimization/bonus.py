"""Bonus-card optimization helpers."""

from __future__ import annotations

import pandas as pd


def bonus_recommendations_from_probabilities(probabilities: pd.DataFrame) -> pd.DataFrame:
    """Rank bonus options generated from model output, not defaults."""

    if probabilities.empty:
        return pd.DataFrame(columns=["bonus_type", "team", "probability", "warning"])
    return probabilities.sort_values("probability", ascending=False).reset_index(drop=True)
