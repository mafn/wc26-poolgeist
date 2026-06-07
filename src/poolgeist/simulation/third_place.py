"""Third-place qualification helpers."""

from __future__ import annotations

import warnings

import pandas as pd


def best_third_placed(
    group_tables: dict[str, pd.DataFrame], *, n_qualifiers: int = 8
) -> pd.DataFrame:
    """Select best third-placed teams by group ranking table fields."""

    thirds = []
    for group, table in group_tables.items():
        if len(table) >= 3:
            row = table.iloc[2].to_dict()
            row["group"] = group
            thirds.append(row)
    frame = pd.DataFrame(thirds)
    if frame.empty:
        return frame
    sort_cols = ["points", "goal_difference", "goals_for"]
    ascending = [False, False, False]
    if "rating_tiebreaker" in frame.columns:
        sort_cols.append("rating_tiebreaker")
        ascending.append(False)
    sort_cols.append("team")
    ascending.append(True)
    return (
        frame.sort_values(sort_cols, ascending=ascending).head(n_qualifiers).reset_index(drop=True)
    )


def load_third_place_mapping(path: str | None = None) -> pd.DataFrame | None:
    """Load official mapping if provided; warn and return None otherwise."""

    if path is None:
        warnings.warn(
            "No third-place mapping CSV provided; using documented neutral fallback bracket order.",
            stacklevel=2,
        )
        return None
    return pd.read_csv(path)
