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
    return (
        frame.sort_values(
            ["points", "goal_difference", "goals_for", "team"],
            ascending=[False, False, False, True],
        )
        .head(n_qualifiers)
        .reset_index(drop=True)
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
