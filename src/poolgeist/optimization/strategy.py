"""High-level strategy helpers."""

from __future__ import annotations

import pandas as pd


def select_recommendations(score_ev_table: pd.DataFrame) -> dict[str, str]:
    """Select recommendation labels from an EV table without hard-coded picks."""

    def top(column: str) -> str:
        row = score_ev_table.sort_values(column, ascending=False).iloc[0]
        return f"{int(row.pred_home_goals)}-{int(row.pred_away_goals)}"

    return {
        "safest": top("exact_score_probability"),
        "highest_raw_ev": top("expected_points"),
        "highest_strategic_value": top("strategic_value"),
        "highest_chaos_value": top("chaos_value"),
    }
