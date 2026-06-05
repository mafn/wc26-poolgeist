"""CSV/JSON export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a dataframe, creating parent directories."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return target


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    """Write JSON, creating parent directories."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return target


def export_prediction_bundle(prediction, output_dir: str | Path = "outputs") -> list[Path]:
    """Export the standard public-demo prediction files."""

    out = Path(output_dir)
    paths = [
        write_csv(prediction.score_ev_table, out / "score_ev_by_match.csv"),
        write_csv(prediction.score_ev_table.head(6), out / "recommended_picks.csv"),
        write_json(prediction.signal.metadata, out / "model_weights.json"),
        write_json({"chaos_index": prediction.chaos_index}, out / "simulation_config.json"),
    ]
    empty = pd.DataFrame()
    for name in [
        "group_winner_probabilities.csv",
        "qualification_probabilities.csv",
        "knockout_probabilities.csv",
        "bonus_recommendations.csv",
        "model_ablation_report.csv",
        "team_cards_used.csv",
        "news_signals_used.csv",
    ]:
        paths.append(write_csv(empty, out / name))
    return paths
