"""Domain schemas used by Poolgeist simulations, news, and reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from poolgeist.utils.probability import assert_probability_vector


@dataclass(frozen=True)
class Team:
    """A neutral team record."""

    team_id: str
    name: str
    group: str | None = None
    rating: float = 0.5


@dataclass(frozen=True)
class Fixture:
    """A scheduled match between two teams."""

    match_id: str
    home_team: str
    away_team: str
    stage: str = "group"
    group: str | None = None


@dataclass(frozen=True)
class TendencyProbabilities:
    """Home/draw/away tendency probabilities."""

    home: float
    draw: float
    away: float

    def as_dict(self) -> dict[str, float]:
        """Return probabilities as a dictionary."""

        return {"home": self.home, "draw": self.draw, "away": self.away}


@dataclass(frozen=True)
class ModelSignal:
    """Full model-council signal for one match."""

    model_name: str
    model_weight: float
    home_team: str
    away_team: str
    expected_goals_home: float
    expected_goals_away: float
    score_matrix: np.ndarray
    tendency_probs: dict[str, float]
    btts_prob: float
    over_2_5_prob: float
    over_3_5_prob: float
    draw_prob: float
    upset_prob: float
    clean_sheet_home_prob: float
    clean_sheet_away_prob: float
    goal_diff_distribution: dict[int, float]
    uncertainty: float
    explanations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        matrix = np.asarray(self.score_matrix, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("score_matrix must be two-dimensional")
        assert_probability_vector(matrix.ravel())
        assert_probability_vector(self.tendency_probs.values())
        if self.model_weight < 0:
            raise ValueError("model_weight must be non-negative")
