"""Chaos-index calculation and classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from poolgeist.schemas import ModelSignal


@dataclass(frozen=True)
class ChaosWeights:
    """Weights for the configurable chaos-index formula."""

    w_entropy: float = 0.18
    w_tendency_entropy: float = 0.14
    w_upset: float = 0.14
    w_draw: float = 0.10
    w_btts: float = 0.08
    w_over35: float = 0.08
    w_disagreement: float = 0.14
    w_spin_flip: float = 0.10
    w_clean_sheet: float = 0.08


def chaos_index(
    signal: ModelSignal,
    *,
    model_disagreement: float = 0.0,
    flip_probability: float | None = None,
    weights: ChaosWeights | None = None,
) -> float:
    """Compute a bounded chaos index from uncertainty, upset, draw, and volatility signals."""

    w = weights or ChaosWeights()
    matrix = signal.score_matrix
    score_entropy_denominator = float(np.log(matrix.size))
    score_entropy = (
        0.0
        if score_entropy_denominator <= 0
        else -float(np.sum(matrix * np.log(np.clip(matrix, 1e-12, 1.0))))
        / score_entropy_denominator
    )
    tendencies = np.array(list(signal.tendency_probs.values()), dtype=float)
    tendency_entropy = -float(np.sum(tendencies * np.log(np.clip(tendencies, 1e-12, 1.0)))) / float(
        np.log(3)
    )
    spin = (
        signal.metadata.get("flip_probability", 0.0)
        if flip_probability is None
        else flip_probability
    )
    clean = signal.metadata.get(
        "favourite_clean_sheet_probability",
        max(signal.clean_sheet_home_prob, signal.clean_sheet_away_prob),
    )
    value = (
        w.w_entropy * score_entropy
        + w.w_tendency_entropy * tendency_entropy
        + w.w_upset * signal.upset_prob
        + w.w_draw * signal.draw_prob
        + w.w_btts * signal.btts_prob
        + w.w_over35 * signal.over_3_5_prob
        + w.w_disagreement * min(model_disagreement, 1.0)
        + w.w_spin_flip * float(spin)
        - w.w_clean_sheet * float(clean)
    )
    return float(np.clip(value, 0.0, 1.0))


def classify_chaos(value: float) -> str:
    """Classify a computed chaos index."""

    if value < 0.22:
        return "anchor"
    if value < 0.42:
        return "moderate_upside"
    if value < 0.68:
        return "chaos"
    return "avoid_chaos"
