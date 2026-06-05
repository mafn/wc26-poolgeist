"""Chaos-measure helpers for pool strategy analysis."""

from __future__ import annotations

import numpy as np


def entropy(probabilities: list[float]) -> float:
    """Compute Shannon entropy for a probability vector."""

    values = np.asarray(probabilities, dtype=float)
    if values.size == 0:
        return 0.0
    total = values.sum()
    if total <= 0:
        raise ValueError("Sum of probabilities must be positive.")
    values = values / total
    values = values[values > 0]
    return float(-(values * np.log2(values)).sum())
