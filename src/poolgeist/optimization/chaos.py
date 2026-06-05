"""Chaos-measure helpers for pool strategy analysis."""

from __future__ import annotations

import numpy as np


def entropy(probabilities: list[float]) -> float:
    """Compute Shannon entropy for a probability vector."""

    values = np.asarray(probabilities, dtype=float)
    values = values[values > 0]
    return float(-(values * np.log2(values)).sum())
