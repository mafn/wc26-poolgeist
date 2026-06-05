"""Random-number helpers."""

from __future__ import annotations

import numpy as np


def rng(seed: int | None = None) -> np.random.Generator:
    """Create a NumPy random generator."""

    return np.random.default_rng(seed)
