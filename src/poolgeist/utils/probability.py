"""Probability normalization and validation utilities."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

DEFAULT_OUTCOMES = ("home", "draw", "away")


def normalize(
    values: Mapping[str, float] | np.ndarray, *, outcomes: tuple[str, ...] = DEFAULT_OUTCOMES
):
    """Normalize non-negative values into probabilities."""

    if isinstance(values, Mapping):
        array = np.array([float(values.get(outcome, 0.0)) for outcome in outcomes], dtype=float)
        normalized = _normalize_array(array)
        return dict(zip(outcomes, normalized, strict=True))
    return _normalize_array(np.asarray(values, dtype=float))


def _normalize_array(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError("Expected a one-dimensional probability vector.")
    if np.isnan(values).any():
        raise ValueError("Probabilities must not contain NaN values.")
    if (values < 0).any():
        if (values < -1e-9).any():
            raise ValueError("Probabilities must be non-negative.")
        values = np.maximum(values, 0.0)
    total = float(values.sum())
    if total <= 0:
        raise ValueError("At least one probability weight must be positive.")
    return values / total


def assert_probability_vector(
    values: Mapping[str, float] | np.ndarray, *, atol: float = 1e-9
) -> None:
    """Raise if values are not finite, non-negative, and summing to one."""

    array = np.array(
        list(values.values()) if isinstance(values, Mapping) else list(values), dtype=float
    )
    if np.isnan(array).any() or not np.isfinite(array).all():
        raise ValueError("Probabilities must be finite and non-NaN.")
    if (array < -atol).any():
        raise ValueError("Probabilities must be non-negative.")
    if not np.isclose(array.sum(), 1.0, atol=atol):
        raise ValueError("Probabilities must sum to one.")
