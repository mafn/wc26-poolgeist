"""Popularity-adjustment helpers."""

from __future__ import annotations


def contrarian_value(model_probability: float, popularity_probability: float) -> float:
    """Return a simple neutral gap between model probability and pool popularity."""

    return model_probability - popularity_probability
