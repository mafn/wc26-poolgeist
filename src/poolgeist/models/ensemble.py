"""Model-signal blending utilities."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from poolgeist.models.base import ModelSignal
from poolgeist.utils.probability import DEFAULT_OUTCOMES, normalize


def normalize_weights(signals: Iterable[ModelSignal]) -> list[float]:
    """Normalize model weights for a sequence of signals."""

    signal_list = list(signals)
    if not signal_list:
        raise ValueError("At least one signal is required.")
    weights = np.array([signal.weight for signal in signal_list], dtype=float)
    return normalize(weights).tolist()


def blend_signals(signals: Iterable[ModelSignal], *, source: str = "ensemble") -> ModelSignal:
    """Blend model signals by normalized model weights."""

    signal_list = list(signals)
    weights = normalize_weights(signal_list)
    blended = {
        outcome: float(
            sum(
                weight * signal.probabilities[outcome]
                for weight, signal in zip(weights, signal_list, strict=True)
            )
        )
        for outcome in DEFAULT_OUTCOMES
    }
    return ModelSignal.from_weights(
        blended,
        weight=sum(signal.weight for signal in signal_list),
        source=source,
        metadata={"components": [signal.source for signal in signal_list]},
    )
