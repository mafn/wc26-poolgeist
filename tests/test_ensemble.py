import pytest

from poolgeist.models.base import ModelSignal
from poolgeist.models.ensemble import blend_signals, normalize_weights


def test_ensemble_weights_normalize():
    signals = [
        ModelSignal(home=0.5, draw=0.25, away=0.25, weight=2),
        ModelSignal(home=0.25, draw=0.25, away=0.5, weight=1),
    ]
    weights = normalize_weights(signals)
    assert weights == pytest.approx([2 / 3, 1 / 3])


def test_blend_signal_sums_to_one():
    signal = blend_signals(
        [
            ModelSignal(home=0.5, draw=0.25, away=0.25, weight=2),
            ModelSignal(home=0.25, draw=0.25, away=0.5, weight=1),
        ]
    )
    assert sum(signal.probabilities.values()) == pytest.approx(1.0)
