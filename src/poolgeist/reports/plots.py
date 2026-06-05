"""Plotting helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt

from poolgeist.models.base import ModelSignal


def plot_signal(signal: ModelSignal):
    """Create a bar chart for a match signal."""

    fig, ax = plt.subplots()
    ax.bar(["home", "draw", "away"], [signal.home, signal.draw, signal.away])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title(signal.source)
    return fig, ax
