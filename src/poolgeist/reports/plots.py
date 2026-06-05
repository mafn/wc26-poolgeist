"""Matplotlib-only plotting helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from poolgeist.schemas import ModelSignal


def score_heatmap(signal: ModelSignal):
    """Plot a score probability heatmap as a separate matplotlib chart."""

    fig, ax = plt.subplots()
    image = ax.imshow(signal.score_matrix, origin="lower")
    ax.set_xlabel("Away goals")
    ax.set_ylabel("Home goals")
    ax.set_title("Score probability heatmap")
    fig.colorbar(image, ax=ax)
    return fig


def bar_probabilities(frame: pd.DataFrame, *, x: str, y: str, title: str):
    """Plot a bar chart for probability tables."""

    fig, ax = plt.subplots()
    ax.bar(frame[x], frame[y])
    ax.set_title(title)
    ax.set_ylabel(y)
    return fig
