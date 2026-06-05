"""Data loading helpers for neutral examples and templates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV file using UTF-8."""

    return pd.read_csv(Path(path))


def load_demo_teams(repo_root: str | Path = ".") -> pd.DataFrame:
    """Load clearly marked neutral demo teams."""

    return read_csv(Path(repo_root) / "examples" / "neutral_demo_teams.csv")


def load_demo_fixtures(repo_root: str | Path = ".") -> pd.DataFrame:
    """Load clearly marked neutral demo fixtures."""

    return read_csv(Path(repo_root) / "examples" / "neutral_demo_fixtures.csv")
