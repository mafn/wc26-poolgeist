"""Shared validation helpers."""

from __future__ import annotations


def require_columns(columns: set[str], required: set[str]) -> None:
    """Raise a ValueError if required columns are missing."""

    missing = required - columns
    if missing:
        msg = f"Missing required columns: {sorted(missing)}"
        raise ValueError(msg)
