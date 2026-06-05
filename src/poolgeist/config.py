"""Configuration helpers for Poolgeist."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class PoolgeistConfig:
    """Minimal runtime configuration."""

    n_simulations: int = 1_000
    random_seed: int | None = 2026
    enable_llm_news: bool = False
    enable_web_search: bool = False


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file, returning an empty dict for empty files."""

    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_environment(dotenv_path: str | Path | None = None) -> None:
    """Load environment variables from a .env file when present."""

    load_dotenv(dotenv_path=dotenv_path)
