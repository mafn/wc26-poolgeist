"""Typed configuration for Poolgeist."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class SimulationConfig:
    """Monte Carlo and score-grid configuration."""

    quick_simulations: int = 5_000
    serious_simulations: int = 50_000
    n_simulations: int = 5_000
    random_seed: int | None = 2026
    max_goals_model: int = 10
    max_goals_display: int = 7
    include_third_place_match: bool = False


@dataclass(frozen=True)
class ScoringConfig:
    """Kicktipp-style 9er-compatible office-pool scoring configuration."""

    exact_score: int = 4
    exact_winning_score: int = 4
    exact_draw_score: int = 4
    correct_goal_difference: int = 3
    correct_draw_tendency: int = 3
    correct_winner_only: int = 2
    wrong_tendency: int = 0


@dataclass(frozen=True)
class ModelWeightsConfig:
    """Configurable model-council weights before normalization."""

    poisson: float = 0.32
    dixon_coles: float = 0.12
    negative_binomial: float = 0.10
    correlated_goals: float = 0.08
    skellam: float = 0.08
    market: float = 0.08
    team_cards: float = 0.08
    news: float = 0.04
    bayesian_update: float = 0.04
    temperature: float = 0.03
    ising: float = 0.02
    spin_flip: float = 0.01
    octopus: float = 0.005


@dataclass(frozen=True)
class EnsembleConfig:
    """Ensemble blending and calibration configuration."""

    weights: ModelWeightsConfig = field(default_factory=ModelWeightsConfig)
    redistribute_missing_model_weight: bool = True
    calibration_temperature: float = 1.0
    dixon_coles_rho: float = -0.08


@dataclass(frozen=True)
class SearchConfig:
    """Optional search/news configuration; disabled by default for public safety."""

    enable_web_search: bool = False
    manual_news_csv: str | None = None
    cache_enabled: bool = False
    max_results: int = 5


@dataclass(frozen=True)
class LLMConfig:
    """Optional LLM extraction config. Model names come only from environment/config."""

    enable_llm_news: bool = False
    openai_api_key: str | None = None
    openai_model: str | None = None

    @classmethod
    def from_environment(cls) -> LLMConfig:
        """Create config from environment variables without hard-coded model names."""

        enabled = os.getenv("POOLGEIST_ENABLE_LLM_NEWS", "false").lower() == "true"
        return cls(
            enable_llm_news=enabled,
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_model=os.getenv("OPENAI_MODEL") or None,
        )


@dataclass(frozen=True)
class StrategyConfig:
    """Strategic optimization weights."""

    lambda_uniqueness: float = 0.4
    lambda_chaos: float = 0.15
    lambda_variance: float = 0.1


@dataclass(frozen=True)
class PoolgeistConfig:
    """Top-level Poolgeist configuration."""

    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    @property
    def n_simulations(self) -> int:
        """Backward-compatible access to simulation count."""

        return self.simulation.n_simulations

    @property
    def random_seed(self) -> int | None:
        """Backward-compatible access to seed."""

        return self.simulation.random_seed


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file, returning an empty dict for empty files."""

    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _dataclass_kwargs(config_type: type[Any], values: dict[str, Any]) -> dict[str, Any]:
    """Return keys accepted by a dataclass config type."""

    allowed = {field.name for field in fields(config_type)}
    return {key: value for key, value in values.items() if key in allowed}


def load_scoring_config(path: str | Path) -> ScoringConfig:
    """Load scoring config from a YAML file with either root or ``scoring`` keys."""

    data = load_yaml_config(path)
    scoring_data = data.get("scoring", data)
    if not isinstance(scoring_data, dict):
        raise ValueError("Scoring config must be a mapping.")
    return ScoringConfig(**_dataclass_kwargs(ScoringConfig, scoring_data))


def load_environment(dotenv_path: str | Path | None = None) -> None:
    """Load environment variables from a .env file when present."""

    load_dotenv(dotenv_path=dotenv_path)
