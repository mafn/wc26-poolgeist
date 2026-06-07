"""Model council and ensemble blending utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from poolgeist.config import EnsembleConfig, ModelWeightsConfig, ScoringConfig, StrategyConfig
from poolgeist.models.base import MatchModel, matrix_to_signal
from poolgeist.models.bayesian_update import BayesianFormUpdateModel
from poolgeist.models.correlated_goals import CorrelatedGoalsModel
from poolgeist.models.dixon_coles import DixonColesModel
from poolgeist.models.ising import IsingPressureModel
from poolgeist.models.market import MarketPriorModel
from poolgeist.models.negative_binomial import NegativeBinomialGoalsModel
from poolgeist.models.news import NewsSignalModel
from poolgeist.models.octopus import RandomOctopusOracle
from poolgeist.models.poisson import PoissonGoalsModel
from poolgeist.models.skellam import SkellamGoalDifferenceModel
from poolgeist.models.spin_flip import SpinFlipVolatilityModel
from poolgeist.models.temperature import TemperatureChaosModel
from poolgeist.optimization.chaos import chaos_index
from poolgeist.optimization.exact_scores import candidate_score_ev_table
from poolgeist.schemas import ModelSignal
from poolgeist.utils.probability import normalize


@dataclass(frozen=True)
class EnsemblePrediction:
    """A blended match prediction with uncertainty and recommendation classes."""

    signal: ModelSignal
    component_signals: list[ModelSignal]
    score_ev_table: pd.DataFrame
    chaos_index: float
    model_disagreement_index: float
    strategic_pool_value: float
    recommendation_classes: dict[str, str]
    warnings: list[str]


def normalize_weights(signals: Iterable[ModelSignal]) -> list[float]:
    """Normalize model weights for a sequence of signals."""

    signal_list = list(signals)
    if not signal_list:
        raise ValueError("At least one signal is required.")
    return normalize(
        np.array([signal.model_weight for signal in signal_list], dtype=float)
    ).tolist()


MODEL_NAME_TO_WEIGHT_KEY = {
    "independent_poisson": "poisson",
    "dixon_coles": "dixon_coles",
    "negative_binomial": "negative_binomial",
    "correlated_goals": "correlated_goals",
    "skellam_goal_difference": "skellam",
    "market_prior": "market",
    "news_signals": "news",
    "bayesian_form_update": "bayesian_update",
    "temperature_chaos": "temperature",
    "ising_pressure": "ising",
    "spin_flip": "spin_flip",
    "random_octopus_oracle": "octopus",
}


def configured_weight_for_signal(
    signal: ModelSignal, weights_config: ModelWeightsConfig
) -> float | None:
    """Return the configured weight for a model signal, falling back for custom models."""

    weight_key = MODEL_NAME_TO_WEIGHT_KEY.get(signal.model_name)
    if weight_key is None:
        return signal.model_weight
    return float(asdict(weights_config)[weight_key])


def apply_configured_weights(
    signals: Iterable[ModelSignal], weights_config: ModelWeightsConfig
) -> list[ModelSignal]:
    """Apply configured model-council weights and drop zero-weight disabled signals."""

    configured_signals = []
    for signal in signals:
        configured_weight = configured_weight_for_signal(signal, weights_config)
        if configured_weight is None:
            configured_weight = signal.model_weight
        if configured_weight < 0:
            raise ValueError(
                f"Configured model weight for {signal.model_name} must be non-negative."
            )
        if configured_weight == 0:
            continue
        configured_signals.append(replace(signal, model_weight=configured_weight))
    return configured_signals


def blend_signals(signals: Iterable[ModelSignal], *, source: str = "ensemble") -> ModelSignal:
    """Blend model signals by normalized model weights."""

    signal_list = list(signals)
    weights = normalize_weights(signal_list)
    matrix = sum(
        weight * signal.score_matrix for weight, signal in zip(weights, signal_list, strict=True)
    )
    component_weights = {
        signal.model_name: weight for signal, weight in zip(signal_list, weights, strict=True)
    }
    return matrix_to_signal(
        matrix,
        model_name=source,
        model_weight=sum(signal.model_weight for signal in signal_list),
        home_team=signal_list[0].home_team,
        away_team=signal_list[0].away_team,
        explanations=["Weighted council blend; unavailable model weights are redistributed."],
        warnings=[warning for signal in signal_list for warning in signal.warnings],
        metadata={
            "components": [signal.model_name for signal in signal_list],
            "component_weights": component_weights,
            "configured_model_weights": {
                signal.model_name: signal.model_weight for signal in signal_list
            },
        },
    )


class ModelCouncil:
    """Run all enabled model-council members and blend their score distributions."""

    def __init__(
        self, models: Sequence[MatchModel] | None = None, config: EnsembleConfig | None = None
    ):
        self.config = config or EnsembleConfig()
        self.models = list(models) if models is not None else self._default_models()

    def _default_models(self) -> list[MatchModel]:
        return [
            PoissonGoalsModel(),
            DixonColesModel(),
            NegativeBinomialGoalsModel(),
            CorrelatedGoalsModel(),
            SkellamGoalDifferenceModel(),
            MarketPriorModel(),
            NewsSignalModel(),
            BayesianFormUpdateModel(),
            TemperatureChaosModel(),
            IsingPressureModel(),
            SpinFlipVolatilityModel(),
            RandomOctopusOracle(seed=2026),
        ]

    def predict_match(
        self, home_team: str, away_team: str, is_knockout: bool = False
    ) -> EnsemblePrediction:
        """Predict a match and produce safe/EV/strategic/chaos recommendation classes."""

        signals: list[ModelSignal] = []
        warnings: list[str] = []
        for model in self.models:
            try:
                signals.append(model.predict_match(home_team, away_team))
            except Exception as exc:  # noqa: BLE001 - model-council degradation is intentional.
                warnings.append(f"Model {type(model).__name__} skipped: {exc}")
        if not signals:
            raise ValueError("No model council members produced a valid signal.")
        signals = apply_configured_weights(signals, self.config.weights)
        if not signals:
            raise ValueError("No model council members have positive configured weight.")
        blended = blend_signals(signals)
        disagreement = float(
            np.mean(
                [np.abs(signal.score_matrix - blended.score_matrix).sum() for signal in signals]
            )
        )
        chaos = chaos_index(blended, model_disagreement=disagreement)
        if is_knockout:
            from poolgeist.optimization.exact_scores import adjust_matrix_for_knockout
            from poolgeist.simulation.penalties import shootout_win_probability

            # Adjust the blended score matrix for extra time and shootout outcomes
            home_prob_win = shootout_win_probability()
            ko_matrix = adjust_matrix_for_knockout(blended.score_matrix, home_prob_win)

            # Rebuild the blended ModelSignal with the knockout-adjusted matrix
            blended = matrix_to_signal(
                ko_matrix,
                model_name=blended.model_name,
                model_weight=blended.model_weight,
                home_team=blended.home_team,
                away_team=blended.away_team,
                explanations=blended.explanations,
                warnings=blended.warnings + ["Adjusted for knockout/penalties"],
                metadata=blended.metadata,
            )

        ev_table = candidate_score_ev_table(
            blended, ScoringConfig(), StrategyConfig(), chaos, disagreement
        )

        recs = {
            "safest": _top_score(ev_table, "exact_score_probability"),
            "highest_raw_ev": _top_score(ev_table, "expected_points"),
            "highest_strategic_value": _top_score(ev_table, "strategic_value"),
            "highest_chaos_value": _top_score(ev_table, "chaos_value"),
            "consensus_pick": _top_score(ev_table, "expected_points"),
            "model_disagreement_pick": _top_score(ev_table, "disagreement_value"),
        }
        return EnsemblePrediction(
            blended,
            signals,
            ev_table,
            chaos,
            disagreement,
            float(ev_table["strategic_value"].max()),
            recs,
            warnings + blended.warnings,
        )


def _top_score(frame: pd.DataFrame, column: str) -> str:
    row = frame.sort_values(column, ascending=False).iloc[0]
    return f"{int(row.pred_home_goals)}-{int(row.pred_away_goals)}"
