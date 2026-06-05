"""Human-readable summary helpers."""

from __future__ import annotations

from poolgeist.models.ensemble import EnsemblePrediction


def summarize_prediction(prediction: EnsemblePrediction) -> str:
    """Summarize uncertainty, disagreement, and recommendation classes."""

    return (
        f"Chaos index={prediction.chaos_index:.3f}; "
        f"model disagreement={prediction.model_disagreement_index:.3f}; "
        f"recommendations={prediction.recommendation_classes}"
    )
