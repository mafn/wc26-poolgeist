import types

import numpy as np
import pytest

from poolgeist.adapters.openai_client import OpenAIResponsesAdapter
from poolgeist.models.base import matrix_to_signal
from poolgeist.models.poisson import PoissonGoalsModel
from poolgeist.optimization.chaos import chaos_index
from poolgeist.simulation.tournament import simulate_simple_knockout


class _FakeCompletions:
    def __init__(self):
        self.called_with = None

    def create(self, **kwargs):
        self.called_with = kwargs
        message = types.SimpleNamespace(content='{"team":"Neutral A","signal_type":"injury"}')
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice])


class _FakeClient:
    def __init__(self):
        self.completions = _FakeCompletions()
        self.chat = types.SimpleNamespace(completions=self.completions)


def test_openai_adapter_uses_chat_completions_content():
    adapter = OpenAIResponsesAdapter(api_key="test-key", model="test-model")
    adapter.enabled = True
    adapter.client = _FakeClient()

    result = adapter.extract_structured_signal("Neutral injury note")

    assert result == {
        "raw_text": '{"team":"Neutral A","signal_type":"injury"}',
        "model": "test-model",
    }
    assert adapter.client.completions.called_with["model"] == "test-model"
    assert adapter.client.completions.called_with["messages"][1]["content"] == "Neutral injury note"


def test_odd_length_knockout_round_gives_bye():
    result = simulate_simple_knockout(
        ["A", "B", "C"],
        PoissonGoalsModel(home_xg=1.0, away_xg=1.0, max_goals=2),
        seed=11,
    )
    assert result.iloc[0]["team"] in {"A", "B", "C"}


def test_empty_knockout_rejected():
    with pytest.raises(ValueError):
        simulate_simple_knockout([], PoissonGoalsModel())


def test_single_cell_matrix_entropy_is_safe():
    signal = matrix_to_signal(
        np.ones((1, 1)),
        model_name="single_cell",
        model_weight=1.0,
        home_team="A",
        away_team="B",
    )
    assert signal.uncertainty == 0.0
    assert chaos_index(signal) >= 0.0
