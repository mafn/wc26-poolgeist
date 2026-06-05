import numpy as np
import pytest

from poolgeist.utils.probability import assert_probability_vector, normalize


def test_normalize_mapping_sums_to_one():
    probs = normalize({"home": 2, "draw": 1, "away": 1})
    assert sum(probs.values()) == pytest.approx(1.0)


def test_normalize_array_sums_to_one():
    probs = normalize(np.array([3.0, 3.0, 4.0]))
    assert probs.sum() == pytest.approx(1.0)


def test_assert_probability_vector_rejects_nan():
    with pytest.raises(ValueError):
        assert_probability_vector([0.5, np.nan, 0.5])
