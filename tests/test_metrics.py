"""Official metric tests."""

import math

import pytest

from retail_forecast.metrics import rmsle


def test_rmsle_is_zero_for_exact_fractional_predictions() -> None:
    assert rmsle([0.0, 1.5, 10.0], [0.0, 1.5, 10.0]) == 0.0


def test_rmsle_matches_manual_formula() -> None:
    expected = math.sqrt(((math.log1p(1.0) - math.log1p(0.0)) ** 2) / 2)
    assert rmsle([0.0, 1.0], [1.0, 1.0]) == pytest.approx(expected)


@pytest.mark.parametrize("actual,predicted", [([-1.0], [0.0]), ([0.0], [-1.0])])
def test_rmsle_rejects_negative_values(actual: list[float], predicted: list[float]) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        rmsle(actual, predicted)
