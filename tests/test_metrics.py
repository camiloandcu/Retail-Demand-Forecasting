"""Official metric tests."""

import math

import pytest

from retail_forecast.metrics import clip_nonnegative_predictions, rmsle


def test_rmsle_is_zero_for_exact_fractional_predictions() -> None:
    assert rmsle([0.0, 1.5, 10.0], [0.0, 1.5, 10.0]) == 0.0


def test_rmsle_matches_manual_formula() -> None:
    expected = math.sqrt(((math.log1p(1.0) - math.log1p(0.0)) ** 2) / 2)
    assert rmsle([0.0, 1.0], [1.0, 1.0]) == pytest.approx(expected)


@pytest.mark.parametrize("actual,predicted", [([-1.0], [0.0]), ([0.0], [-1.0])])
def test_rmsle_rejects_negative_values(actual: list[float], predicted: list[float]) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        rmsle(actual, predicted)


def test_rmsle_rejects_shape_mismatch_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        rmsle([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="finite"):
        rmsle([1.0], [math.nan])
    with pytest.raises(ValueError, match="at least one"):
        rmsle([], [])


def test_prediction_clipping_is_explicit_and_rejects_nonfinite_values() -> None:
    assert clip_nonnegative_predictions([-2.0, 0.0, 3.5]).tolist() == [0.0, 0.0, 3.5]
    with pytest.raises(ValueError, match="finite"):
        clip_nonnegative_predictions([math.inf])
