"""Seasonal-naive causality tests."""

import pandas as pd

from retail_forecast.baseline import fit_seasonal_naive, predict_seasonal_naive


def test_predictions_after_day_seven_do_not_read_validation_sales() -> None:
    history_dates = pd.date_range("2020-01-01", periods=21, freq="D")
    history = pd.DataFrame(
        {
            "date": history_dates,
            "store_nbr": 1,
            "family": "SYNTHETIC_FAMILY_01",
            "sales": range(1, 22),
        }
    )
    future = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-22", periods=16, freq="D"),
            "store_nbr": 1,
            "family": "SYNTHETIC_FAMILY_01",
            "sales": 99999.0,
        }
    )
    model = fit_seasonal_naive(history, history_dates.max())
    first = predict_seasonal_naive(model, future)
    future["sales"] = -99999.0
    second = predict_seasonal_naive(model, future)
    assert first.tolist() == second.tolist()
    assert first.iloc[0] == first.iloc[7] == first.iloc[14]
