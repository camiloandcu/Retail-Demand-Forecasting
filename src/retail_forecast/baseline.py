"""Mandatory seasonal-naive baseline shared by notebooks and batch commands."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd


def _key(store: object, family: object, weekday: int | str) -> str:
    return f"{store}|{family}|{weekday}"


def fit_seasonal_naive(
    train: pd.DataFrame, origin: date | pd.Timestamp, seasonal_period: int = 7
) -> dict[str, Any]:
    """Fit the last observed value by weekday without reading beyond the origin."""

    if seasonal_period != 7:
        raise ValueError("The current daily seasonal-naive baseline requires a period of 7")
    history = train.copy()
    history["date"] = pd.to_datetime(history["date"], errors="raise")
    cutoff = pd.Timestamp(origin)
    history = history.loc[history["date"] <= cutoff].sort_values("date")
    if history.empty:
        raise ValueError("No observations are available on or before the forecast origin")
    if (history["sales"] < 0).any():
        raise ValueError("Seasonal-naive training sales must be non-negative")
    history["weekday"] = history["date"].dt.dayofweek

    weekday_last = history.groupby(["store_nbr", "family", "weekday"], sort=True).tail(1)
    series_last = history.groupby(["store_nbr", "family"], sort=True).tail(1)
    return {
        "model_type": "seasonal_naive_weekday",
        "origin": cutoff.date().isoformat(),
        "seasonal_period": seasonal_period,
        "weekday_values": {
            _key(row.store_nbr, row.family, row.weekday): float(row.sales)
            for row in weekday_last.itertuples(index=False)
        },
        "fallback_values": {
            _key(row.store_nbr, row.family, "fallback"): float(row.sales)
            for row in series_last.itertuples(index=False)
        },
    }


def predict_seasonal_naive(model: dict[str, Any], frame: pd.DataFrame) -> pd.Series:
    """Predict arbitrary future rows from a fitted seasonal-naive artifact."""

    dates = pd.to_datetime(frame["date"], errors="raise")
    origin = pd.Timestamp(model["origin"])
    if (dates <= origin).any():
        raise ValueError("Prediction dates must be strictly after the model origin")

    values: list[float] = []
    for row, weekday in zip(frame.itertuples(index=False), dates.dt.dayofweek, strict=True):
        seasonal_key = _key(row.store_nbr, row.family, int(weekday))
        fallback_key = _key(row.store_nbr, row.family, "fallback")
        value = model["weekday_values"].get(seasonal_key)
        if value is None:
            value = model["fallback_values"].get(fallback_key)
        if value is None:
            raise ValueError(
                f"Unknown series at inference: store={row.store_nbr}, family={row.family}"
            )
        values.append(max(0.0, float(value)))
    return pd.Series(values, index=frame.index, name="sales", dtype=float)


def fit_seasonal_average(
    train: pd.DataFrame,
    origin: date | pd.Timestamp,
    weeks: int = 4,
    seasonal_period: int = 7,
) -> dict[str, Any]:
    """Average the last pre-registered number of same-weekday observations."""

    if weeks <= 0:
        raise ValueError("Seasonal-average weeks must be positive")
    if seasonal_period != 7:
        raise ValueError("The current daily seasonal average requires a period of 7")
    history = train.copy()
    history["date"] = pd.to_datetime(history["date"], errors="raise")
    cutoff = pd.Timestamp(origin)
    history = history.loc[history["date"] <= cutoff].sort_values("date")
    if history.empty or (history["sales"] < 0).any():
        raise ValueError("Seasonal-average history must be present and non-negative")
    history["weekday"] = history["date"].dt.dayofweek
    recent = history.groupby(["store_nbr", "family", "weekday"], sort=True).tail(weeks)
    weekday_mean = recent.groupby(["store_nbr", "family", "weekday"], observed=True)["sales"].mean()
    series_mean = history.groupby(["store_nbr", "family"], observed=True)["sales"].tail(
        weeks * seasonal_period
    )
    fallback_frame = history.loc[series_mean.index]
    fallback_mean = fallback_frame.groupby(["store_nbr", "family"], observed=True)["sales"].mean()
    return {
        "model_type": f"seasonal_average_{weeks}w",
        "origin": cutoff.date().isoformat(),
        "seasonal_period": seasonal_period,
        "weeks": weeks,
        "weekday_values": {
            _key(store, family, weekday): float(value)
            for (store, family, weekday), value in weekday_mean.items()
        },
        "fallback_values": {
            _key(store, family, "fallback"): float(value)
            for (store, family), value in fallback_mean.items()
        },
    }


def predict_seasonal_average(model: dict[str, Any], frame: pd.DataFrame) -> pd.Series:
    """Predict from a fitted same-weekday seasonal-average artifact."""

    return predict_seasonal_naive(model, frame)
