"""Kaggle submission export: a separate keyed one-to-one mapping, not a model schema."""

from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["target_date", "store_nbr", "family"]


class ExportError(ValueError):
    """Export refused; no submission may be published."""


def export_kaggle_submission(forecast: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Join forecast rows to `test.csv` IDs by key and restore `test.csv` order exactly."""

    targets = test[["id", "date", "store_nbr", "family"]].rename(columns={"date": "target_date"})
    targets = targets.assign(
        target_date=pd.to_datetime(targets["target_date"], format="%Y-%m-%d"),
        store_nbr=targets["store_nbr"].astype("int64"),
        family=targets["family"].astype(str),
    )
    if targets["id"].duplicated().any():
        raise ExportError("EXPORT_KEY_INVALID: duplicate test id")
    if targets.duplicated(KEY).any():
        raise ExportError("EXPORT_KEY_INVALID: duplicate test key")
    values = forecast[[*KEY, "prediction"]].copy()
    values["target_date"] = pd.to_datetime(values["target_date"]).astype("datetime64[ns]")
    targets["target_date"] = targets["target_date"].astype("datetime64[ns]")
    if values.duplicated(KEY).any():
        raise ExportError("EXPORT_KEY_INVALID: duplicate forecast key")
    joined = targets.merge(values, on=KEY, how="outer", indicator=True, validate="one_to_one")
    if joined["_merge"].eq("left_only").any():
        raise ExportError("EXPORT_KEY_INVALID: test key without a forecast")
    if joined["_merge"].eq("right_only").any():
        raise ExportError("EXPORT_KEY_INVALID: forecast key absent from test")
    submission = targets[["id", *KEY]].merge(values, on=KEY, how="left", validate="one_to_one")
    sales = submission["prediction"].to_numpy(dtype=np.float64)
    if not np.isfinite(sales).all() or (sales < 0).any():
        raise ExportError("EXPORT_VALUE_INVALID: predictions must be finite and nonnegative")
    result = pd.DataFrame({"id": submission["id"].astype("int64"), "sales": sales})
    if not result["id"].equals(test["id"].astype("int64").reset_index(drop=True)):
        raise ExportError("EXPORT_ORDER_INVALID: test order not preserved")
    return result
