"""Canonical prediction from a verified bundle to a validated ForecastFrame."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from retail_forecast.bundles import BundleError, LoadedModel
from retail_forecast.data_contracts import (
    FORECAST_COLUMNS,
    FORECAST_ORDER,
    ForecastRequest,
    ForecastRequestError,
    validate_forecast_frame,
    validate_forecast_request,
)
from retail_forecast.models import get_adapter
from retail_forecast.models.base import RAW_PREDICTION_COLUMNS, PredictionContext


def predict_bundle(
    loaded: LoadedModel, request: ForecastRequest, *, context: PredictionContext
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Predict without fitting, network or labels; nothing partial is ever returned."""

    manifest = loaded.manifest
    fitted = loaded.fitted
    if request.availability_policy != manifest["availability_policy"]:
        raise BundleError("POLICY_MISMATCH", "request policy differs from the bundle policy")
    if request.input_contract_version != manifest["input_contract_version"]:
        raise BundleError("POLICY_MISMATCH", "request input contract version differs")
    if (context.execution_mode, context.data_source) != (
        manifest["execution_mode"],
        manifest["data_source"],
    ) or request.data_source != manifest["data_source"]:
        raise BundleError("BUNDLE_INVALID", "bundle mode/source differs from this prediction")
    if request.origin < fitted.training_cutoff:
        raise BundleError("ORIGIN_BEFORE_CUTOFF", "request origin precedes the training cutoff")
    validate_forecast_request(
        request,
        supported_series=fitted.supported_series,
        min_context_days=fitted.min_context_days,
    )

    observed = set(
        map(tuple, request.sales_history[["store_nbr", "family"]].drop_duplicates().to_numpy())
    )
    observed = {(int(s), str(f)) for s, f in observed}
    for store, family in request.series_keys:
        if (int(store), family) not in observed:
            raise ForecastRequestError(
                "HISTORY_INSUFFICIENT", "sales_history", f"store_nbr={store},family={family}"
            )
    # 2: models whose state is a snapshot at the cutoff cannot serve later origins.
    if (
        fitted.signature.get("origin_rule") == "equals_training_cutoff"
        and request.origin != fitted.training_cutoff
    ):
        raise BundleError(
            "ORIGIN_UNSUPPORTED", "this model forecasts only from its training cutoff"
        )

    raw = get_adapter(fitted.adapter_id).predict_raw(fitted, request)
    if tuple(raw.columns) != RAW_PREDICTION_COLUMNS:
        raise ForecastRequestError("SCHEMA_INVALID", "adapter.output")
    values = raw["prediction"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ForecastRequestError("NONFINITE_VALUE", "prediction")
    clipped = int((values < 0).sum())
    frame = raw[["store_nbr", "family", "target_date"]].copy()
    frame["store_nbr"] = frame["store_nbr"].astype("int64")
    frame["family"] = frame["family"].astype(str)
    frame["target_date"] = pd.to_datetime(frame["target_date"]).astype("datetime64[ns]")
    frame["forecast_origin"] = pd.Timestamp(request.origin)
    frame["forecast_origin"] = frame["forecast_origin"].astype("datetime64[ns]")
    frame["horizon"] = (frame["target_date"] - frame["forecast_origin"]).dt.days.astype("int64")
    frame["prediction"] = np.maximum(values, 0.0)
    lineage = {
        "model_id": manifest["model_id"],
        "model_version": manifest["model_version"],
        "bundle_id": manifest["bundle_id"],
        "training_run_id": manifest["training_run_id"],
        "prediction_run_id": context.prediction_run_id,
        "evaluation_run_id": context.evaluation_run_id,
        "snapshot_id": request.snapshot_id,
        "availability_policy": request.availability_policy,
        "input_contract_version": request.input_contract_version,
        "data_source": request.data_source,
        "execution_mode": context.execution_mode,
    }
    for column, value in lineage.items():
        frame[column] = value
    frame["evaluation_run_id"] = frame["evaluation_run_id"].astype(object)
    frame = (
        frame[list(FORECAST_COLUMNS)]
        .sort_values(list(FORECAST_ORDER), kind="mergesort")
        .reset_index(drop=True)
    )
    validate_forecast_frame(frame, request=request, expected_bundle_id=manifest["bundle_id"])
    diagnostics = {
        "rows": len(frame),
        "series": len(request.series_keys),
        "clipped_negative_count": clipped,
        "bundle_id": manifest["bundle_id"],
        "model_id": manifest["model_id"],
    }
    return frame, diagnostics


def write_forecast_csv(frame: pd.DataFrame, path: Any) -> None:
    """CSV keeps core installs pyarrow-free; floats round-trip with full precision."""

    output = frame.copy()
    for column in ("forecast_origin", "target_date"):
        output[column] = output[column].dt.strftime("%Y-%m-%d")
    output.to_csv(path, index=False, float_format="%.17g")


def read_forecast_csv(path: Any) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={
            "store_nbr": "int64",
            "family": str,
            "horizon": "int64",
            "prediction": "float64",
            "evaluation_run_id": object,
            "input_contract_version": "int64",
        },
        keep_default_na=False,
        na_values={"evaluation_run_id": [""]},
    )
    for column in ("forecast_origin", "target_date"):
        frame[column] = pd.to_datetime(frame[column], format="%Y-%m-%d").astype("datetime64[ns]")
    frame["evaluation_run_id"] = (
        frame["evaluation_run_id"].astype(object).where(frame["evaluation_run_id"].notna(), None)
    )
    return frame[list(FORECAST_COLUMNS)]
