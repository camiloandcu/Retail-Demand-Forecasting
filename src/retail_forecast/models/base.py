"""Typed adapter surface shared by every model family (see the model contract).

Adapters never read files or assign provenance. They fit from a `TrainingSet`,
serialize native state to bytes, rebuild native state from verified bytes, and
return raw keyed predictions. Bundle publication and prediction lineage belong to
`retail_forecast.bundles` and `retail_forecast.inference`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import pandas as pd

from retail_forecast.data_contracts import ForecastRequest, SeriesKey
from retail_forecast.features import FeatureBatch, FeatureSpec

RAW_PREDICTION_COLUMNS = ("store_nbr", "family", "target_date", "prediction")


@dataclass(frozen=True)
class FitContext:
    """Training identity issued by the run orchestrator, never by a caller."""

    training_run_id: str
    adapter_id: str
    availability_policy: str
    data_source: Literal["synthetic", "kaggle"]
    execution_mode: Literal["smoke", "full"]
    train_cutoff: pd.Timestamp
    selected_epochs: int | None
    seed: int
    config_sha256: str
    dataset_sha256: str
    input_contract_version: int


@dataclass(frozen=True)
class PredictionContext:
    """Prediction identity issued by the run orchestrator from trusted metadata."""

    prediction_run_id: str
    evaluation_run_id: str | None
    data_source: Literal["synthetic", "kaggle"]
    execution_mode: Literal["smoke", "full"]


@dataclass(frozen=True)
class TrainingSet:
    """Eligible training inputs ending at the training cutoff.

    `request` is the label-free request at the cutoff (history through the cutoff).
    `batch` holds labeled direct rows from eligible training origins; only adapters
    that declare `needs_training_batch` receive it.
    """

    cutoff: pd.Timestamp
    request: ForecastRequest
    spec: FeatureSpec
    supported_series: tuple[SeriesKey, ...]
    batch: FeatureBatch | None
    training_origins: tuple[pd.Timestamp, ...]
    params: Mapping[str, Any]


@dataclass(frozen=True)
class StateFile:
    """One native state file produced by an adapter."""

    data: bytes
    format: str


@dataclass(frozen=True)
class FittedModel:
    """Adapter output: immutable learned state plus its input/output signature."""

    adapter_id: str
    adapter_version: str
    availability_policy: str
    input_contract_version: int
    training_cutoff: pd.Timestamp
    supported_series: tuple[SeriesKey, ...]
    min_context_days: int
    request_context_days: int
    signature: dict[str, Any]
    preprocessing: dict[str, Any]
    native: Any = field(repr=False, compare=False)


class Adapter(Protocol):
    """Contract each registered model family implements."""

    adapter_id: str
    version: str
    needs_training_batch: bool

    def fit(self, training: TrainingSet, context: FitContext) -> FittedModel: ...

    def state_files(self, fitted: FittedModel) -> dict[str, StateFile]: ...

    def load(
        self,
        *,
        signature: dict[str, Any],
        preprocessing: dict[str, Any],
        state: Mapping[str, bytes],
        identity: dict[str, Any],
    ) -> FittedModel: ...

    def predict_raw(self, fitted: FittedModel, request: ForecastRequest) -> pd.DataFrame: ...


def base_signature(
    *,
    adapter_id: str,
    adapter_version: str,
    spec: FeatureSpec,
    request_context_days: int,
    min_context_days: int,
    target_transform: str,
) -> dict[str, Any]:
    """Signature fields every adapter records."""

    return {
        "schema_version": 1,
        "adapter_id": adapter_id,
        "adapter_version": adapter_version,
        "availability_policy": spec.availability_policy,
        "input_contract_version": spec.policy.input_contract_version,
        "horizon": spec.horizon,
        "request_context_days": request_context_days,
        "min_context_days": min_context_days,
        "target_transform": target_transform,
        "output": {
            "grain": "store_nbr,family,target_date",
            "scale": "original_sales",
            "nonnegative": True,
        },
    }
