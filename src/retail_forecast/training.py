"""Eligible training inputs for one cutoff, shared by the baseline and bundle fits."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from retail_forecast.features import (
    FeatureBatch,
    FeatureSpec,
    build_origin_feature_batch,
    project_benchmark_request,
)
from retail_forecast.models.base import TrainingSet

SeriesKey = tuple[int, str]


def cutoff_training_origins(
    cutoff: pd.Timestamp,
    horizon: int,
    available_dates: pd.Series | pd.DatetimeIndex,
    *,
    stride_days: int,
    max_origins: int,
    name: str,
) -> tuple[pd.Timestamp, ...]:
    """Keep only pre-registered origins whose full target horizon ends by the cutoff."""

    observed = pd.DatetimeIndex(pd.to_datetime(available_dates, errors="raise")).normalize()
    if observed.empty:
        raise ValueError("Observed training dates cannot be empty")
    observed = pd.DatetimeIndex(observed.unique()).sort_values()
    if observed.max() < cutoff:
        raise ValueError(f"Training dates do not reach the forecast origin for {name}")
    latest = cutoff - pd.offsets.Day(horizon)
    candidates = [latest - pd.offsets.Day(index * stride_days) for index in range(max_origins)]
    valid = [
        origin
        for origin in candidates
        if origin >= observed.min()
        and pd.date_range(origin + pd.offsets.Day(1), periods=horizon, freq="D")
        .isin(observed)
        .all()
    ]
    if not valid:
        raise ValueError(f"No complete training horizon is available for {name}")
    return tuple(sorted(valid))


def combine_feature_batches(batches: list[FeatureBatch]) -> FeatureBatch:
    """Combine identically specified historical origins without changing their features."""

    if not batches:
        raise ValueError("At least one feature batch is required")
    first = batches[0]
    for batch in batches[1:]:
        if (
            batch.numeric_features != first.numeric_features
            or batch.categorical_features != first.categorical_features
            or batch.spec != first.spec
        ):
            raise ValueError("Historical feature batches must share one schema")
    frame = pd.concat([batch.frame for batch in batches], ignore_index=True)
    return FeatureBatch(
        frame=frame,
        numeric_features=first.numeric_features,
        categorical_features=first.categorical_features,
        origin=max(batch.origin for batch in batches),
        spec=first.spec,
    )


def prepare_feature_frames(frames: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Parse the dated panels once; other frames pass through unchanged."""

    prepared = dict(frames)
    for name in ("train.csv", "transactions.csv"):
        frame = frames[name].copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        prepared[name] = frame
    return prepared


@dataclass
class FeatureBatchCache:
    """Origin batches keyed by snapshot, origin and policy-bearing feature spec."""

    entries: dict[tuple[str, pd.Timestamp, str], FeatureBatch]

    @classmethod
    def empty(cls) -> FeatureBatchCache:
        return cls({})


def build_training_set(
    frames: Mapping[str, pd.DataFrame],
    *,
    cutoff: pd.Timestamp,
    spec: FeatureSpec,
    config: Any,
    snapshot_id: str,
    data_source: str,
    supported_series: tuple[SeriesKey, ...],
    needs_batch: bool,
    params: Mapping[str, Any],
    name: str,
    cache: FeatureBatchCache | None = None,
) -> tuple[TrainingSet, dict[str, Any]]:
    """Build the cutoff request and, when needed, labeled origins ending by the cutoff."""

    cutoff = pd.Timestamp(cutoff).normalize()
    identity = {
        "snapshot_id": snapshot_id,
        "data_source": data_source,
        "supported_series": supported_series,
    }
    request = project_benchmark_request(frames, cutoff, spec, **identity)
    train_dates = frames["train.csv"]["date"]
    origins: tuple[pd.Timestamp, ...] = ()
    batch = None
    record: dict[str, Any] = {"training_origins": [], "training_rows": 0}
    if needs_batch:
        origins = cutoff_training_origins(
            cutoff,
            spec.horizon,
            train_dates,
            stride_days=config.baseline.training_origin_stride_days,
            max_origins=config.baseline.max_training_origins,
            name=name,
        )
        cache = cache or FeatureBatchCache.empty()
        keys = [(snapshot_id, origin, spec.sha256) for origin in origins]
        for key, origin in zip(keys, origins, strict=True):
            if key not in cache.entries:
                cache.entries[key] = build_origin_feature_batch(frames, origin, spec, **identity)
        batch = combine_feature_batches([cache.entries[key] for key in keys])
        if batch.frame["target_date"].max() > cutoff:
            raise ValueError(f"{name} training labels cross the cutoff")
        expected = len(origins) * len(supported_series) * spec.horizon
        if len(batch.frame) != expected:
            raise ValueError(f"{name} training grid is incomplete")
        record = {
            "training_origins": [origin.date().isoformat() for origin in origins],
            "training_rows": len(batch.frame),
            "training_rows_expected": expected,
        }
    training = TrainingSet(
        cutoff=cutoff,
        request=request,
        spec=spec,
        supported_series=supported_series,
        batch=batch,
        training_origins=origins,
        params=params,
    )
    return training, record
