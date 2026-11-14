"""Origin-anchored features that cannot read targets inside a forecast horizon."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from retail_forecast.availability import AvailabilityPolicy, PolicyMismatchError, resolve_policy
from retail_forecast.config import PROJECT_HORIZON
from retail_forecast.data_contracts import (
    ForecastRequest,
    attach_labels,
    benchmark_labels,
    project_snapshot,
    validate_forecast_request,
)
from retail_forecast.io_utils import write_json
from retail_forecast.splits import RollingOriginFold, SplitContractError

SERIES_KEYS: Final[tuple[str, str]] = ("store_nbr", "family")
TIME_COLUMN: Final = "date"
TARGET_COLUMN: Final = "sales"
UNKNOWN_CATEGORY: Final = "__UNKNOWN__"


class FeatureContractError(ValueError):
    """Raised when feature inputs could leak or violate the panel contract."""


@dataclass(frozen=True)
class FeatureSpec:
    """Pre-registered feature set; its identity includes the availability policy."""

    availability_policy: str
    sales_lags: tuple[int, ...] = (1, 7, 14, 28, 56)
    sales_windows: tuple[int, ...] = (7, 14, 28, 56)
    transaction_lags: tuple[int, ...] = (1, 7, 14)
    transaction_windows: tuple[int, ...] = (7, 28)
    horizon: int = PROJECT_HORIZON

    def __post_init__(self) -> None:
        sequences = (
            self.sales_lags,
            self.sales_windows,
            self.transaction_lags,
            self.transaction_windows,
        )
        resolve_policy(self.availability_policy)
        if self.horizon != PROJECT_HORIZON:
            raise FeatureContractError(
                f"Features must match the real {PROJECT_HORIZON}-day forecast horizon"
            )
        if any(not values or tuple(sorted(set(values))) != values for values in sequences):
            raise FeatureContractError("Lag and window values must be unique positive sequences")
        if any(value <= 0 for values in sequences for value in values):
            raise FeatureContractError("Lags and windows must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""

        return asdict(self)

    @property
    def policy(self) -> AvailabilityPolicy:
        return resolve_policy(self.availability_policy)

    @property
    def context_days(self) -> int:
        """Calendar days of history any feature can read, ending at the origin."""

        return max(
            *self.sales_lags,
            *self.sales_windows,
            *self.transaction_lags,
            *self.transaction_windows,
        )

    @property
    def sha256(self) -> str:
        """Hash the exact feature definition."""

        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class FeatureBatch:
    """Direct multi-horizon rows plus explicit model feature contracts."""

    frame: pd.DataFrame
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    origin: pd.Timestamp
    spec: FeatureSpec

    @property
    def feature_count(self) -> int:
        """Return the number of model inputs before categorical encoding."""

        return len(self.numeric_features) + len(self.categorical_features)


@dataclass(frozen=True)
class FittedPreprocessor:
    """Fold-fitted median imputation and categorical vocabularies."""

    fitted_through: pd.Timestamp
    numeric_medians: dict[str, float]
    categorical_vocabulary: dict[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize fitted statistics for a small manifest."""

        return {
            "fitted_through": self.fitted_through.date().isoformat(),
            "numeric_medians": self.numeric_medians,
            "categorical_vocabulary": {
                key: list(values) for key, values in self.categorical_vocabulary.items()
            },
        }

    @property
    def sha256(self) -> str:
        """Hash the exact fold-fitted imputation and vocabulary state."""

        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def _parse_dates(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    parsed = frame.copy()
    if TIME_COLUMN not in parsed:
        raise FeatureContractError(f"{name} must contain '{TIME_COLUMN}'")
    parsed[TIME_COLUMN] = pd.to_datetime(parsed[TIME_COLUMN], errors="raise").dt.normalize()
    return parsed


def _require_history_boundary(frame: pd.DataFrame, origin: pd.Timestamp, name: str) -> None:
    if frame.empty:
        raise FeatureContractError(f"{name} cannot be empty")
    if frame[TIME_COLUMN].max() > origin:
        raise FeatureContractError(f"{name} contains information after the forecast origin")


def _validate_target_grid(target: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> None:
    required = {*SERIES_KEYS, TIME_COLUMN, "onpromotion"}
    missing = required.difference(target.columns)
    if missing:
        raise FeatureContractError(f"Target grid is missing columns: {sorted(missing)}")
    if target.duplicated([TIME_COLUMN, *SERIES_KEYS]).any():
        raise FeatureContractError("Target store-family-date keys must be unique")
    expected_dates = pd.date_range(origin + pd.offsets.Day(1), periods=horizon, freq="D")
    actual_dates = pd.DatetimeIndex(sorted(target[TIME_COLUMN].unique()))
    if not actual_dates.equals(expected_dates):
        raise FeatureContractError("Target dates must be the contiguous horizon after origin")
    counts = target.groupby(list(SERIES_KEYS), observed=True)[TIME_COLUMN].nunique()
    if counts.empty or not counts.eq(horizon).all():
        raise FeatureContractError("Every store-family series must contain every horizon date")


def _sales_history_features(
    history: pd.DataFrame, origin: pd.Timestamp, spec: FeatureSpec
) -> pd.DataFrame:
    keys = list(SERIES_KEYS)
    series = history[keys].drop_duplicates().sort_values(keys)
    output = series.copy()
    indexed = history.set_index([*keys, TIME_COLUMN])[TARGET_COLUMN]

    for lag in spec.sales_lags:
        lookup_date = origin - pd.offsets.Day(lag - 1)
        values = indexed.reindex(
            pd.MultiIndex.from_frame(series.assign(date=lookup_date)[[*keys, TIME_COLUMN]])
        )
        column = f"sales_origin_lag_{lag}"
        output[column] = values.to_numpy()
        output[f"{column}_missing"] = output[column].isna().astype("int8")

    for window in spec.sales_windows:
        start = origin - pd.offsets.Day(window - 1)
        window_frame = history.loc[history[TIME_COLUMN].between(start, origin)]
        grouped = window_frame.groupby(keys, observed=True)[TARGET_COLUMN].agg(
            ["mean", "std", "min", "max", "count"]
        )
        zero_rate = (
            window_frame[TARGET_COLUMN]
            .eq(0)
            .groupby([window_frame[key] for key in keys], observed=True)
            .mean()
        )
        grouped[f"sales_roll_{window}_zero_rate"] = zero_rate
        grouped = grouped.rename(
            columns={
                "mean": f"sales_roll_{window}_mean",
                "std": f"sales_roll_{window}_std",
                "min": f"sales_roll_{window}_min",
                "max": f"sales_roll_{window}_max",
                "count": f"sales_roll_{window}_observed",
            }
        )
        output = output.merge(grouped.reset_index(), on=keys, how="left", validate="one_to_one")
    return output


def _transaction_history_features(
    transactions: pd.DataFrame, stores: pd.Series, origin: pd.Timestamp, spec: FeatureSpec
) -> pd.DataFrame:
    output = pd.DataFrame({"store_nbr": sorted(stores.unique())})
    indexed = transactions.set_index(["store_nbr", TIME_COLUMN])["transactions"]
    for lag in spec.transaction_lags:
        lookup_date = origin - pd.offsets.Day(lag - 1)
        lookup = output.assign(date=lookup_date)[["store_nbr", TIME_COLUMN]]
        values = indexed.reindex(pd.MultiIndex.from_frame(lookup))
        column = f"transactions_origin_lag_{lag}"
        output[column] = values.to_numpy()
        output[f"{column}_missing"] = output[column].isna().astype("int8")

    for window in spec.transaction_windows:
        start = origin - pd.offsets.Day(window - 1)
        window_frame = transactions.loc[transactions[TIME_COLUMN].between(start, origin)]
        grouped = window_frame.groupby("store_nbr", observed=True)["transactions"].agg(
            ["mean", "std", "count"]
        )
        grouped = grouped.rename(
            columns={
                "mean": f"transactions_roll_{window}_mean",
                "std": f"transactions_roll_{window}_std",
                "count": f"transactions_roll_{window}_observed",
            }
        )
        output = output.merge(
            grouped.reset_index(), on="store_nbr", how="left", validate="one_to_one"
        )
    return output


def build_store_holiday_calendar(
    stores: pd.DataFrame,
    holidays: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    policy: AvailabilityPolicy,
) -> pd.DataFrame:
    """Resolve allowed calendar rows by locale to one row per store-date."""

    store_frame = stores[["store_nbr", "city", "state"]].copy()
    holiday_frame = _parse_dates(holidays, "holidays")
    holiday_frame = holiday_frame.loc[
        holiday_frame[TIME_COLUMN].isin(dates)
        & ~holiday_frame["type"].isin(policy.excluded_calendar_types)
    ].reset_index(drop=True)
    unknown = ~holiday_frame["locale"].isin({"National", "Regional", "Local"})
    if unknown.any():
        raise FeatureContractError("Unsupported holiday locale")
    holiday_frame["source_row"] = np.arange(len(holiday_frame))
    parts = [
        holiday_frame.loc[holiday_frame["locale"].eq("National")].merge(
            store_frame[["store_nbr"]], how="cross"
        ),
        holiday_frame.loc[holiday_frame["locale"].eq("Regional")].merge(
            store_frame[["store_nbr", "state"]],
            left_on="locale_name",
            right_on="state",
            how="inner",
            validate="many_to_many",
        ),
        holiday_frame.loc[holiday_frame["locale"].eq("Local")].merge(
            store_frame[["store_nbr", "city"]],
            left_on="locale_name",
            right_on="city",
            how="inner",
            validate="many_to_many",
        ),
    ]
    applicable = pd.concat(
        [part[["source_row", "store_nbr", TIME_COLUMN, "type", "transferred"]] for part in parts],
        ignore_index=True,
    )
    if applicable.duplicated(["source_row", "store_nbr"]).any():
        raise FeatureContractError("A calendar source row applied twice to one store")
    transferred = applicable["transferred"].astype(bool)
    records = pd.DataFrame(
        {
            "store_nbr": applicable["store_nbr"],
            TIME_COLUMN: applicable[TIME_COLUMN],
            "holiday_event_count": 1,
            "is_holiday_event": (~transferred & applicable["type"].ne("Work Day")).astype(int),
            "is_work_day": applicable["type"].eq("Work Day").astype(int),
            "is_bridge": applicable["type"].eq("Bridge").astype(int),
            "is_transfer": applicable["type"].eq("Transfer").astype(int),
            "is_transferred_source": transferred.astype(int),
        }
    )

    calendar = pd.MultiIndex.from_product(
        [sorted(store_frame["store_nbr"].unique()), dates], names=["store_nbr", TIME_COLUMN]
    ).to_frame(index=False)
    value_columns = [
        "holiday_event_count",
        "is_holiday_event",
        "is_work_day",
        "is_bridge",
        "is_transfer",
        "is_transferred_source",
    ]
    if not records.empty:
        resolved = records.groupby(["store_nbr", TIME_COLUMN], as_index=False, observed=True).agg(
            holiday_event_count=("holiday_event_count", "sum"),
            is_holiday_event=("is_holiday_event", "max"),
            is_work_day=("is_work_day", "max"),
            is_bridge=("is_bridge", "max"),
            is_transfer=("is_transfer", "max"),
            is_transferred_source=("is_transferred_source", "max"),
        )
        calendar = calendar.merge(
            resolved, on=["store_nbr", TIME_COLUMN], how="left", validate="one_to_one"
        )
    for column in value_columns:
        if column not in calendar:
            calendar[column] = 0
        calendar[column] = calendar[column].fillna(0).astype("int8")
    if calendar.duplicated(["store_nbr", TIME_COLUMN]).any():
        raise FeatureContractError("Resolved holiday calendar is not unique by store-date")
    if len(calendar) != store_frame["store_nbr"].nunique() * len(dates):
        raise FeatureContractError("Resolved holiday calendar does not cover the store-date grid")
    return calendar


def build_request_feature_batch(
    request: ForecastRequest,
    spec: FeatureSpec,
    *,
    supported_series: tuple[tuple[int, str], ...],
    min_context_days: int = 1,
) -> FeatureBatch:
    """Validate a request, then build label-free direct rows under its policy.

    Validation runs here as well as at projection so that callers constructing a
    ForecastRequest directly (API, bundles) cannot bypass the input contract.
    """

    if request.availability_policy != spec.availability_policy:
        raise PolicyMismatchError("POLICY_MISMATCH: request and feature spec policies differ")
    validate_forecast_request(
        request, supported_series=supported_series, min_context_days=min_context_days
    )
    if request.horizon != spec.horizon:
        raise FeatureContractError("Request horizon differs from the feature specification")
    policy = spec.policy
    origin = request.origin
    history = request.sales_history
    transactions = request.transactions_history
    target = request.future_plan
    stores = request.stores
    _require_history_boundary(history, origin, "sales_history")
    if not transactions.empty:
        _require_history_boundary(transactions, origin, "transaction_history")
    _validate_target_grid(target, origin, spec.horizon)
    sales_lookback = max(max(spec.sales_lags), max(spec.sales_windows))
    transaction_lookback = max(max(spec.transaction_lags), max(spec.transaction_windows))
    history = history.loc[
        history[TIME_COLUMN] >= origin - pd.offsets.Day(sales_lookback - 1)
    ].copy()
    transactions = transactions.loc[
        transactions[TIME_COLUMN] >= origin - pd.offsets.Day(transaction_lookback - 1)
    ].copy()
    if history.duplicated([TIME_COLUMN, *SERIES_KEYS]).any():
        raise FeatureContractError("Sales history keys must be unique")
    if transactions.duplicated([TIME_COLUMN, "store_nbr"]).any():
        raise FeatureContractError("Transaction history keys must be unique")

    frame = target[[*SERIES_KEYS, TIME_COLUMN, "onpromotion"]].copy()
    frame = frame.rename(columns={TIME_COLUMN: "target_date"})
    frame["forecast_origin"] = origin
    frame["horizon"] = (frame["target_date"] - origin).dt.days.astype("int8")
    frame["onpromotion_log1p"] = np.log1p(frame["onpromotion"]).astype("float32")
    frame["is_promoted"] = frame["onpromotion"].gt(0).astype("int8")

    target_dates = frame["target_date"]
    frame["day_of_week"] = target_dates.dt.dayofweek.astype("int8")
    frame["day_of_month"] = target_dates.dt.day.astype("int8")
    frame["month"] = target_dates.dt.month.astype("int8")
    frame["is_weekend"] = target_dates.dt.dayofweek.ge(5).astype("int8")
    frame["is_month_start"] = target_dates.dt.is_month_start.astype("int8")
    frame["is_month_end"] = target_dates.dt.is_month_end.astype("int8")
    frame["is_payday"] = (target_dates.dt.day.eq(15) | target_dates.dt.is_month_end).astype("int8")
    day_angle = 2 * np.pi * target_dates.dt.dayofyear / 365.25
    frame["day_of_year_sin"] = np.sin(day_angle).astype("float32")
    frame["day_of_year_cos"] = np.cos(day_angle).astype("float32")

    store_columns = ["store_nbr", "city", "state", "type", "cluster"]
    frame = frame.merge(stores[store_columns], on="store_nbr", how="left", validate="many_to_one")
    if frame[store_columns[1:]].isna().any().any():
        raise FeatureContractError("Store metadata is incomplete")

    sales_features = _sales_history_features(history, origin, spec)
    frame = frame.merge(sales_features, on=list(SERIES_KEYS), how="left", validate="many_to_one")
    transaction_features = _transaction_history_features(
        transactions, frame["store_nbr"], origin, spec
    )
    frame = frame.merge(transaction_features, on="store_nbr", how="left", validate="many_to_one")

    if policy.includes_oil:
        oil_frame = request.competition_oil
        if oil_frame is None or oil_frame.duplicated(TIME_COLUMN).any():
            raise FeatureContractError("Competition oil must be present with unique dates")
        oil_target = oil_frame[[TIME_COLUMN, "dcoilwtico"]].rename(
            columns={TIME_COLUMN: "target_date", "dcoilwtico": "oil_target"}
        )
        frame = frame.merge(oil_target, on="target_date", how="left", validate="many_to_one")
        frame["oil_target_missing"] = frame["oil_target"].isna().astype("int8")
    elif request.competition_oil is not None:
        raise PolicyMismatchError("POLICY_MISMATCH: oil supplied to a policy that excludes it")

    holiday_calendar = build_store_holiday_calendar(
        stores,
        request.calendar,
        pd.DatetimeIndex(sorted(target[TIME_COLUMN].unique())),
        policy=policy,
    ).rename(columns={TIME_COLUMN: "target_date"})
    before_join = len(frame)
    frame = frame.merge(
        holiday_calendar,
        on=["store_nbr", "target_date"],
        how="left",
        validate="many_to_one",
    )
    if len(frame) != before_join:
        raise FeatureContractError("Holiday join changed row cardinality")

    categorical = ("store_nbr", "family", "city", "state", "type", "cluster")
    excluded = {*categorical, "forecast_origin", "target_date"}
    numeric = tuple(column for column in frame.columns if column not in excluded)
    frame = frame.sort_values(["target_date", *SERIES_KEYS]).reset_index(drop=True)
    return FeatureBatch(frame, numeric, categorical, origin, spec)


def build_fold_feature_batches(
    frames: dict[str, pd.DataFrame],
    fold: RollingOriginFold,
    spec: FeatureSpec,
    *,
    snapshot_id: str,
    data_source: str,
) -> tuple[FeatureBatch, FeatureBatch]:
    """Create a closed training window and the following validation horizon."""

    training_origin = fold.origin - pd.offsets.Day(fold.horizon)
    identity = {"snapshot_id": snapshot_id, "data_source": data_source}
    training = build_origin_feature_batch(frames, training_origin, spec, **identity)
    validation = build_origin_feature_batch(frames, fold.origin, spec, **identity)
    if training.frame["target_date"].max() > fold.origin:
        raise SplitContractError("Preprocessor training targets cross the validation origin")
    if validation.frame["target_date"].min() <= fold.origin:
        raise SplitContractError("Validation features overlap the training window")
    return training, validation


def project_benchmark_request(
    frames: dict[str, pd.DataFrame],
    origin: pd.Timestamp,
    spec: FeatureSpec,
    *,
    snapshot_id: str,
    data_source: str,
    supported_series: tuple[tuple[int, str], ...] | None = None,
) -> ForecastRequest:
    """Project raw benchmark frames to a validated request for this specification."""

    return project_snapshot(
        frames,
        snapshot_id=snapshot_id,
        data_source=data_source,
        origin=pd.Timestamp(origin).normalize(),
        policy=spec.availability_policy,
        context_days=spec.context_days,
        supported_series=supported_series,
        horizon=spec.horizon,
    )


def build_origin_feature_batch(
    frames: dict[str, pd.DataFrame],
    origin: pd.Timestamp,
    spec: FeatureSpec,
    *,
    snapshot_id: str,
    data_source: str,
    supported_series: tuple[tuple[int, str], ...] | None = None,
) -> FeatureBatch:
    """Build one supervised batch: request features, then a keyed label join."""

    request = project_benchmark_request(
        frames,
        origin,
        spec,
        snapshot_id=snapshot_id,
        data_source=data_source,
        supported_series=supported_series,
    )
    batch = build_request_feature_batch(
        request,
        spec,
        supported_series=request.series_keys if supported_series is None else supported_series,
    )
    labels = benchmark_labels(frames["train.csv"], request.target_dates)
    labeled = attach_labels(batch.frame, labels[["target_date", "store_nbr", "family", "sales"]])
    return FeatureBatch(
        labeled, batch.numeric_features, batch.categorical_features, batch.origin, spec
    )


def fit_preprocessor(batch: FeatureBatch, fitted_through: pd.Timestamp) -> FittedPreprocessor:
    """Fit medians and vocabularies only on rows whose targets are already observed."""

    cutoff = pd.Timestamp(fitted_through).normalize()
    if batch.frame["target_date"].max() > cutoff:
        raise FeatureContractError("Preprocessor fit rows extend beyond the training cutoff")
    medians: dict[str, float] = {}
    for column in batch.numeric_features:
        values = batch.frame[column].dropna()
        medians[column] = 0.0 if values.empty else float(values.median())
    vocabulary: dict[str, tuple[str, ...]] = {}
    for column in batch.categorical_features:
        observed = sorted(batch.frame[column].dropna().astype(str).unique())
        vocabulary[column] = (UNKNOWN_CATEGORY, *observed)
    return FittedPreprocessor(cutoff, medians, vocabulary)


def transform_features(batch: FeatureBatch, fitted: FittedPreprocessor) -> pd.DataFrame:
    """Apply train-only statistics; unseen categories map to code zero."""

    output = batch.frame[["forecast_origin", "target_date", "horizon"]].copy()
    if TARGET_COLUMN in batch.frame:
        output[TARGET_COLUMN] = batch.frame[TARGET_COLUMN].astype("float32")
    for column in batch.numeric_features:
        output[column] = (
            batch.frame[column].fillna(fitted.numeric_medians[column]).astype("float32")
        )
    for column in batch.categorical_features:
        mapping = {
            value: index for index, value in enumerate(fitted.categorical_vocabulary[column])
        }
        output[f"cat__{column}"] = (
            batch.frame[column].astype(str).map(mapping).fillna(0).astype("int32")
        )
    if output.isna().any().any():
        raise FeatureContractError("Transformed model inputs must not contain missing values")
    return output


def model_input_contract(batch: FeatureBatch) -> dict[str, Any]:
    """Describe tabular and recurrent consumers without materializing model arrays."""

    known_future = [
        "horizon",
        "onpromotion",
        "onpromotion_log1p",
        "is_promoted",
        "day_of_week",
        "day_of_month",
        "month",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "is_payday",
        "day_of_year_sin",
        "day_of_year_cos",
        *(("oil_target", "oil_target_missing") if batch.spec.policy.includes_oil else ()),
        "holiday_event_count",
        "is_holiday_event",
        "is_work_day",
        "is_bridge",
        "is_transfer",
        "is_transferred_source",
    ]
    history_features = [
        column
        for column in batch.numeric_features
        if column.startswith(("sales_", "transactions_"))
    ]
    payload = {
        "availability_policy": batch.spec.availability_policy,
        "input_contract_version": batch.spec.policy.input_contract_version,
        "row_key": [*SERIES_KEYS, "forecast_origin", "target_date", "horizon"],
        "target": TARGET_COLUMN,
        "target_dtype": "float32_non_negative",
        "tabular": {
            "grain": "one row per series-origin-horizon",
            "numeric_features": list(batch.numeric_features),
            "categorical_features": list(batch.categorical_features),
        },
        "recurrent": {
            "history_anchor": "forecast_origin",
            "lookback_days": max(batch.spec.sales_windows),
            "history_tensor": {
                "shape": ["batch", "lookback_days", "dynamic_history_features"],
                "features": [
                    "sales",
                    "onpromotion",
                    "transactions",
                    "sales_observed",
                    "transactions_observed",
                ],
                "must_end_at_origin": True,
            },
            "history_summary_features": history_features,
            "known_future_features": known_future,
            "known_future_tensor_shape": ["batch", PROJECT_HORIZON, len(known_future)],
            "static_categorical_features": list(batch.categorical_features),
            "target_shape": ["batch", PROJECT_HORIZON],
            "history_must_end_at_origin": True,
        },
    }
    return payload


def feature_manifest(
    spec: FeatureSpec,
    split_sha256: str,
    fold_runs: list[dict[str, Any]],
    example_batch: FeatureBatch,
    run_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build small feature metadata without exporting the transformed panel."""

    payload = {
        "availability_policy": spec.availability_policy,
        "input_contract_version": spec.policy.input_contract_version,
        "feature_spec": spec.to_dict(),
        "feature_spec_sha256": spec.sha256,
        "split_manifest_sha256": split_sha256,
        "feature_count": example_batch.feature_count,
        "numeric_feature_count": len(example_batch.numeric_features),
        "categorical_feature_count": len(example_batch.categorical_features),
        "fold_runs": fold_runs,
        "model_input_contract": model_input_contract(example_batch),
        "processed_dataset_exported": False,
    }
    if run_evidence is not None:
        payload["run_evidence"] = run_evidence
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {**payload, "sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def export_feature_manifest(manifest: dict[str, Any], path: Path) -> Path:
    """Write feature metadata only."""

    write_json(path, manifest)
    return path
