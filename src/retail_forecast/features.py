"""Origin-anchored features that cannot read targets inside a forecast horizon."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from retail_forecast.config import PROJECT_HORIZON
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
    """Pre-registered feature set shared by smoke and full runs."""

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
    stores: pd.DataFrame, holidays: pd.DataFrame, dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Resolve holiday locale and semantics to one row per store-date."""

    store_frame = stores.copy()
    holiday_frame = _parse_dates(holidays, "holidays")
    holiday_frame = holiday_frame.loc[holiday_frame[TIME_COLUMN].isin(dates)]
    records: list[dict[str, Any]] = []
    for row in holiday_frame.itertuples(index=False):
        if row.locale == "National":
            applicable = store_frame
        elif row.locale == "Regional":
            applicable = store_frame.loc[store_frame["state"] == row.locale_name]
        elif row.locale == "Local":
            applicable = store_frame.loc[store_frame["city"] == row.locale_name]
        else:
            raise FeatureContractError(f"Unsupported holiday locale: {row.locale}")
        for store_nbr in applicable["store_nbr"]:
            active = not bool(row.transferred) and row.type != "Work Day"
            records.append(
                {
                    "store_nbr": store_nbr,
                    TIME_COLUMN: row.date,
                    "holiday_event_count": 1,
                    "is_holiday_event": int(active),
                    "is_work_day": int(row.type == "Work Day"),
                    "is_bridge": int(row.type == "Bridge"),
                    "is_transfer": int(row.type == "Transfer"),
                    "is_transferred_source": int(bool(row.transferred)),
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
    if records:
        resolved = (
            pd.DataFrame(records)
            .groupby(["store_nbr", TIME_COLUMN], as_index=False, observed=True)
            .agg(
                holiday_event_count=("holiday_event_count", "sum"),
                is_holiday_event=("is_holiday_event", "max"),
                is_work_day=("is_work_day", "max"),
                is_bridge=("is_bridge", "max"),
                is_transfer=("is_transfer", "max"),
                is_transferred_source=("is_transferred_source", "max"),
            )
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
    return calendar


def build_feature_batch(
    sales_history: pd.DataFrame,
    target_grid: pd.DataFrame,
    stores: pd.DataFrame,
    oil: pd.DataFrame,
    holidays: pd.DataFrame,
    transaction_history: pd.DataFrame,
    origin: pd.Timestamp,
    spec: FeatureSpec,
) -> FeatureBatch:
    """Build direct rows using target-date known covariates and origin-only histories."""

    origin = pd.Timestamp(origin).normalize()
    history = _parse_dates(sales_history, "sales_history")
    target = _parse_dates(target_grid, "target_grid")
    oil_frame = _parse_dates(oil, "oil")
    transactions = _parse_dates(transaction_history, "transaction_history")
    _require_history_boundary(history, origin, "sales_history")
    _require_history_boundary(transactions, origin, "transaction_history")
    _validate_target_grid(target, origin, spec.horizon)
    if history.duplicated([TIME_COLUMN, *SERIES_KEYS]).any():
        raise FeatureContractError("Sales history keys must be unique")
    if transactions.duplicated([TIME_COLUMN, "store_nbr"]).any():
        raise FeatureContractError("Transaction history keys must be unique")

    frame = target[[*SERIES_KEYS, TIME_COLUMN, "onpromotion"]].copy()
    if TARGET_COLUMN in target:
        frame[TARGET_COLUMN] = target[TARGET_COLUMN].to_numpy()
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

    if oil_frame.duplicated(TIME_COLUMN).any():
        raise FeatureContractError("Oil dates must be unique")
    oil_target = oil_frame[[TIME_COLUMN, "dcoilwtico"]].rename(
        columns={TIME_COLUMN: "target_date", "dcoilwtico": "oil_target"}
    )
    frame = frame.merge(oil_target, on="target_date", how="left", validate="many_to_one")
    frame["oil_target_missing"] = frame["oil_target"].isna().astype("int8")

    holiday_calendar = build_store_holiday_calendar(
        stores, holidays, pd.DatetimeIndex(sorted(target[TIME_COLUMN].unique()))
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
    excluded = {
        *categorical,
        TARGET_COLUMN,
        "forecast_origin",
        "target_date",
    }
    numeric = tuple(column for column in frame.columns if column not in excluded)
    frame = frame.sort_values(["target_date", *SERIES_KEYS]).reset_index(drop=True)
    return FeatureBatch(frame, numeric, categorical, origin, spec)


def build_fold_feature_batches(
    frames: dict[str, pd.DataFrame], fold: RollingOriginFold, spec: FeatureSpec
) -> tuple[FeatureBatch, FeatureBatch]:
    """Create a closed training window and the following validation horizon."""

    train = _parse_dates(frames["train.csv"], "train.csv")
    transactions = _parse_dates(frames["transactions.csv"], "transactions.csv")
    training_origin = fold.origin - pd.offsets.Day(fold.horizon)

    def make(origin: pd.Timestamp) -> FeatureBatch:
        target_dates = pd.date_range(origin + pd.offsets.Day(1), periods=spec.horizon, freq="D")
        target = train.loc[train[TIME_COLUMN].isin(target_dates)].copy()
        history = train.loc[train[TIME_COLUMN] <= origin].copy()
        transaction_history = transactions.loc[transactions[TIME_COLUMN] <= origin].copy()
        return build_feature_batch(
            history,
            target,
            frames["stores.csv"],
            frames["oil.csv"],
            frames["holidays_events.csv"],
            transaction_history,
            origin,
            spec,
        )

    training = make(training_origin)
    validation = make(fold.origin)
    if training.frame["target_date"].max() > fold.origin:
        raise SplitContractError("Preprocessor training targets cross the validation origin")
    if validation.frame["target_date"].min() <= fold.origin:
        raise SplitContractError("Validation features overlap the training window")
    return training, validation


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
        "oil_target",
        "oil_target_missing",
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
