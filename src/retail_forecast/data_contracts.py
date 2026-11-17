"""Normalized ForecastRequest v1, strict validation and benchmark projection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from retail_forecast.availability import (
    INPUT_CONTRACT_VERSION,
    PolicyMismatchError,
    resolve_policy,
)
from retail_forecast.config import PROJECT_HORIZON

SeriesKey = tuple[int, str]

ERROR_CODES: Final = frozenset(
    {
        "SCHEMA_INVALID",
        "UNSUPPORTED_SERIES",
        "HISTORY_INSUFFICIENT",
        "HISTORY_AFTER_ORIGIN",
        "FUTURE_PLAN_INCOMPLETE",
        "POLICY_MISMATCH",
        "NONFINITE_VALUE",
    }
)
CALENDAR_LOCALES: Final = frozenset({"National", "Regional", "Local"})
CALENDAR_TYPES: Final = frozenset(
    {"Holiday", "Event", "Additional", "Transfer", "Bridge", "Work Day"}
)
MAX_KEY_TEXT: Final = 64

TABLE_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "sales_history": ("date", "store_nbr", "family", "sales", "onpromotion"),
    "transactions_history": ("date", "store_nbr", "transactions"),
    "future_plan": ("date", "store_nbr", "family", "onpromotion"),
    "stores": ("store_nbr", "city", "state", "type", "cluster"),
    "calendar": ("date", "type", "locale", "locale_name", "description", "transferred"),
    "competition_oil": ("date", "dcoilwtico"),
}
TABLE_KEYS: Final[dict[str, tuple[str, ...]]] = {
    "sales_history": ("date", "store_nbr", "family"),
    "transactions_history": ("date", "store_nbr"),
    "future_plan": ("date", "store_nbr", "family"),
    "stores": ("store_nbr",),
    "calendar": ("date", "type", "locale", "locale_name", "description"),
    "competition_oil": ("date",),
}
TEXT_COLUMNS: Final = frozenset(
    {"family", "city", "state", "type", "locale", "locale_name", "description"}
)
INTEGER_COLUMNS: Final = frozenset({"store_nbr", "onpromotion", "transactions", "cluster"})
FLOAT_COLUMNS: Final = frozenset({"sales"})


class ForecastRequestError(ValueError):
    """Validation failure carrying a contract category and safe context only."""

    def __init__(self, code: str, field: str, key: object | None = None) -> None:
        if code not in ERROR_CODES:
            raise ValueError("Unknown forecast request error code")
        self.code = code
        self.field = field
        self.key = key
        detail = f"{code}: {field}"
        if key is not None:
            detail += f" key={key}"
        super().__init__(detail)


@dataclass(frozen=True, eq=False)
class ForecastRequest:
    """Inputs for one origin; tables never include target-period labels."""

    input_contract_version: int
    snapshot_id: str
    data_source: str
    availability_policy: str
    origin: pd.Timestamp
    horizon: int
    series_keys: tuple[SeriesKey, ...]
    sales_history: pd.DataFrame
    transactions_history: pd.DataFrame
    future_plan: pd.DataFrame
    stores: pd.DataFrame
    calendar: pd.DataFrame
    competition_oil: pd.DataFrame | None = None
    history_start: pd.Timestamp | None = None

    @property
    def target_dates(self) -> pd.DatetimeIndex:
        return pd.date_range(self.origin + pd.offsets.Day(1), periods=self.horizon, freq="D")


def _safe_key(row: pd.Series, columns: Iterable[str]) -> str:
    parts = []
    for column in columns:
        value = row[column]
        if isinstance(value, pd.Timestamp):
            value = value.date().isoformat()
        text = str(value)
        value = text if len(text) <= MAX_KEY_TEXT else f"{text[:MAX_KEY_TEXT]}..."
        parts.append(f"{column}={value}")
    return ",".join(parts)


def _check_table(name: str, frame: object) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ForecastRequestError("SCHEMA_INVALID", name)
    expected = TABLE_COLUMNS[name]
    if tuple(frame.columns) != expected:
        raise ForecastRequestError("SCHEMA_INVALID", f"{name}.columns")
    for column in expected:
        values = frame[column]
        field = f"{name}.{column}"
        if column == "date":
            if not pd.api.types.is_datetime64_dtype(values) or getattr(values.dt, "tz", None):
                raise ForecastRequestError("SCHEMA_INVALID", field)
            if values.isna().any() or not values.eq(values.dt.normalize()).all():
                raise ForecastRequestError("SCHEMA_INVALID", field)
        elif column in TEXT_COLUMNS:
            if values.isna().any() or values.astype(str).str.len().eq(0).any():
                raise ForecastRequestError("SCHEMA_INVALID", field)
        elif column == "transferred":
            if not pd.api.types.is_bool_dtype(values):
                raise ForecastRequestError("SCHEMA_INVALID", field)
        elif column in INTEGER_COLUMNS | FLOAT_COLUMNS | {"dcoilwtico"}:
            if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
                raise ForecastRequestError("SCHEMA_INVALID", field)
            numeric = values.to_numpy(dtype=np.float64)
            if column == "dcoilwtico":
                numeric = numeric[~np.isnan(numeric)]
            if not np.isfinite(numeric).all():
                raise ForecastRequestError("NONFINITE_VALUE", field)
            if column != "dcoilwtico" and (numeric < 0).any():
                raise ForecastRequestError("SCHEMA_INVALID", field)
            if column in INTEGER_COLUMNS and not np.equal(np.floor(numeric), numeric).all():
                raise ForecastRequestError("SCHEMA_INVALID", field)
    keys = TABLE_KEYS.get(name)
    if keys:
        duplicated = frame.duplicated(list(keys))
        if duplicated.any():
            row = frame.loc[duplicated].iloc[0]
            raise ForecastRequestError("SCHEMA_INVALID", f"{name}.key", _safe_key(row, keys))
    return frame


def validate_forecast_request(
    request: ForecastRequest,
    *,
    supported_series: Iterable[SeriesKey],
    min_context_days: int = 1,
) -> None:
    """Reject malformed, leaking, incomplete or policy-inconsistent requests."""

    if request.input_contract_version != INPUT_CONTRACT_VERSION:
        raise ForecastRequestError("SCHEMA_INVALID", "input_contract_version")
    try:
        policy = resolve_policy(request.availability_policy)
    except PolicyMismatchError as error:
        raise ForecastRequestError("POLICY_MISMATCH", "availability_policy") from error
    if policy.includes_oil != (request.competition_oil is not None):
        raise ForecastRequestError("POLICY_MISMATCH", "competition_oil")
    if request.data_source not in {"kaggle", "synthetic"} or not request.snapshot_id:
        raise ForecastRequestError("SCHEMA_INVALID", "snapshot identity")
    if request.horizon != PROJECT_HORIZON:
        raise ForecastRequestError("SCHEMA_INVALID", "horizon")
    origin = request.origin
    if not isinstance(origin, pd.Timestamp) or origin != origin.normalize() or origin.tz:
        raise ForecastRequestError("SCHEMA_INVALID", "origin")
    if min_context_days <= 0:
        raise ValueError("min_context_days must be positive")

    keys = request.series_keys
    if (
        not isinstance(keys, tuple)
        or not keys
        or len(set(keys)) != len(keys)
        or any(
            not isinstance(key, tuple)
            or len(key) != 2
            or isinstance(key[0], bool)
            or not isinstance(key[0], (int, np.integer))
            or not isinstance(key[1], str)
            for key in keys
        )
    ):
        raise ForecastRequestError("SCHEMA_INVALID", "series_keys")
    supported = {(int(store), str(family)) for store, family in supported_series}
    for store, family in keys:
        if (int(store), family) not in supported:
            raise ForecastRequestError(
                "UNSUPPORTED_SERIES", "series_keys", f"store_nbr={store},family={family}"
            )

    sales = _check_table("sales_history", request.sales_history)
    transactions = _check_table("transactions_history", request.transactions_history)
    plan = _check_table("future_plan", request.future_plan)
    stores = _check_table("stores", request.stores)
    calendar = _check_table("calendar", request.calendar)
    if request.competition_oil is not None:
        _check_table("competition_oil", request.competition_oil)
    if not calendar["locale"].isin(CALENDAR_LOCALES).all():
        raise ForecastRequestError("SCHEMA_INVALID", "calendar.locale")
    if not calendar["type"].isin(CALENDAR_TYPES).all():
        raise ForecastRequestError("SCHEMA_INVALID", "calendar.type")

    for name, frame in (("sales_history", sales), ("transactions_history", transactions)):
        late = frame["date"] > origin
        if late.any():
            row = frame.loc[late].iloc[0]
            raise ForecastRequestError(
                "HISTORY_AFTER_ORIGIN", name, _safe_key(row, TABLE_KEYS[name])
            )
    history_start = sales["date"].min()
    if request.history_start is not None:
        declared = request.history_start
        if (
            not isinstance(declared, pd.Timestamp)
            or declared.tz is not None
            or pd.isna(declared)
            or declared != declared.normalize()
            or declared > origin
            or (not sales.empty and declared > sales["date"].min())
        ):
            raise ForecastRequestError("SCHEMA_INVALID", "history_start")
        history_start = declared
    if sales.empty or (origin - history_start).days + 1 < min_context_days:
        raise ForecastRequestError("HISTORY_INSUFFICIENT", "sales_history")

    requested = pd.DataFrame(keys, columns=["store_nbr", "family"])
    history_keys = sales[["store_nbr", "family"]].drop_duplicates()
    extra = history_keys.merge(requested, how="left", indicator=True)
    if extra["_merge"].eq("left_only").any():
        raise ForecastRequestError("SCHEMA_INVALID", "sales_history.series")

    expected = requested.merge(pd.DataFrame({"date": request.target_dates}), how="cross")
    joined = plan.merge(expected, on=["date", "store_nbr", "family"], how="outer", indicator=True)
    if not joined["_merge"].eq("both").all():
        row = joined.loc[joined["_merge"].ne("both")].iloc[0]
        raise ForecastRequestError(
            "FUTURE_PLAN_INCOMPLETE",
            "future_plan",
            _safe_key(row, ("date", "store_nbr", "family")),
        )

    missing_stores = set(requested["store_nbr"]) - set(stores["store_nbr"])
    if missing_stores:
        raise ForecastRequestError("SCHEMA_INVALID", "stores", f"store_nbr={min(missing_stores)}")


def benchmark_series(train: pd.DataFrame) -> tuple[SeriesKey, ...]:
    """Return sorted known series of a benchmark panel."""

    series = train[["store_nbr", "family"]].drop_duplicates().sort_values(["store_nbr", "family"])
    return tuple((int(row.store_nbr), str(row.family)) for row in series.itertuples(index=False))


def _dated(frame: pd.DataFrame) -> pd.DataFrame:
    if pd.api.types.is_datetime64_dtype(frame["date"]):
        return frame
    parsed = frame.copy()
    parsed["date"] = pd.to_datetime(parsed["date"], format="%Y-%m-%d", errors="raise")
    return parsed


def project_snapshot(
    frames: Mapping[str, pd.DataFrame],
    *,
    snapshot_id: str,
    data_source: str,
    origin: pd.Timestamp,
    policy: str,
    context_days: int,
    min_context_days: int = 1,
    series_keys: tuple[SeriesKey, ...] | None = None,
    supported_series: Iterable[SeriesKey] | None = None,
    horizon: int = PROJECT_HORIZON,
) -> ForecastRequest:
    """Project raw benchmark frames to one validated, label-free request."""

    resolved = resolve_policy(policy)
    origin = pd.Timestamp(origin).normalize()
    train = _dated(frames["train.csv"])
    transactions = _dated(frames["transactions.csv"])
    known = benchmark_series(train) if supported_series is None else tuple(supported_series)
    keys = known if series_keys is None else series_keys
    start = origin - pd.offsets.Day(context_days - 1)
    targets = pd.date_range(origin + pd.offsets.Day(1), periods=horizon, freq="D")
    selected = pd.MultiIndex.from_tuples(keys, names=["store_nbr", "family"])

    def requested(frame: pd.DataFrame) -> pd.DataFrame:
        if series_keys is None:
            return frame
        index = pd.MultiIndex.from_frame(frame[["store_nbr", "family"]])
        return frame.loc[index.isin(selected)]

    history = requested(train.loc[train["date"].between(start, origin)])
    plan_source = requested(train.loc[train["date"].isin(targets)])
    if "test.csv" in frames and plan_source.empty:
        plan_source = requested(_dated(frames["test.csv"]))
        plan_source = plan_source.loc[plan_source["date"].isin(targets)]
    stores_requested = {store for store, _ in keys}
    calendar = _dated(frames["holidays_events.csv"])
    oil = None
    if resolved.includes_oil:
        oil = _dated(frames["oil.csv"])[list(TABLE_COLUMNS["competition_oil"])].reset_index(
            drop=True
        )
    request = ForecastRequest(
        input_contract_version=resolved.input_contract_version,
        snapshot_id=snapshot_id,
        data_source=data_source,
        availability_policy=resolved.name,
        origin=origin,
        horizon=horizon,
        series_keys=keys,
        sales_history=history[list(TABLE_COLUMNS["sales_history"])].reset_index(drop=True),
        transactions_history=transactions.loc[
            transactions["date"].between(start, origin)
            & transactions["store_nbr"].isin(stores_requested),
            list(TABLE_COLUMNS["transactions_history"]),
        ].reset_index(drop=True),
        future_plan=plan_source[list(TABLE_COLUMNS["future_plan"])].reset_index(drop=True),
        stores=frames["stores.csv"][list(TABLE_COLUMNS["stores"])].reset_index(drop=True),
        calendar=calendar.loc[
            calendar["date"].isin(targets), list(TABLE_COLUMNS["calendar"])
        ].reset_index(drop=True),
        competition_oil=oil,
        history_start=max(start, train["date"].min()),
    )
    validate_forecast_request(request, supported_series=known, min_context_days=min_context_days)
    return request


def benchmark_labels(train: pd.DataFrame, target_dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    """Return keyed observed labels for scoring, separate from model inputs."""

    frame = _dated(train)
    labels = frame.loc[
        frame["date"].isin(pd.DatetimeIndex(target_dates)),
        ["id", "date", "store_nbr", "family", "sales"],
    ].rename(columns={"date": "target_date"})
    if labels.duplicated(["target_date", "store_nbr", "family"]).any():
        raise ForecastRequestError("SCHEMA_INVALID", "labels.key")
    if not np.isfinite(labels["sales"].to_numpy(dtype=np.float64)).all():
        raise ForecastRequestError("NONFINITE_VALUE", "labels.sales")
    return labels.reset_index(drop=True)


def attach_labels(frame: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join labels by key and fail on any missing or duplicate key."""

    key = ["target_date", "store_nbr", "family"]
    if labels.duplicated(key).any():
        raise ForecastRequestError("SCHEMA_INVALID", "labels.key")
    joined = frame.merge(labels, on=key, how="left", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        row = joined.loc[joined["_merge"].ne("both")].iloc[0]
        raise ForecastRequestError("SCHEMA_INVALID", "labels.missing", _safe_key(row, key))
    return joined.drop(columns="_merge")


FORECAST_COLUMNS: Final = (
    "store_nbr",
    "family",
    "forecast_origin",
    "target_date",
    "horizon",
    "prediction",
    "model_id",
    "model_version",
    "bundle_id",
    "training_run_id",
    "prediction_run_id",
    "evaluation_run_id",
    "snapshot_id",
    "availability_policy",
    "input_contract_version",
    "data_source",
    "execution_mode",
)
FORECAST_ORDER: Final = ("store_nbr", "family", "target_date")
_LINEAGE_TEXT: Final = (
    "model_id",
    "model_version",
    "bundle_id",
    "training_run_id",
    "prediction_run_id",
)


def validate_forecast_frame(
    frame: pd.DataFrame, *, request: ForecastRequest, expected_bundle_id: str
) -> None:
    """Accept only the exact requested grid with finite, attributed, nonnegative values."""

    if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != FORECAST_COLUMNS:
        raise ForecastRequestError("SCHEMA_INVALID", "forecast.columns")
    expected = pd.DataFrame(request.series_keys, columns=["store_nbr", "family"]).merge(
        pd.DataFrame({"target_date": request.target_dates}), how="cross"
    )
    if frame.duplicated(list(FORECAST_ORDER)).any():
        row = frame.loc[frame.duplicated(list(FORECAST_ORDER))].iloc[0]
        raise ForecastRequestError("SCHEMA_INVALID", "forecast.key", _safe_key(row, FORECAST_ORDER))
    joined = frame[list(FORECAST_ORDER)].merge(
        expected, on=list(FORECAST_ORDER), how="outer", indicator=True
    )
    if not joined["_merge"].eq("both").all():
        row = joined.loc[joined["_merge"].ne("both")].iloc[0]
        raise ForecastRequestError(
            "SCHEMA_INVALID", "forecast.grid", _safe_key(row, FORECAST_ORDER)
        )
    if not frame["forecast_origin"].eq(request.origin).all():
        raise ForecastRequestError("SCHEMA_INVALID", "forecast.forecast_origin")
    horizon = (frame["target_date"] - frame["forecast_origin"]).dt.days
    if (
        not frame["horizon"].astype("int64").eq(horizon).all()
        or not horizon.between(1, request.horizon).all()
    ):
        raise ForecastRequestError("SCHEMA_INVALID", "forecast.horizon")
    values = frame["prediction"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ForecastRequestError("NONFINITE_VALUE", "forecast.prediction")
    if (values < 0).any():
        raise ForecastRequestError("SCHEMA_INVALID", "forecast.prediction")
    for column in _LINEAGE_TEXT:
        if frame[column].isna().any() or frame[column].astype(str).str.len().eq(0).any():
            raise ForecastRequestError("SCHEMA_INVALID", f"forecast.{column}")
        if frame[column].nunique() != 1:
            raise ForecastRequestError("SCHEMA_INVALID", f"forecast.{column}")
    evaluation = frame["evaluation_run_id"]
    if evaluation.notna().any() and (evaluation.isna().any() or evaluation.nunique() != 1):
        raise ForecastRequestError("SCHEMA_INVALID", "forecast.evaluation_run_id")
    if not frame["bundle_id"].eq(expected_bundle_id).all():
        raise ForecastRequestError("SCHEMA_INVALID", "forecast.bundle_id")
    for column, value in (
        ("snapshot_id", request.snapshot_id),
        ("availability_policy", request.availability_policy),
        ("input_contract_version", request.input_contract_version),
        ("data_source", request.data_source),
    ):
        if not frame[column].eq(value).all():
            raise ForecastRequestError(
                "POLICY_MISMATCH" if column == "availability_policy" else "SCHEMA_INVALID",
                f"forecast.{column}",
            )
    if not frame["execution_mode"].isin(["smoke", "full"]).all() or (
        frame["execution_mode"].nunique() != 1
    ):
        raise ForecastRequestError("SCHEMA_INVALID", "forecast.execution_mode")
    ordered = frame[list(FORECAST_ORDER)].reset_index(drop=True)
    if not ordered.equals(
        ordered.sort_values(list(FORECAST_ORDER), kind="mergesort").reset_index(drop=True)
    ):
        raise ForecastRequestError("SCHEMA_INVALID", "forecast.order")


COMPLETE_LINEAGE_COLUMNS: Final = (
    "model_id",
    "model_version",
    "bundle_id",
    "training_run_id",
    "prediction_run_id",
    "evaluation_run_id",
    "snapshot_id",
    "availability_policy",
    "input_contract_version",
    "data_source",
    "execution_mode",
)


class LineageIneligibleError(ValueError):
    """A table lacks the producer identities required for complete-record publication."""

    def __init__(self, missing: Iterable[str]) -> None:
        self.missing = tuple(sorted(missing))
        super().__init__(
            "LINEAGE_INCOMPLETE: missing producer identities " + ", ".join(self.missing)
        )


def require_complete_lineage(frame: pd.DataFrame) -> None:
    """Admit only scored rows whose every producer identity is present and non-null.

    Research-only rows (D2-01 `research_partial` OOF, zero sanity controls) fail
    here; they must be regenerated through verified bundles, never patched.
    """

    missing = {column for column in COMPLETE_LINEAGE_COLUMNS if column not in frame}
    missing |= {
        column
        for column in COMPLETE_LINEAGE_COLUMNS
        if column in frame and (frame[column].isna().any() or frame[column].eq("").any())
    }
    if "lineage_status" not in frame or not frame["lineage_status"].eq("complete").all():
        missing.add("lineage_status=complete")
    if missing:
        raise LineageIneligibleError(missing)
