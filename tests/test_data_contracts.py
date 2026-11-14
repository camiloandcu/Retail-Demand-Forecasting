"""ForecastRequest v1 validation, projection boundaries and keyed label joins."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from retail_forecast.config import ProjectConfig
from retail_forecast.data import load_frames
from retail_forecast.data_contracts import (
    ForecastRequestError,
    attach_labels,
    benchmark_labels,
    project_snapshot,
    validate_forecast_request,
)
from retail_forecast.features import FeatureSpec, build_request_feature_batch

ORIGIN = pd.Timestamp("2020-02-01")
SECRET = "sk-test-SENTINEL-do-not-print"


@pytest.fixture
def frames(synthetic_config: ProjectConfig):
    return load_frames(synthetic_config.paths.raw_data_dir)


def request_for(frames, policy="asof_v1", **overrides):
    request = project_snapshot(
        frames,
        snapshot_id="b" * 64,
        data_source="synthetic",
        origin=ORIGIN,
        policy=policy,
        context_days=56,
    )
    return replace(request, **overrides) if overrides else request


def code_of(request, **kwargs) -> str:
    with pytest.raises(ForecastRequestError) as error:
        validate_forecast_request(
            request, supported_series=kwargs.get("supported", request.series_keys)
        )
    assert SECRET not in str(error.value)
    return error.value.code


def test_minimal_asof_request_needs_no_benchmark_only_inputs(frames) -> None:
    minimal = {
        name: frames[name]
        for name in ("train.csv", "transactions.csv", "stores.csv", "holidays_events.csv")
    }
    request = request_for(minimal)
    assert request.competition_oil is None
    assert "sales" not in request.future_plan.columns
    batch = build_request_feature_batch(
        request, FeatureSpec("asof_v1"), supported_series=request.series_keys
    )
    assert len(batch.frame) == len(request.series_keys) * 16
    assert "sales" not in batch.frame.columns


def test_policy_and_oil_must_agree(frames) -> None:
    asof = request_for(frames)
    assert (
        code_of(
            replace(
                asof,
                competition_oil=frames["oil.csv"].assign(
                    date=pd.to_datetime(frames["oil.csv"]["date"])
                ),
            )
        )
        == "POLICY_MISMATCH"
    )
    assert code_of(replace(asof, availability_policy="realtime_v9")) == "POLICY_MISMATCH"
    competition = request_for(frames, "competition_v1")
    assert code_of(replace(competition, competition_oil=None)) == "POLICY_MISMATCH"


def test_post_origin_history_is_rejected(frames) -> None:
    request = request_for(frames)
    for field in ("sales_history", "transactions_history"):
        table = getattr(request, field)
        late = table.tail(1).assign(date=ORIGIN + pd.offsets.Day(1))
        leaked = replace(request, **{field: pd.concat([table, late], ignore_index=True)})
        assert code_of(leaked) == "HISTORY_AFTER_ORIGIN"


def test_incomplete_or_extra_plan_is_rejected_without_zero_fill(frames) -> None:
    request = request_for(frames)
    assert code_of(replace(request, future_plan=request.future_plan.iloc[1:])) == (
        "FUTURE_PLAN_INCOMPLETE"
    )
    extra = request.future_plan.tail(1).assign(date=ORIGIN + pd.offsets.Day(17))
    assert (
        code_of(
            replace(request, future_plan=pd.concat([request.future_plan, extra], ignore_index=True))
        )
        == "FUTURE_PLAN_INCOMPLETE"
    )
    labeled = request.future_plan.assign(sales=1.0)
    assert code_of(replace(request, future_plan=labeled)) == "SCHEMA_INVALID"


@pytest.mark.parametrize(
    ("field", "column"),
    [
        ("sales_history", "sales"),
        ("sales_history", "onpromotion"),
        ("transactions_history", "transactions"),
        ("future_plan", "onpromotion"),
    ],
)
@pytest.mark.parametrize("value", [np.inf, np.nan])
def test_nonfinite_values_are_rejected(frames, field, column, value) -> None:
    request = request_for(frames)
    table = getattr(request, field).copy()
    table[column] = table[column].astype(float)
    table.loc[table.index[0], column] = value
    assert code_of(replace(request, **{field: table})) == "NONFINITE_VALUE"


def test_unknown_series_duplicate_keys_and_invalid_schema(frames) -> None:
    request = request_for(frames)
    unknown = ((99, "SYNTHETIC_FAMILY_01"),)
    widened = replace(request, series_keys=request.series_keys + unknown)
    assert code_of(widened, supported=request.series_keys) == "UNSUPPORTED_SERIES"
    duplicated = pd.concat([request.sales_history, request.sales_history.head(1)])
    assert code_of(replace(request, sales_history=duplicated)) == "SCHEMA_INVALID"
    with_column = request.sales_history.assign(note=SECRET)
    assert code_of(replace(request, sales_history=with_column)) == "SCHEMA_INVALID"
    string_dates = request.sales_history.assign(date=request.sales_history["date"].astype(str))
    assert code_of(replace(request, sales_history=string_dates)) == "SCHEMA_INVALID"
    timed = request.sales_history.assign(date=request.sales_history["date"] + pd.Timedelta(hours=5))
    assert code_of(replace(request, sales_history=timed)) == "SCHEMA_INVALID"
    negative = request.sales_history.assign(sales=-1.0)
    assert code_of(replace(request, sales_history=negative)) == "SCHEMA_INVALID"
    empty_family = request.stores.assign(city="")
    assert code_of(replace(request, stores=empty_family)) == "SCHEMA_INVALID"
    assert code_of(replace(request, stores=request.stores.iloc[1:])) == "SCHEMA_INVALID"


def test_insufficient_context_is_explicit(frames) -> None:
    request = request_for(frames)
    with pytest.raises(ForecastRequestError) as error:
        validate_forecast_request(
            request, supported_series=request.series_keys, min_context_days=400
        )
    assert error.value.code == "HISTORY_INSUFFICIENT"


def test_missing_history_stays_missing(frames) -> None:
    request = request_for(frames)
    gap = request.sales_history["date"].eq(ORIGIN)
    sparse = replace(request, sales_history=request.sales_history.loc[~gap].reset_index(drop=True))
    validate_forecast_request(sparse, supported_series=sparse.series_keys)
    batch = build_request_feature_batch(
        sparse, FeatureSpec("asof_v1"), supported_series=sparse.series_keys
    )
    assert batch.frame["sales_origin_lag_1"].isna().all()
    assert batch.frame["sales_origin_lag_1_missing"].eq(1).all()


def test_label_join_is_keyed(frames) -> None:
    request = request_for(frames)
    batch = build_request_feature_batch(
        request, FeatureSpec("asof_v1"), supported_series=request.series_keys
    )
    labels = benchmark_labels(frames["train.csv"], request.target_dates)
    ordered = attach_labels(batch.frame, labels)
    shuffled = attach_labels(batch.frame, labels.sample(frac=1.0, random_state=7))
    pd.testing.assert_frame_equal(ordered, shuffled)
    with pytest.raises(ForecastRequestError, match="labels.key"):
        attach_labels(batch.frame, pd.concat([labels, labels.head(1)]))
    with pytest.raises(ForecastRequestError, match="labels.missing"):
        attach_labels(batch.frame, labels.iloc[1:])


@pytest.mark.parametrize("value", ["event", "Event ", "Festival"])
def test_unknown_calendar_type_cannot_bypass_event_exclusion(frames, value) -> None:
    request = request_for(frames)
    calendar = request.calendar.copy()
    calendar.loc[calendar.index[0], "type"] = value
    assert code_of(replace(request, calendar=calendar)) == "SCHEMA_INVALID"


def test_duplicate_calendar_source_row_is_rejected(frames) -> None:
    request = request_for(frames)
    calendar = pd.concat([request.calendar, request.calendar.head(1)], ignore_index=True)
    assert code_of(replace(request, calendar=calendar)) == "SCHEMA_INVALID"


def test_feature_builder_validates_directly_constructed_requests(frames) -> None:
    request = request_for(frames)
    bad = request.sales_history.copy()
    bad["sales"] = bad["sales"].astype(float)
    bad.loc[bad.index[0], "sales"] = np.inf
    with pytest.raises(ForecastRequestError) as error:
        build_request_feature_batch(
            replace(request, sales_history=bad),
            FeatureSpec("asof_v1"),
            supported_series=request.series_keys,
        )
    assert error.value.code == "NONFINITE_VALUE"


def test_projection_enforces_declared_minimum_context(frames) -> None:
    with pytest.raises(ForecastRequestError) as error:
        project_snapshot(
            frames,
            snapshot_id="b" * 64,
            data_source="synthetic",
            origin=ORIGIN,
            policy="asof_v1",
            context_days=56,
            min_context_days=56,
        )
    assert error.value.code == "HISTORY_INSUFFICIENT"


def test_error_key_text_is_bounded(frames) -> None:
    request = request_for(frames)
    long = "X" * 5000
    history = request.sales_history.copy()
    history["family"] = long
    with pytest.raises(ForecastRequestError) as error:
        validate_forecast_request(
            replace(request, sales_history=pd.concat([history, history.head(1)])),
            supported_series=request.series_keys,
        )
    assert len(str(error.value)) < 300
