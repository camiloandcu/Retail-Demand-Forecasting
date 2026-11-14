"""Feature causality, policy separation, calendar joins and model input contracts."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from retail_forecast.availability import resolve_policy
from retail_forecast.config import ProjectConfig
from retail_forecast.data import load_frames
from retail_forecast.data_contracts import ForecastRequestError
from retail_forecast.features import (
    FeatureContractError,
    FeatureSpec,
    build_fold_feature_batches,
    build_origin_feature_batch,
    build_request_feature_batch,
    build_store_holiday_calendar,
    feature_manifest,
    fit_preprocessor,
    model_input_contract,
    project_benchmark_request,
    transform_features,
)
from retail_forecast.splits import make_rolling_origin_folds, split_manifest

IDENTITY = {"snapshot_id": "a" * 64, "data_source": "synthetic"}


def frame_hash(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(["target_date", "store_nbr", "family"]).reset_index(drop=True)
    ordered = ordered[sorted(ordered.columns)]
    return hashlib.sha256(pd.util.hash_pandas_object(ordered, index=False).values).hexdigest()


@pytest.fixture
def smoke_feature_data(synthetic_config: ProjectConfig):
    frames = load_frames(synthetic_config.paths.raw_data_dir)
    folds = make_rolling_origin_folds(synthetic_config, frames["train.csv"]["date"])
    return frames, folds, FeatureSpec("competition_v1")


def features(frames, origin, policy="competition_v1"):
    return build_origin_feature_batch(frames, origin, FeatureSpec(policy), **IDENTITY)


def test_origin_lags_are_constant_across_horizons(smoke_feature_data) -> None:
    frames, folds, spec = smoke_feature_data
    _, validation = build_fold_feature_batches(frames, folds[0], spec, **IDENTITY)
    one_series = validation.frame.loc[
        (validation.frame["store_nbr"] == 1) & (validation.frame["family"] == "SYNTHETIC_FAMILY_01")
    ]
    history = frames["train.csv"].copy()
    expected = history.loc[
        (history["date"] == folds[0].origin.date().isoformat())
        & (history["store_nbr"] == 1)
        & (history["family"] == "SYNTHETIC_FAMILY_01"),
        "sales",
    ].item()

    assert one_series["sales_origin_lag_1"].nunique() == 1
    assert one_series["sales_origin_lag_1"].iloc[0] == expected
    assert set(one_series["horizon"]) == set(range(1, 17))


@pytest.mark.parametrize("policy", ["competition_v1", "asof_v1"])
def test_future_label_poisoning_cannot_change_features(smoke_feature_data, policy) -> None:
    frames, folds, _ = smoke_feature_data
    origin = folds[0].origin
    clean = features(frames, origin, policy)
    poisoned = {name: frame.copy() for name, frame in frames.items()}
    for name, column in (("train.csv", "sales"), ("transactions.csv", "transactions")):
        dates = pd.to_datetime(poisoned[name]["date"])
        poisoned[name].loc[dates > origin, column] *= 1_000_003
    rebuilt = features(poisoned, origin, policy)

    assert frame_hash(clean.frame.drop(columns="sales")) == frame_hash(
        rebuilt.frame.drop(columns="sales")
    )
    assert not clean.frame["sales"].equals(rebuilt.frame["sales"])


def test_asof_ignores_poisoned_oil_and_events(smoke_feature_data) -> None:
    frames, folds, _ = smoke_feature_data
    origin = folds[1].origin
    target = (origin + pd.offsets.Day(3)).date().isoformat()
    poisoned = {name: frame.copy() for name, frame in frames.items()}
    poisoned["oil.csv"]["dcoilwtico"] = 9e300
    quake = pd.DataFrame(
        [[target, "Event", "National", "Synthetic Country", "Synthetic quake", False]] * 5,
        columns=poisoned["holidays_events.csv"].columns,
    )
    quake["description"] = [f"Synthetic quake {index}" for index in range(5)]
    poisoned["holidays_events.csv"] = pd.concat(
        [poisoned["holidays_events.csv"], quake], ignore_index=True
    )

    clean_asof = features(frames, origin, "asof_v1")
    poisoned_asof = features(poisoned, origin, "asof_v1")
    assert frame_hash(clean_asof.frame) == frame_hash(poisoned_asof.frame)
    assert not any("oil" in column for column in clean_asof.frame.columns)

    clean_competition = features(frames, origin)
    poisoned_competition = features(poisoned, origin)
    assert frame_hash(clean_competition.frame) != frame_hash(poisoned_competition.frame)
    assert "oil_target" in clean_competition.numeric_features


def test_policies_differ_only_where_event_rows_apply(smoke_feature_data) -> None:
    frames, folds, _ = smoke_feature_data
    competition = features(frames, folds[1].origin).frame
    asof = features(frames, folds[1].origin, "asof_v1").frame
    key = ["target_date", "store_nbr", "family"]
    merged = competition.merge(asof, on=key, suffixes=("_c", "_a"), validate="one_to_one")
    changed = merged["holiday_event_count_c"].ne(merged["holiday_event_count_a"])
    calendar = frames["holidays_events.csv"]
    events = calendar.loc[calendar["type"].eq("Event"), "date"]
    assert changed.any()
    assert set(merged.loc[changed, "target_date"].dt.date.astype(str)) <= set(events)
    shared = [column for column in asof.columns if column not in key]
    unchanged = merged.loc[~changed]
    for column in shared:
        np.testing.assert_array_equal(unchanged[f"{column}_c"], unchanged[f"{column}_a"])


def test_feature_spec_identity_separates_policies() -> None:
    competition, asof = FeatureSpec("competition_v1"), FeatureSpec("asof_v1")
    assert competition.sha256 != asof.sha256
    with pytest.raises(ValueError, match="POLICY_MISMATCH"):
        FeatureSpec("realtime_v9")


def test_request_and_spec_policy_must_match(smoke_feature_data) -> None:
    frames, folds, _ = smoke_feature_data
    request = project_benchmark_request(frames, folds[0].origin, FeatureSpec("asof_v1"), **IDENTITY)
    with pytest.raises(FeatureContractError, match="POLICY_MISMATCH"):
        build_request_feature_batch(request, FeatureSpec("competition_v1"))


def test_explicit_future_history_is_rejected(smoke_feature_data) -> None:
    frames, folds, spec = smoke_feature_data
    request = project_benchmark_request(frames, folds[0].origin, spec, **IDENTITY)
    late = request.sales_history.tail(1).assign(date=folds[0].origin + pd.offsets.Day(1))
    leaked = replace(
        request, sales_history=pd.concat([request.sales_history, late], ignore_index=True)
    )
    from retail_forecast.data_contracts import validate_forecast_request

    with pytest.raises(ForecastRequestError) as error:
        validate_forecast_request(leaked, supported_series=request.series_keys)
    assert error.value.code == "HISTORY_AFTER_ORIGIN"


def test_fold_fitted_imputation_and_unknown_category(smoke_feature_data) -> None:
    frames, folds, spec = smoke_feature_data
    training, validation = build_fold_feature_batches(frames, folds[0], spec, **IDENTITY)
    fitted = fit_preprocessor(training, folds[0].origin)
    changed = validation.frame.copy()
    changed.loc[0, "family"] = "NEVER_SEEN_IN_TRAIN"
    changed.loc[0, "oil_target"] = float("nan")
    changed.loc[0, "oil_target_missing"] = 1
    transformed = transform_features(replace(validation, frame=changed), fitted)

    assert transformed.loc[0, "cat__family"] == 0
    assert transformed.loc[0, "oil_target"] == pytest.approx(fitted.numeric_medians["oil_target"])
    assert transformed.loc[0, "oil_target_missing"] == 1
    with pytest.raises(FeatureContractError, match="beyond the training cutoff"):
        fit_preprocessor(validation, folds[0].origin)


STORES = pd.DataFrame(
    {
        "store_nbr": [1, 2, 3],
        "city": ["Quito", "Cuenca", "Quito"],
        "state": ["Pichincha", "Azuay", "Pichincha"],
        "type": ["A", "B", "C"],
        "cluster": [1, 2, 3],
    }
)


def calendar_of(rows, policy="competition_v1", dates=("2020-01-02", "2020-01-03")):
    holidays = pd.DataFrame(
        rows, columns=["date", "type", "locale", "locale_name", "description", "transferred"]
    )
    return build_store_holiday_calendar(
        STORES,
        holidays,
        pd.DatetimeIndex(pd.to_datetime(list(dates))),
        policy=resolve_policy(policy),
    ).set_index(["store_nbr", "date"])


def test_holiday_scopes_resolve_by_state_and_city() -> None:
    calendar = calendar_of(
        [
            ("2020-01-02", "Holiday", "National", "Ecuador", "N", False),
            ("2020-01-02", "Holiday", "Regional", "Azuay", "A", False),
            ("2020-01-02", "Holiday", "Local", "Quito", "Q1", False),
            ("2020-01-02", "Additional", "Local", "Quito", "Q2", False),
        ]
    )
    day = pd.Timestamp("2020-01-02")
    assert len(calendar) == 3 * 2
    assert calendar.loc[(1, day), "holiday_event_count"] == 3
    assert calendar.loc[(2, day), "holiday_event_count"] == 2
    assert calendar.loc[(3, day), "holiday_event_count"] == 3
    assert calendar.loc[(2, pd.Timestamp("2020-01-03")), "holiday_event_count"] == 0


def test_transfer_semantics_are_preserved() -> None:
    calendar = calendar_of(
        [
            ("2020-01-02", "Holiday", "National", "Ecuador", "Moved", True),
            ("2020-01-03", "Transfer", "National", "Ecuador", "Moved", False),
        ]
    )
    original = calendar.loc[(1, pd.Timestamp("2020-01-02"))]
    moved = calendar.loc[(1, pd.Timestamp("2020-01-03"))]
    assert original["is_holiday_event"] == 0 and original["is_transferred_source"] == 1
    assert moved["is_holiday_event"] == 1 and moved["is_transfer"] == 1


def test_event_rows_are_excluded_only_under_asof() -> None:
    rows = [("2020-01-02", "Event", "National", "Ecuador", "Quake", False)]
    assert calendar_of(rows)["holiday_event_count"].sum() == 3
    assert calendar_of(rows, "asof_v1")["holiday_event_count"].sum() == 0


def test_unknown_locale_is_rejected() -> None:
    with pytest.raises(FeatureContractError, match="locale"):
        calendar_of([("2020-01-02", "Holiday", "Province", "X", "X", False)])


def test_calendar_join_preserves_grid(smoke_feature_data) -> None:
    frames, folds, _ = smoke_feature_data
    for policy in ("competition_v1", "asof_v1"):
        batch = features(frames, folds[0].origin, policy)
        assert len(batch.frame) == 2 * 2 * 16
        assert not batch.frame.duplicated(["target_date", "store_nbr", "family"]).any()


def test_manifest_documents_policy_and_model_contracts(
    smoke_feature_data, synthetic_config
) -> None:
    frames, folds, spec = smoke_feature_data
    training, _ = build_fold_feature_batches(frames, folds[0], spec, **IDENTITY)
    splits = split_manifest(synthetic_config, folds)
    contract = model_input_contract(training)
    fitted = fit_preprocessor(training, folds[0].origin)
    run_evidence = {"mode": "smoke", "gate_status": "PASS"}
    manifest = feature_manifest(spec, splits["sha256"], [], training, run_evidence)

    assert contract["row_key"] == [
        "store_nbr",
        "family",
        "forecast_origin",
        "target_date",
        "horizon",
    ]
    assert contract["availability_policy"] == "competition_v1"
    assert contract["recurrent"]["target_shape"] == ["batch", 16]
    assert manifest["availability_policy"] == "competition_v1"
    assert manifest["input_contract_version"] == 1
    assert manifest["processed_dataset_exported"] is False
    assert manifest["feature_spec_sha256"] == spec.sha256
    assert manifest["run_evidence"] == run_evidence
    assert len(fitted.sha256) == 64
    unhashed = {key: value for key, value in manifest.items() if key != "sha256"}
    canonical = json.dumps(unhashed, sort_keys=True, separators=(",", ":"))
    assert manifest["sha256"] == hashlib.sha256(canonical.encode()).hexdigest()

    asof_training, _ = build_fold_feature_batches(
        frames, folds[0], FeatureSpec("asof_v1"), **IDENTITY
    )
    asof_contract = model_input_contract(asof_training)
    assert "oil_target" not in asof_contract["recurrent"]["known_future_features"]
