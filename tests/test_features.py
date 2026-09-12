"""Feature causality, imputation, join, and model input contract tests."""

from dataclasses import replace

import pandas as pd
import pytest

from retail_forecast.config import ProjectConfig
from retail_forecast.data import load_frames
from retail_forecast.features import (
    FeatureContractError,
    FeatureSpec,
    build_feature_batch,
    build_fold_feature_batches,
    build_store_holiday_calendar,
    feature_manifest,
    fit_preprocessor,
    model_input_contract,
    transform_features,
)
from retail_forecast.splits import make_rolling_origin_folds, split_manifest


@pytest.fixture
def smoke_feature_data(synthetic_config: ProjectConfig):
    frames = load_frames(synthetic_config.paths.raw_data_dir)
    folds = make_rolling_origin_folds(synthetic_config, frames["train.csv"]["date"])
    return frames, folds, FeatureSpec()


def test_origin_lags_are_constant_across_horizons(smoke_feature_data) -> None:
    frames, folds, spec = smoke_feature_data
    _, validation = build_fold_feature_batches(frames, folds[0], spec)
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


def test_future_target_sentinel_cannot_change_validation_features(
    smoke_feature_data,
) -> None:
    frames, folds, spec = smoke_feature_data
    _, clean = build_fold_feature_batches(frames, folds[0], spec)
    poisoned = {name: frame.copy() for name, frame in frames.items()}
    dates = pd.to_datetime(poisoned["train.csv"]["date"])
    future = dates.isin(folds[0].validation_dates)
    poisoned["train.csv"].loc[future, "sales"] = 999_999_999.0
    _, rebuilt = build_fold_feature_batches(poisoned, folds[0], spec)

    pd.testing.assert_frame_equal(
        clean.frame.drop(columns="sales"),
        rebuilt.frame.drop(columns="sales"),
    )
    assert not clean.frame["sales"].equals(rebuilt.frame["sales"])


def test_explicit_future_histories_are_rejected(smoke_feature_data) -> None:
    frames, folds, spec = smoke_feature_data
    origin = folds[0].origin
    train = frames["train.csv"].copy()
    dates = pd.to_datetime(train["date"])
    history = train.loc[dates <= origin].copy()
    target = train.loc[dates.isin(folds[0].validation_dates)].copy()
    leaked_history = pd.concat([history, target.head(1)], ignore_index=True)
    transaction_history = frames["transactions.csv"].copy()
    transaction_dates = pd.to_datetime(transaction_history["date"])
    transaction_history = transaction_history.loc[transaction_dates <= origin]

    with pytest.raises(FeatureContractError, match="after the forecast origin"):
        build_feature_batch(
            leaked_history,
            target,
            frames["stores.csv"],
            frames["oil.csv"],
            frames["holidays_events.csv"],
            transaction_history,
            origin,
            spec,
        )

    leaked_transactions = pd.concat(
        [
            transaction_history,
            pd.DataFrame(
                {"date": [origin + pd.offsets.Day(1)], "store_nbr": [1], "transactions": [1]}
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(FeatureContractError, match="after the forecast origin"):
        build_feature_batch(
            history,
            target,
            frames["stores.csv"],
            frames["oil.csv"],
            frames["holidays_events.csv"],
            leaked_transactions,
            origin,
            spec,
        )


def test_fold_fitted_imputation_and_unknown_category(smoke_feature_data) -> None:
    frames, folds, spec = smoke_feature_data
    training, validation = build_fold_feature_batches(frames, folds[0], spec)
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


def test_holiday_resolution_is_unique_and_store_aware() -> None:
    stores = pd.DataFrame(
        {
            "store_nbr": [1, 2],
            "city": ["Quito", "Cuenca"],
            "state": ["Pichincha", "Azuay"],
            "type": ["A", "B"],
            "cluster": [1, 2],
        }
    )
    holidays = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-02", "2020-01-02"],
            "type": ["Holiday", "Event", "Holiday"],
            "locale": ["National", "Local", "Regional"],
            "locale_name": ["Ecuador", "Quito", "Azuay"],
            "description": ["N", "Q", "A"],
            "transferred": [False, False, True],
        }
    )
    calendar = build_store_holiday_calendar(
        stores, holidays, pd.DatetimeIndex([pd.Timestamp("2020-01-02")])
    )

    assert len(calendar) == 2
    assert not calendar.duplicated(["store_nbr", "date"]).any()
    assert calendar.set_index("store_nbr").loc[1, "holiday_event_count"] == 2
    assert calendar.set_index("store_nbr").loc[2, "is_transferred_source"] == 1


def test_manifest_documents_both_model_contracts(smoke_feature_data, synthetic_config) -> None:
    frames, folds, spec = smoke_feature_data
    training, _ = build_fold_feature_batches(frames, folds[0], spec)
    splits = split_manifest(synthetic_config, folds)
    contract = model_input_contract(training)
    manifest = feature_manifest(spec, splits["sha256"], [], training)

    assert contract["row_key"] == [
        "store_nbr",
        "family",
        "forecast_origin",
        "target_date",
        "horizon",
    ]
    assert contract["recurrent"]["target_shape"] == ["batch", 16]
    assert manifest["processed_dataset_exported"] is False
    assert manifest["feature_spec_sha256"] == spec.sha256
