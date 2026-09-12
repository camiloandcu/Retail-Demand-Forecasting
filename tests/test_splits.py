"""Rolling-origin split and overlap rejection tests."""

from dataclasses import replace

import pandas as pd
import pytest

from retail_forecast.config import ForecastConfig, ProjectConfig
from retail_forecast.data import load_frames
from retail_forecast.splits import (
    RollingOriginFold,
    SplitContractError,
    make_rolling_origin_folds,
    split_manifest,
    validate_frame_against_fold,
)


def test_smoke_has_two_exact_16_day_folds(synthetic_config: ProjectConfig) -> None:
    frames = load_frames(synthetic_config.paths.raw_data_dir)
    folds = make_rolling_origin_folds(synthetic_config, frames["train.csv"]["date"])

    assert len(folds) == 2
    assert all(len(fold.validation_dates) == 16 for fold in folds)
    assert all(fold.train_end < fold.validation_start for fold in folds)
    assert split_manifest(synthetic_config, folds)["horizon"] == 16


def test_15_day_regression_is_rejected(synthetic_config: ProjectConfig) -> None:
    invalid = replace(
        synthetic_config,
        forecast=ForecastConfig(horizon=15, seasonal_period=7),
    )
    frames = load_frames(synthetic_config.paths.raw_data_dir)

    with pytest.raises(SplitContractError, match="Kaggle test grid"):
        make_rolling_origin_folds(invalid, frames["train.csv"]["date"])


def test_fold_rejects_train_validation_overlap() -> None:
    origin = pd.Timestamp("2020-02-01")
    with pytest.raises(SplitContractError, match="start one day"):
        RollingOriginFold(
            name="overlap",
            origin=origin,
            train_start=pd.Timestamp("2020-01-01"),
            train_end=origin,
            validation_start=origin,
            validation_end=origin + pd.offsets.Day(16),
            horizon=16,
        )


def test_frame_contract_rejects_future_training_rows(
    synthetic_config: ProjectConfig,
) -> None:
    frames = load_frames(synthetic_config.paths.raw_data_dir)
    fold = make_rolling_origin_folds(synthetic_config, frames["train.csv"]["date"])[0]
    dates = pd.to_datetime(frames["train.csv"]["date"])
    train = frames["train.csv"].loc[dates <= fold.origin].copy()
    validation = frames["train.csv"].loc[dates.isin(fold.validation_dates)].copy()
    leaked = pd.concat([train, validation.head(1)], ignore_index=True)

    with pytest.raises(SplitContractError, match="after the forecast origin"):
        validate_frame_against_fold(leaked, validation, fold)
