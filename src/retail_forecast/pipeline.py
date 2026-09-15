"""Batch orchestration used identically by CLI, CI, and future notebooks."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from retail_forecast.baseline import fit_seasonal_naive, predict_seasonal_naive
from retail_forecast.config import ProjectConfig
from retail_forecast.data import load_frames, validate_dataset, write_synthetic_dataset
from retail_forecast.experiments import run_baseline_experiments
from retail_forecast.io_utils import read_json, write_json
from retail_forecast.metrics import rmsle
from retail_forecast.reproducibility import set_global_seed
from retail_forecast.versions import capture_versions

LOGGER = logging.getLogger(__name__)


def baseline_experiment(config: ProjectConfig) -> dict[str, Any]:
    """Run the complete non-neural benchmark from the command line."""

    set_global_seed(config.runtime.seed)
    validate_dataset(config)
    frames = load_frames(config.paths.raw_data_dir)
    return run_baseline_experiments(
        config,
        frames,
        config.project_root / "artifacts/metrics",
        config.project_root / "artifacts/figures/baselines",
        config.project_root / "mlruns",
    )["manifest"]


def _metadata(config: ProjectConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["versions"] = capture_versions()
    return payload


def train(config: ProjectConfig) -> Path:
    """Validate data and fit the mandatory baseline inference artifact."""

    set_global_seed(config.runtime.seed)
    summary = validate_dataset(config)
    train_frame = pd.read_csv(config.paths.raw_data_dir / "train.csv")
    origin = pd.to_datetime(train_frame["date"], errors="raise").max()
    model = fit_seasonal_naive(train_frame, origin, config.forecast.seasonal_period)
    model_path = config.paths.artifacts_dir / "model.json"
    write_json(model_path, model)
    write_json(config.paths.artifacts_dir / "run_metadata.json", _metadata(config))
    write_json(config.paths.artifacts_dir / "dataset_summary.json", summary.to_dict())
    LOGGER.info("Wrote baseline model to %s", model_path)
    return model_path


def evaluate(config: ProjectConfig) -> Path:
    """Evaluate each pre-registered origin without fitting on its future targets."""

    set_global_seed(config.runtime.seed)
    validate_dataset(config)
    train_frame = pd.read_csv(config.paths.raw_data_dir / "train.csv")
    train_frame["date"] = pd.to_datetime(train_frame["date"], errors="raise")
    fold_metrics: list[dict[str, Any]] = []

    for origin in config.validation.origins:
        origin_timestamp = pd.Timestamp(origin)
        validation_end = origin_timestamp + pd.offsets.Day(config.forecast.horizon)
        validation = train_frame.loc[
            (train_frame["date"] > origin_timestamp) & (train_frame["date"] <= validation_end)
        ].copy()
        expected_rows = (
            config.data.expected_stores * config.data.expected_families * config.forecast.horizon
        )
        if len(validation) != expected_rows:
            raise ValueError(
                f"Origin {origin} has {len(validation)} validation rows; expected {expected_rows}"
            )
        model = fit_seasonal_naive(train_frame, origin_timestamp, config.forecast.seasonal_period)
        predictions = predict_seasonal_naive(model, validation)
        fold_metrics.append(
            {
                "origin": origin.isoformat(),
                "rows": len(validation),
                "rmsle": rmsle(validation["sales"], predictions),
            }
        )

    output = {
        "metric": "rmsle",
        "model_type": "seasonal_naive_weekday",
        "smoke": config.mode == "smoke",
        "folds": fold_metrics,
        "mean_rmsle": float(pd.Series([fold["rmsle"] for fold in fold_metrics]).mean()),
    }
    path = config.paths.artifacts_dir / "evaluation.json"
    write_json(path, output)
    LOGGER.info("Wrote evaluation to %s", path)
    return path


def predict(config: ProjectConfig) -> Path:
    """Run batch inference and validate the Kaggle-shaped output."""

    set_global_seed(config.runtime.seed)
    validate_dataset(config)
    model = read_json(config.paths.artifacts_dir / "model.json")
    test = pd.read_csv(config.paths.raw_data_dir / "test.csv")
    predictions = predict_seasonal_naive(model, test)
    submission = pd.DataFrame({"id": test["id"].astype(int), "sales": predictions})
    if list(submission.columns) != ["id", "sales"]:
        raise AssertionError("Submission column order changed unexpectedly")
    if len(submission) != len(test) or submission["id"].duplicated().any():
        raise AssertionError("Submission identity contract failed")
    if submission["sales"].isna().any() or (submission["sales"] < 0).any():
        raise AssertionError("Submission predictions must be finite and non-negative")
    path = config.paths.artifacts_dir / "submission.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)
    LOGGER.info("Wrote %d predictions to %s", len(submission), path)
    return path


def smoke(config: ProjectConfig) -> dict[str, str]:
    """Generate schema fixtures and execute validation, evaluation, training, and inference."""

    if config.mode != "smoke" or config.data.source != "synthetic":
        raise ValueError("The smoke command requires the smoke synthetic configuration")
    fixture_dir = write_synthetic_dataset(config)
    summary = validate_dataset(config)
    evaluation_path = evaluate(config)
    model_path = train(config)
    submission_path = predict(config)
    result = {
        "fixture_dir": str(fixture_dir),
        "evaluation": str(evaluation_path),
        "model": str(model_path),
        "submission": str(submission_path),
        "test_rows": str(summary.test_rows),
    }
    write_json(config.paths.artifacts_dir / "smoke_result.json", result)
    return result
