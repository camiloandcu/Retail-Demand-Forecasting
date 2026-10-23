"""Run-scoped batch orchestration shared by CLI, CI and notebooks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retail_forecast.artifacts import (
    RunContext,
    allocate_run,
    capture_inputs,
    finish_run,
    read_completed_run,
    resolve_run,
    run_result,
    write_json_artifact,
    write_run_artifact,
)
from retail_forecast.baseline import fit_seasonal_naive, predict_seasonal_naive
from retail_forecast.config import ProjectConfig
from retail_forecast.data import validate_dataset, write_synthetic_dataset
from retail_forecast.experiments import run_baseline_experiments
from retail_forecast.io_utils import read_json
from retail_forecast.metrics import rmsle
from retail_forecast.reproducibility import set_global_seed


def _execute(
    config: ProjectConfig,
    kind: str,
    operation: Callable[[RunContext], dict[str, Any]],
    *,
    run_id: str | None,
    resume_from: str | None,
    parent_run_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Own a single execution and never rewrite a terminal manifest on failure."""

    if resume_from is not None:
        resolve_run(config, resume_from)
        raise ValueError("RESUME_UNSUPPORTED: current adapters have no eligible checkpoint")
    run = allocate_run(config, kind=kind, run_id=run_id, parent_run_ids=parent_run_ids)
    try:
        details = operation(run)
        finish_run(run)
    except BaseException:
        if run.manifest["state"] == "RUNNING":
            finish_run(run, state="FAILED", error_code="EXECUTION_FAILED")
        raise
    return {**run_result(run), **details}


def _inputs(run: RunContext, config: ProjectConfig) -> None:
    """Bind input bytes before any producer computes results."""

    capture_inputs(run, input_dir=config.paths.raw_data_dir)
    set_global_seed(config.runtime.seed)


def baseline_experiment(
    config: ProjectConfig,
    *,
    run_id: str | None = None,
    resume_from: str | None = None,
    tracking: bool = True,
) -> dict[str, Any]:
    """Run the unchanged four-configuration benchmark in one isolated execution."""

    def operation(run: RunContext) -> dict[str, Any]:
        active = config
        if config.data.source == "synthetic":
            staged = run.path("inputs/synthetic")
            active = replace(config, paths=replace(config.paths, raw_data_dir=staged))
            write_synthetic_dataset(active)
        _inputs(run, active)
        result = run_baseline_experiments(active, run=run, tracking=tracking)
        return {
            ("experiment_manifest" if name == "manifest" else name): str(path)
            for name, path in result["paths"].items()
        }

    return _execute(config, "baseline", operation, run_id=run_id, resume_from=resume_from)


def _train(run: RunContext, config: ProjectConfig) -> Path:
    summary = validate_dataset(config)
    train_frame = pd.read_csv(config.paths.raw_data_dir / "train.csv")
    origin = pd.to_datetime(train_frame["date"], errors="raise").max()
    model = fit_seasonal_naive(train_frame, origin, config.forecast.seasonal_period)
    write_json_artifact(run, "models/seasonal_naive.json", model, kind="model")
    write_json_artifact(run, "dataset_summary.json", summary.to_dict(), kind="dataset_summary")
    return run.root / "models/seasonal_naive.json"


def train(
    config: ProjectConfig, *, run_id: str | None = None, resume_from: str | None = None
) -> dict[str, Any]:
    """Fit the existing seasonal-naive model and return verified run paths."""

    def operation(run: RunContext) -> dict[str, Any]:
        _inputs(run, config)
        return {"model": str(_train(run, config))}

    return _execute(config, "training", operation, run_id=run_id, resume_from=resume_from)


def _evaluate(run: RunContext, config: ProjectConfig) -> Path:
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
            raise ValueError("EVALUATION_GRID_INVALID")
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
    write_json_artifact(run, "metrics/evaluation.json", output, kind="evaluation")
    return run.root / "metrics/evaluation.json"


def evaluate(
    config: ProjectConfig, *, run_id: str | None = None, resume_from: str | None = None
) -> dict[str, Any]:
    """Evaluate rolling origins independently of the final training artifact."""

    def operation(run: RunContext) -> dict[str, Any]:
        _inputs(run, config)
        return {"evaluation": str(_evaluate(run, config))}

    return _execute(config, "evaluation", operation, run_id=run_id, resume_from=resume_from)


def _predict(run: RunContext, config: ProjectConfig, model_path: Path) -> Path:
    model = read_json(model_path)
    if model.get("model_type") != "seasonal_naive_weekday":
        raise ValueError("MODEL_UNSUPPORTED")
    test = pd.read_csv(config.paths.raw_data_dir / "test.csv")
    predictions = predict_seasonal_naive(model, test)
    submission = pd.DataFrame({"id": test["id"].astype(int), "sales": predictions})
    if (
        list(submission.columns) != ["id", "sales"]
        or len(submission) != len(test)
        or submission["id"].duplicated().any()
        or not np.isfinite(submission["sales"]).all()
        or (submission["sales"] < 0).any()
    ):
        raise ValueError("PREDICTION_GRID_INVALID")
    write_run_artifact(
        run,
        "predictions/submission.csv",
        lambda path: submission.to_csv(path, index=False),
        kind="submission",
    )
    return run.root / "predictions/submission.csv"


def predict(
    config: ProjectConfig,
    *,
    training_run: str | None = None,
    run_id: str | None = None,
    resume_from: str | None = None,
) -> dict[str, Any]:
    """Predict only from an explicitly selected compatible completed run."""

    if training_run is None:
        raise ValueError("TRAINING_RUN_REQUIRED: use predict --training-run <run_id>")
    parent_root = resolve_run(config, training_run)

    def operation(run: RunContext) -> dict[str, Any]:
        _inputs(run, config)
        parent = read_completed_run(
            parent_root, expected_mode=config.mode, expected_source=config.data.source
        )
        if parent["kind"] not in {"training", "smoke"}:
            raise ValueError("TRAINING_RUN_KIND_INVALID")
        for identity in ("config_sha256", "snapshot_id", "availability_policy"):
            if parent[identity] != run.manifest[identity]:
                raise ValueError("TRAINING_RUN_IDENTITY_MISMATCH")
        models = [entry for entry in parent["artifacts"] if entry["kind"] == "model"]
        if len(models) != 1:
            raise ValueError("TRAINING_MODEL_INVALID")
        return {"submission": str(_predict(run, config, parent_root / models[0]["path"]))}

    return _execute(
        config,
        "prediction",
        operation,
        run_id=run_id,
        resume_from=resume_from,
        parent_run_ids=(training_run,),
    )


def smoke(
    config: ProjectConfig, *, run_id: str | None = None, resume_from: str | None = None
) -> dict[str, Any]:
    """Exercise all seasonal-naive stages with isolated synthetic inputs."""

    if config.mode != "smoke" or config.data.source != "synthetic":
        raise ValueError("The smoke command requires the smoke synthetic configuration")

    def operation(run: RunContext) -> dict[str, Any]:
        fixture_dir = run.path("inputs/synthetic")
        isolated = replace(config, paths=replace(config.paths, raw_data_dir=fixture_dir))
        write_synthetic_dataset(isolated)
        _inputs(run, isolated)
        evaluation_path = _evaluate(run, isolated)
        model_path = _train(run, isolated)
        submission_path = _predict(run, isolated, model_path)
        details = {
            "evaluation": str(evaluation_path.relative_to(run.root)),
            "model": str(model_path.relative_to(run.root)),
            "submission": str(submission_path.relative_to(run.root)),
            "test_rows": validate_dataset(isolated).test_rows,
        }
        write_json_artifact(run, "smoke_result.json", details, kind="smoke_summary")
        return {
            **{
                name: str(run.root / details[name])
                for name in ("evaluation", "model", "submission")
            },
            "test_rows": details["test_rows"],
        }

    return _execute(config, "smoke", operation, run_id=run_id, resume_from=resume_from)
