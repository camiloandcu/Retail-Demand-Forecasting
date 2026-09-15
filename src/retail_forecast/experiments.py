"""Leakage-safe non-neural baseline experiments and segmented evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retail_forecast.baseline import (
    fit_seasonal_average,
    fit_seasonal_naive,
    predict_seasonal_average,
    predict_seasonal_naive,
)
from retail_forecast.config import ProjectConfig
from retail_forecast.data import FILE_COLUMNS
from retail_forecast.eda import sha256_file
from retail_forecast.features import (
    FeatureBatch,
    FeatureSpec,
    build_origin_feature_batch,
    fit_preprocessor,
    transform_features,
)
from retail_forecast.io_utils import write_json
from retail_forecast.metrics import clip_nonnegative_predictions, rmsle
from retail_forecast.splits import RollingOriginFold, make_rolling_origin_folds
from retail_forecast.versions import capture_versions

MODEL_IDS = ("zero", "seasonal_naive_7d", "seasonal_average_4w", "lightgbm_global")
TREE_MODEL_ID = "lightgbm_global"
BUSINESS_BASELINE_IDS = ("seasonal_naive_7d", "seasonal_average_4w")


def training_origins(
    fold: RollingOriginFold,
    available_dates: pd.Series | pd.DatetimeIndex,
    *,
    stride_days: int,
    max_origins: int,
) -> tuple[pd.Timestamp, ...]:
    """Keep only pre-registered origins with a complete observed target horizon."""

    observed = pd.DatetimeIndex(pd.to_datetime(available_dates, errors="raise")).normalize()
    if observed.empty:
        raise ValueError("Observed training dates cannot be empty")
    observed = pd.DatetimeIndex(observed.unique()).sort_values()
    if observed.max() < fold.origin:
        raise ValueError(f"Training dates do not reach the forecast origin for {fold.name}")
    latest = fold.origin - pd.offsets.Day(fold.horizon)
    candidates = [latest - pd.offsets.Day(index * stride_days) for index in range(max_origins)]
    valid = [
        origin
        for origin in candidates
        if origin >= observed.min()
        and pd.date_range(origin + pd.offsets.Day(1), periods=fold.horizon, freq="D")
        .isin(observed)
        .all()
    ]
    if not valid:
        raise ValueError(f"No complete training horizon is available for {fold.name}")
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


def temporal_regime(frame: pd.DataFrame) -> pd.Series:
    """Assign one pre-defined, mutually exclusive target-date regime."""

    dates = pd.to_datetime(frame["target_date"], errors="raise")
    values = np.select(
        [
            frame["is_holiday_event"].astype(bool),
            dates.dt.day.eq(15) | dates.dt.is_month_end,
            dates.dt.dayofweek.ge(5),
        ],
        ["holiday_event", "payday", "weekend"],
        default="regular_weekday",
    )
    return pd.Series(values, index=frame.index, name="temporal_regime")


def segmented_rmsle(oof: pd.DataFrame) -> pd.DataFrame:
    """Return tidy global and required subgroup RMSLE values."""

    required = {
        "model_id",
        "fold",
        "horizon",
        "store_nbr",
        "family",
        "promotion_regime",
        "temporal_regime",
        "y_true",
        "y_pred",
    }
    missing = required.difference(oof.columns)
    if missing:
        raise ValueError(f"OOF frame is missing columns: {sorted(missing)}")
    records: list[dict[str, Any]] = []

    def append(model_id: str, scope: str, segment: str, frame: pd.DataFrame) -> None:
        records.append(
            {
                "model_id": model_id,
                "scope": scope,
                "segment": str(segment),
                "rows": len(frame),
                "rmsle": rmsle(frame["y_true"], frame["y_pred"]),
            }
        )

    for model_id, model_frame in oof.groupby("model_id", sort=False):
        append(model_id, "global", "all", model_frame)
        for scope, column in (
            ("fold", "fold"),
            ("horizon", "horizon"),
            ("store", "store_nbr"),
            ("family", "family"),
            ("promotion", "promotion_regime"),
            ("temporal_regime", "temporal_regime"),
        ):
            for segment, segment_frame in model_frame.groupby(column, observed=True, sort=True):
                append(model_id, scope, str(segment), segment_frame)
    return (
        pd.DataFrame(records).sort_values(["scope", "model_id", "segment"]).reset_index(drop=True)
    )


def _lightgbm_params(config: ProjectConfig) -> dict[str, Any]:
    baseline = config.baseline
    return {
        "objective": "regression",
        "n_estimators": baseline.tree_n_estimators,
        "learning_rate": baseline.tree_learning_rate,
        "num_leaves": baseline.tree_num_leaves,
        "min_child_samples": baseline.tree_min_child_samples,
        "colsample_bytree": baseline.tree_feature_fraction,
        "reg_lambda": baseline.tree_l2,
        "random_state": config.runtime.seed,
        "n_jobs": -1,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def _model_matrix(transformed: pd.DataFrame) -> pd.DataFrame:
    return transformed.drop(columns=["forecast_origin", "target_date", "sales"])


def run_baseline_experiments(
    config: ProjectConfig,
    frames: dict[str, pd.DataFrame],
    metrics_dir: Path,
    figures_dir: Path,
    mlflow_dir: Path,
) -> dict[str, Any]:
    """Run four pre-registered configurations on identical rolling-origin folds."""

    if len(MODEL_IDS) > 10 or config.baseline.tree_model != "lightgbm":
        raise ValueError("The experiment budget or single-tree contract was violated")
    try:
        import lightgbm as lgb
        import matplotlib.pyplot as plt
        import mlflow
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("Install the 'models' and 'notebook' extras") from error

    train = frames["train.csv"].copy()
    train["date"] = pd.to_datetime(train["date"], errors="raise")
    feature_frames = dict(frames)
    feature_frames["train.csv"] = train
    feature_frames["transactions.csv"] = frames["transactions.csv"].copy()
    feature_frames["transactions.csv"]["date"] = pd.to_datetime(
        feature_frames["transactions.csv"]["date"], errors="raise"
    )
    folds = make_rolling_origin_folds(config, train["date"])
    spec = FeatureSpec(horizon=config.forecast.horizon)
    batch_cache: dict[pd.Timestamp, FeatureBatch] = {}
    oof_parts: list[pd.DataFrame] = []
    cost_records: list[dict[str, Any]] = []
    model_paths: list[Path] = []
    training_origins_by_fold: dict[str, list[str]] = {}
    skipped_training_origins_by_fold: dict[str, dict[str, list[str]]] = {}
    feature_contract: dict[str, Any] | None = None

    for fold in folds:
        validation_batch = build_origin_feature_batch(feature_frames, fold.origin, spec)
        if feature_contract is None:
            feature_contract = {
                "count": validation_batch.feature_count,
                "numeric": list(validation_batch.numeric_features),
                "categorical": list(validation_batch.categorical_features),
                "spec": spec.to_dict(),
                "spec_sha256": spec.sha256,
            }
        validation = validation_batch.frame.copy()
        id_lookup = train.loc[
            train["date"].isin(fold.validation_dates),
            ["id", "date", "store_nbr", "family"],
        ].rename(columns={"date": "target_date"})
        validation = validation.merge(
            id_lookup,
            on=["target_date", "store_nbr", "family"],
            how="left",
            validate="one_to_one",
        )
        if validation["id"].isna().any():
            raise ValueError(f"{fold.name} OOF identifiers are incomplete")
        validation["promotion_regime"] = np.where(
            validation["onpromotion"].gt(0), "promoted", "not_promoted"
        )
        validation["temporal_regime"] = temporal_regime(validation)
        forecast_frame = validation.rename(columns={"target_date": "date"})
        predictions: dict[str, np.ndarray] = {}
        durations: dict[str, float] = {}
        memory_by_model: dict[str, int] = {}

        started = time.perf_counter()
        predictions["zero"] = np.zeros(len(validation), dtype=np.float64)
        durations["zero"] = time.perf_counter() - started
        memory_by_model["zero"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        started = time.perf_counter()
        weekly = fit_seasonal_naive(train, fold.origin, config.forecast.seasonal_period)
        predictions["seasonal_naive_7d"] = predict_seasonal_naive(weekly, forecast_frame).to_numpy()
        durations["seasonal_naive_7d"] = time.perf_counter() - started
        memory_by_model["seasonal_naive_7d"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        started = time.perf_counter()
        averaged = fit_seasonal_average(
            train,
            fold.origin,
            weeks=config.baseline.seasonal_average_weeks,
            seasonal_period=config.forecast.seasonal_period,
        )
        predictions["seasonal_average_4w"] = predict_seasonal_average(
            averaged, forecast_frame
        ).to_numpy()
        durations["seasonal_average_4w"] = time.perf_counter() - started
        memory_by_model["seasonal_average_4w"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        tree_started = time.perf_counter()
        origins = training_origins(
            fold,
            train["date"],
            stride_days=config.baseline.training_origin_stride_days,
            max_origins=config.baseline.max_training_origins,
        )
        training_origins_by_fold[fold.name] = [origin.date().isoformat() for origin in origins]
        observed_dates = pd.DatetimeIndex(train["date"].unique())
        selected = set(origins)
        latest = fold.origin - pd.offsets.Day(fold.horizon)
        skipped_training_origins_by_fold[fold.name] = {
            origin.date().isoformat(): [
                date.date().isoformat()
                for date in pd.date_range(
                    origin + pd.offsets.Day(1), periods=fold.horizon, freq="D"
                )
                if date not in observed_dates
            ]
            for index in range(config.baseline.max_training_origins)
            if (
                origin := latest
                - pd.offsets.Day(index * config.baseline.training_origin_stride_days)
            )
            >= observed_dates.min()
            and origin not in selected
        }
        for origin in origins:
            if origin not in batch_cache:
                batch_cache[origin] = build_origin_feature_batch(feature_frames, origin, spec)
        training_batch = combine_feature_batches([batch_cache[origin] for origin in origins])
        fitted = fit_preprocessor(training_batch, fold.origin)
        transformed_train = transform_features(training_batch, fitted)
        transformed_validation = transform_features(validation_batch, fitted)
        x_train = _model_matrix(transformed_train)
        x_validation = _model_matrix(transformed_validation)
        categorical = [column for column in x_train if column.startswith("cat__")]
        tree = lgb.LGBMRegressor(**_lightgbm_params(config))
        tree.fit(
            x_train,
            np.log1p(transformed_train["sales"].to_numpy()),
            categorical_feature=categorical,
        )
        raw_tree = np.expm1(tree.predict(x_validation))
        predictions[TREE_MODEL_ID] = clip_nonnegative_predictions(raw_tree)
        durations[TREE_MODEL_ID] = time.perf_counter() - tree_started
        memory_by_model[TREE_MODEL_ID] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        model_path = config.paths.artifacts_dir / "models" / f"lightgbm_{fold.name}.txt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        tree.booster_.save_model(str(model_path))
        model_paths.append(model_path)

        for model_id in MODEL_IDS:
            part = validation[
                [
                    "store_nbr",
                    "family",
                    "id",
                    "forecast_origin",
                    "target_date",
                    "horizon",
                    "onpromotion",
                    "promotion_regime",
                    "temporal_regime",
                    "sales",
                ]
            ].rename(columns={"sales": "y_true"})
            part.insert(0, "fold", fold.name)
            part.insert(0, "model_id", model_id)
            part["y_pred"] = predictions[model_id]
            part["squared_log_error"] = np.square(
                np.log1p(part["y_pred"]) - np.log1p(part["y_true"])
            )
            oof_parts.append(part)
            cost_records.append(
                {
                    "model_id": model_id,
                    "fold": fold.name,
                    "training_rows": len(training_batch.frame) if model_id == TREE_MODEL_ID else 0,
                    "duration_seconds": durations[model_id],
                    "process_max_rss_kib": memory_by_model[model_id],
                }
            )

    oof = pd.concat(oof_parts, ignore_index=True)
    results = segmented_rmsle(oof)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_path = metrics_dir / "baseline_results.csv"
    results.to_csv(results_path, index=False)
    oof_path = config.paths.artifacts_dir / "predictions" / "baseline_oof.csv.gz"
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(oof_path, index=False, compression="gzip")

    costs = pd.DataFrame(cost_records)
    global_scores = results.loc[results["scope"].eq("global"), ["model_id", "rmsle"]]
    cost_quality = (
        costs.groupby("model_id", as_index=False)
        .agg(
            duration_seconds=("duration_seconds", "sum"),
            process_max_rss_kib=("process_max_rss_kib", "max"),
            max_training_rows=("training_rows", "max"),
        )
        .merge(global_scores, on="model_id", validate="one_to_one")
        .sort_values("rmsle")
    )
    cost_path = metrics_dir / "baseline_cost_quality.csv"
    cost_quality.to_csv(cost_path, index=False)

    business_scores = global_scores.loc[global_scores["model_id"].isin(BUSINESS_BASELINE_IDS)]
    business_champion = business_scores.sort_values("rmsle").iloc[0]
    tree_rmsle = float(
        global_scores.loc[global_scores["model_id"].eq(TREE_MODEL_ID), "rmsle"].item()
    )
    champion = {
        "mode": config.mode,
        "empirical_results_are_synthetic": config.mode == "smoke",
        "business_baseline": business_champion["model_id"],
        "business_baseline_rmsle": float(business_champion["rmsle"]),
        "non_neural_reference": TREE_MODEL_ID,
        "non_neural_reference_rmsle": tree_rmsle,
        "neural_min_relative_improvement": config.baseline.neural_min_relative_improvement,
        "neural_rmsle_threshold": tree_rmsle
        * (1.0 - config.baseline.neural_min_relative_improvement),
        "configuration_count": len(MODEL_IDS),
        "tree_parameters": _lightgbm_params(config),
        "limitations_for_recurrent_model": [
            "Origin-level summaries do not preserve the full order of the 56-day history.",
            "One pooled tree objective may underrepresent intermittent, high-zero series.",
            "Separate horizon rows cannot learn a joint 16-step sequence representation.",
        ],
    }
    champion_path = metrics_dir / "baseline_champion.json"
    write_json(champion_path, champion)

    global_plot = global_scores.sort_values("rmsle")
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(global_plot["model_id"], global_plot["rmsle"])
    axis.set_ylabel("OOF RMSLE")
    axis.set_title("Global non-neural baseline comparison")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    global_figure = figures_dir / "01_global_rmsle.png"
    figure.savefig(global_figure, dpi=140)
    plt.close(figure)

    horizon = results.loc[results["scope"].eq("horizon")].copy()
    horizon["segment"] = horizon["segment"].astype(int)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    for model_id, model_frame in horizon.groupby("model_id", sort=False):
        axis.plot(model_frame["segment"], model_frame["rmsle"], marker="o", label=model_id)
    axis.set(xlabel="Horizon", ylabel="RMSLE", title="Error by forecast horizon")
    axis.legend()
    figure.tight_layout()
    horizon_figure = figures_dir / "02_horizon_rmsle.png"
    figure.savefig(horizon_figure, dpi=140)
    plt.close(figure)

    mlflow_dir.mkdir(parents=True, exist_ok=True)
    mlflow_database_path = (mlflow_dir / "mlflow.db").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_database_path}")
    mlflow.set_experiment("retail-demand-non-neural-baselines")
    runtime = capture_versions()
    hardware_params = {
        "runtime_platform": platform.platform(),
        "runtime_cpu_count": os.cpu_count() or "unknown",
    }
    baseline_params = {f"baseline_{key}": value for key, value in asdict(config.baseline).items()}
    for model_id in MODEL_IDS:
        model_scores = results.loc[results["model_id"].eq(model_id)]
        model_costs = costs.loc[costs["model_id"].eq(model_id)]
        with mlflow.start_run(run_name=f"{config.mode}-{model_id}"):
            mlflow.set_tags(
                {
                    "mode": config.mode,
                    "model_family": "tree" if model_id == TREE_MODEL_ID else "baseline",
                    "production_ready": "false",
                }
            )
            mlflow.log_params(
                {
                    "model_id": model_id,
                    "fold_count": len(folds),
                    "horizon": config.forecast.horizon,
                    "seed": config.runtime.seed,
                    **hardware_params,
                    **baseline_params,
                    **(_lightgbm_params(config) if model_id == TREE_MODEL_ID else {}),
                }
            )
            mlflow.log_metric(
                "rmsle_global",
                float(model_scores.loc[model_scores["scope"].eq("global"), "rmsle"].item()),
            )
            for row in model_scores.loc[model_scores["scope"].eq("fold")].itertuples():
                mlflow.log_metric(f"rmsle_{row.segment}", float(row.rmsle))
            mlflow.log_metric("duration_seconds", float(model_costs["duration_seconds"].sum()))
            mlflow.log_metric(
                "process_max_rss_kib", float(model_costs["process_max_rss_kib"].max())
            )
            mlflow.log_dict(runtime, "runtime_versions.json")
            mlflow.log_artifact(str(results_path))
            mlflow.log_artifact(str(cost_path))
            mlflow.log_artifact(str(champion_path))
            mlflow.log_artifact(str(global_figure), artifact_path="figures")
            mlflow.log_artifact(str(horizon_figure), artifact_path="figures")
            if model_id == TREE_MODEL_ID:
                for model_path in model_paths:
                    mlflow.log_artifact(str(model_path), artifact_path="models")

    oof_digest = hashlib.sha256(oof_path.read_bytes()).hexdigest()
    input_files = {
        filename: {
            "bytes": (config.paths.raw_data_dir / filename).stat().st_size,
            "sha256": sha256_file(config.paths.raw_data_dir / filename),
        }
        for filename in FILE_COLUMNS
    }
    manifest = {
        "mode": config.mode,
        "source": config.data.source,
        "models": list(MODEL_IDS),
        "tree_selected_before_training": "lightgbm",
        "folds": [fold.to_dict() for fold in folds],
        "training_origins_by_fold": training_origins_by_fold,
        "skipped_training_origins_by_fold": skipped_training_origins_by_fold,
        "feature_contract": feature_contract,
        "input_files": input_files,
        "config_path": str(config.config_path),
        "config_sha256": sha256_file(config.config_path),
        "versions": runtime,
        "results_path": str(results_path),
        "cost_quality_path": str(cost_path),
        "champion_path": str(champion_path),
        "oof_path": str(oof_path),
        "oof_rows": len(oof),
        "oof_bytes": oof_path.stat().st_size,
        "oof_sha256": oof_digest,
        "oof_git_limit_mb": config.baseline.oof_git_limit_mb,
        "oof_should_be_committed": False,
        "mlflow_tracking_uri": mlflow.get_tracking_uri(),
        "mlflow_database_path": str(mlflow_database_path),
        "hardware": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "process_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "config": asdict(config.baseline),
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=str)
    manifest["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    manifest_path = metrics_dir / "baseline_manifest.json"
    write_json(manifest_path, manifest)
    return {
        "results": results,
        "cost_quality": cost_quality,
        "champion": champion,
        "manifest": manifest,
        "oof": oof,
        "paths": {
            "results": results_path,
            "cost_quality": cost_path,
            "champion": champion_path,
            "manifest": manifest_path,
            "oof": oof_path,
        },
    }
