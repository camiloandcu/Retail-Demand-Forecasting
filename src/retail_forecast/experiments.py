"""Leakage-safe non-neural baseline experiments and segmented evaluation."""

from __future__ import annotations

import resource
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retail_forecast.artifacts import (
    RunContext,
    ensure_running,
    write_json_artifact,
    write_run_artifact,
)
from retail_forecast.baseline import (
    fit_seasonal_average,
    fit_seasonal_naive,
    predict_seasonal_average,
    predict_seasonal_naive,
)
from retail_forecast.config import ProjectConfig
from retail_forecast.data import load_frames
from retail_forecast.eda import sha256_file
from retail_forecast.features import (
    FeatureBatch,
    FeatureSpec,
    build_origin_feature_batch,
    fit_preprocessor,
    transform_features,
)
from retail_forecast.metrics import clip_nonnegative_predictions, rmsle
from retail_forecast.provenance import verify_inputs
from retail_forecast.splits import RollingOriginFold, make_rolling_origin_folds
from retail_forecast.tracking_runtime import log_baseline_tracking
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
    *,
    run: RunContext,
    tracking: bool = True,
) -> dict[str, Any]:
    """Run four pre-registered configurations on identical rolling-origin folds."""

    ensure_running(run)
    if run.manifest["kind"] != "baseline" or run.snapshot is None or run.input_dir is None:
        raise ValueError("Baseline requires its own run with captured input identity")
    if replace(config, paths=run.config.paths) != run.config:
        raise ValueError("Baseline config differs from allocated run identity")
    if len(MODEL_IDS) > 10 or config.baseline.tree_model != "lightgbm":
        raise ValueError("The experiment budget or single-tree contract was violated")
    try:
        import lightgbm as lgb
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("Install the 'models' and 'notebook' extras") from error

    metrics_dir = run.root / "metrics"
    figures_dir = run.root / "figures" / "baselines"

    def write(relative: str, writer: Any, kind: str) -> Path:
        write_run_artifact(run, relative, writer, kind=kind)
        return run.root / relative

    # The run's captured input root is the sole source of scientific frames.
    # Accepting caller-owned dataframes would let modified data borrow unrelated
    # disk snapshot provenance. Verify before and after loading to detect changes.
    verify_inputs(run.snapshot, run.input_dir)
    frames = load_frames(run.input_dir)
    verify_inputs(run.snapshot, run.input_dir)
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
        model_path = write(
            f"models/lightgbm_{fold.name}.txt",
            lambda path, tree=tree: tree.booster_.save_model(str(path)),
            "native_model",
        )
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
    results_path = write(
        "metrics/baseline_results.csv",
        lambda path: results.to_csv(path, index=False),
        "segmented_metrics",
    )
    oof_path = write(
        "predictions/baseline_oof.csv.gz",
        lambda path: oof.to_csv(path, index=False, compression="gzip"),
        "oof",
    )

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
    cost_path = write(
        "metrics/baseline_cost_quality.csv",
        lambda path: cost_quality.to_csv(path, index=False),
        "costs",
    )

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
    write_json_artifact(run, "metrics/baseline_champion.json", champion, kind="selection")

    global_plot = global_scores.sort_values("rmsle")
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(global_plot["model_id"], global_plot["rmsle"])
    axis.set_ylabel("OOF RMSLE")
    axis.set_title("Global non-neural baseline comparison")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    global_figure = figures_dir / "01_global_rmsle.png"
    write(
        "figures/baselines/01_global_rmsle.png",
        lambda path: figure.savefig(path, dpi=140),
        "figure",
    )
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
    write(
        "figures/baselines/02_horizon_rmsle.png",
        lambda path: figure.savefig(path, dpi=140),
        "figure",
    )
    plt.close(figure)

    runtime = capture_versions()
    tracking_status = log_baseline_tracking(
        run,
        enabled=tracking,
        model_ids=MODEL_IDS,
        tree_model_id=TREE_MODEL_ID,
        config=config,
        folds=len(folds),
        results=results,
        costs=costs,
        paths=[results_path, cost_path, champion_path, global_figure, horizon_figure],
        models=model_paths,
        tree_parameters=_lightgbm_params(config),
        versions=runtime,
    )
    manifest = {
        "run_id": run.run_id,
        "mode": config.mode,
        "source": config.data.source,
        "availability_policy": "competition_v1",
        "models": list(MODEL_IDS),
        "tree_selected_before_training": "lightgbm",
        "folds": [fold.to_dict() for fold in folds],
        "training_origins_by_fold": training_origins_by_fold,
        "skipped_training_origins_by_fold": skipped_training_origins_by_fold,
        "feature_contract": feature_contract,
        "snapshot_id": run.manifest["snapshot_id"],
        "config_sha256": run.manifest["config_sha256"],
        "versions": runtime,
        "results_path": str(results_path.relative_to(run.root)),
        "cost_quality_path": str(cost_path.relative_to(run.root)),
        "champion_path": str(champion_path.relative_to(run.root)),
        "oof_path": str(oof_path.relative_to(run.root)),
        "oof_rows": len(oof),
        "oof_bytes": oof_path.stat().st_size,
        "oof_sha256": sha256_file(oof_path),
        "oof_git_limit_mb": config.baseline.oof_git_limit_mb,
        "oof_should_be_committed": False,
        "tracking": tracking_status,
        "hardware": run.manifest["hardware"],
        "config": asdict(config.baseline),
    }
    manifest_path = metrics_dir / "baseline_manifest.json"
    write_json_artifact(run, "metrics/baseline_manifest.json", manifest, kind="experiment_summary")
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
