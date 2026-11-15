"""Run-scoped batch orchestration shared by CLI, CI and notebooks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retail_forecast.artifacts import (
    RunContext,
    allocate_run,
    capture_inputs,
    capture_request_inputs,
    finish_run,
    read_completed_run,
    resolve_run,
    run_result,
    safe_path,
    write_json_artifact,
    write_run_artifact,
)
from retail_forecast.baseline import fit_seasonal_naive, predict_seasonal_naive
from retail_forecast.bundles import (
    BundleRef,
    LoadedModel,
    bundle_relative_dir,
    load_bundle,
    parse_bundle_ref,
    save_bundle,
)
from retail_forecast.config import ProjectConfig
from retail_forecast.data import load_frames, validate_dataset, write_synthetic_dataset
from retail_forecast.data_contracts import (
    FORECAST_COLUMNS,
    ForecastRequest,
    attach_labels,
    benchmark_labels,
    project_snapshot,
)
from retail_forecast.experiments import run_baseline_experiments, segmented_rmsle, temporal_regime
from retail_forecast.features import FeatureSpec, build_request_feature_batch
from retail_forecast.inference import predict_bundle, read_forecast_csv, write_forecast_csv
from retail_forecast.io_utils import canonical_json
from retail_forecast.kaggle_export import export_kaggle_submission
from retail_forecast.metrics import rmsle
from retail_forecast.models import COMPARATOR_MODEL_IDS, SERVING_MODEL_IDS, get_adapter
from retail_forecast.models.base import FitContext, PredictionContext
from retail_forecast.models.lightgbm import lightgbm_params
from retail_forecast.provenance import verify_inputs
from retail_forecast.reproducibility import set_global_seed
from retail_forecast.request_store import MANIFEST_NAME as REQUEST_MANIFEST
from retail_forecast.request_store import request_from_files, serialize_request
from retail_forecast.splits import make_rolling_origin_folds
from retail_forecast.training import (
    FeatureBatchCache,
    build_training_set,
    prepare_feature_frames,
)

RELOAD_RTOL = 1e-10
RELOAD_ATOL = 1e-8


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


def _stage_inputs(run: RunContext, config: ProjectConfig) -> ProjectConfig:
    """Bind inputs; synthetic runs regenerate their deterministic fixture inside the run."""

    active = config
    if config.data.source == "synthetic":
        staged = run.path("inputs/synthetic")
        active = replace(config, paths=replace(config.paths, raw_data_dir=staged))
        write_synthetic_dataset(active)
    _inputs(run, active)
    return active


def _verified_frames(run: RunContext) -> dict[str, pd.DataFrame]:
    if run.snapshot is None or run.input_dir is None:
        raise ValueError("Run has no captured inputs")
    verify_inputs(run.snapshot, run.input_dir)
    frames = load_frames(run.input_dir)
    verify_inputs(run.snapshot, run.input_dir)
    return prepare_feature_frames(frames)


def _supported_series(run: RunContext) -> tuple[tuple[int, str], ...]:
    assert run.snapshot is not None
    return tuple(
        (int(item["store_nbr"]), str(item["family"])) for item in run.snapshot["supported_series"]
    )


def _adapter_params(config: ProjectConfig, model_id: str) -> dict[str, Any]:
    if model_id == "lightgbm_global":
        if config.baseline.tree_model != "lightgbm":
            raise ValueError("The configured tree model is not lightgbm")
        return lightgbm_params(config.baseline, config.runtime.seed)
    if model_id == "seasonal_average_4w" and config.baseline.seasonal_average_weeks != 4:
        raise ValueError("seasonal_average_4w requires seasonal_average_weeks: 4")
    return {"seasonal_period": config.forecast.seasonal_period}


def _fit_bundle(
    run: RunContext,
    config: ProjectConfig,
    frames: dict[str, pd.DataFrame],
    *,
    model_id: str,
    cutoff: pd.Timestamp,
    slot: str,
    cache: FeatureBatchCache | None = None,
) -> dict[str, Any]:
    """Fit one adapter at a cutoff, publish its bundle, then prove reload parity."""

    adapter = get_adapter(model_id)
    spec = FeatureSpec(config.data.availability_policy, horizon=config.forecast.horizon)
    supported = _supported_series(run)
    training, record = build_training_set(
        frames,
        cutoff=cutoff,
        spec=spec,
        config=config,
        snapshot_id=run.manifest["snapshot_id"],
        data_source=config.data.source,
        supported_series=supported,
        needs_batch=adapter.needs_training_batch,
        params=_adapter_params(config, model_id),
        name=f"{model_id}/{slot}",
        cache=cache,
    )
    context = FitContext(
        training_run_id=run.run_id,
        adapter_id=model_id,
        availability_policy=spec.availability_policy,
        data_source=config.data.source,  # type: ignore[arg-type]
        execution_mode=config.mode,  # type: ignore[arg-type]
        train_cutoff=training.cutoff,
        selected_epochs=None,
        seed=config.runtime.seed,
        config_sha256=run.manifest["config_sha256"],
        dataset_sha256=run.manifest["snapshot_id"],
        input_contract_version=spec.policy.input_contract_version,
    )
    fitted = adapter.fit(training, context)
    manifest = save_bundle(fitted, run=run, context=context, slot=slot)
    relative = bundle_relative_dir(model_id, slot)
    loaded = load_bundle(
        BundleRef("running", run.root, relative),
        expected_policy=spec.availability_policy,
        running=run,
    )
    if loaded.bundle_id != manifest["bundle_id"]:
        raise ValueError("Reloaded bundle identity differs")
    original = adapter.predict_raw(fitted, training.request)
    reloaded = adapter.predict_raw(loaded.fitted, training.request)
    parity = compare_raw_predictions(original, reloaded)
    if not parity["passed"]:
        raise ValueError(f"RELOAD_PARITY_FAILED for {model_id}/{slot}")
    write_run_artifact(
        run,
        f"checks/{model_id}-{slot}-fit-predictions.csv",
        lambda path: original.assign(
            target_date=original["target_date"].dt.strftime("%Y-%m-%d")
        ).to_csv(path, index=False, float_format="%.17g"),
        kind="verification",
    )
    write_json_artifact(
        run,
        f"checks/{model_id}-{slot}-reload-parity.json",
        {**parity, "bundle_id": manifest["bundle_id"], "grid": record},
        kind="verification",
    )
    return {
        "bundle": f"{run.run_id}:{relative}",
        "bundle_id": manifest["bundle_id"],
        "bundle_dir": str(run.root / relative),
        "reload_parity": parity,
    }


def compare_raw_predictions(
    left: pd.DataFrame, right: pd.DataFrame, model_id: str | None = None
) -> dict[str, Any]:
    """Exact keys first, then the model contract's CPU tree/seasonal tolerance."""

    rtol, atol = (1e-5, 1e-4) if model_id == "gru_global" else (RELOAD_RTOL, RELOAD_ATOL)
    key = ["store_nbr", "family", "target_date"]
    a = left.sort_values(key, kind="mergesort").reset_index(drop=True)
    b = right.sort_values(key, kind="mergesort").reset_index(drop=True)
    keys_equal = a[key].equals(b[key])
    x = a["prediction"].to_numpy(dtype=np.float64)
    y = b["prediction"].to_numpy(dtype=np.float64)
    finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
    if not keys_equal or not finite or len(x) == 0:
        return {"passed": False, "keys_equal": keys_equal, "finite": finite, "rows": len(x)}
    absolute = np.abs(x - y)
    relative = absolute / np.maximum(np.abs(y), np.finfo(np.float64).tiny)
    return {
        "passed": bool(np.allclose(x, y, rtol=rtol, atol=atol)),
        "keys_equal": True,
        "finite": True,
        "rows": len(x),
        "rtol": rtol,
        "atol": atol,
        "max_abs_diff": float(absolute.max()),
        "max_rel_diff": float(relative.max()),
        "precision": "float32" if model_id == "gru_global" else "float64",
        "device": "cpu",
    }


def train(
    config: ProjectConfig,
    *,
    model_id: str | None = None,
    origin: date | None = None,
    run_id: str | None = None,
    resume_from: str | None = None,
) -> dict[str, Any]:
    """Fit one explicitly selected model at a cutoff and publish a verified bundle."""

    if model_id is None:
        raise ValueError(
            f"MODEL_ID_REQUIRED: use train --model-id one of {', '.join(SERVING_MODEL_IDS)}"
        )
    get_adapter(model_id)
    if model_id == "gru_global":
        from retail_forecast.recurrent import train_recurrent

        return train_recurrent(config, origin=origin, run_id=run_id, resume_from=resume_from)

    def operation(run: RunContext) -> dict[str, Any]:
        _inputs(run, config)
        active = config
        frames = _verified_frames(run)
        cutoff = pd.Timestamp(origin) if origin else frames["train.csv"]["date"].max()
        result = _fit_bundle(run, active, frames, model_id=model_id, cutoff=cutoff, slot="final")
        summary = validate_dataset(active)
        write_json_artifact(run, "dataset_summary.json", summary.to_dict(), kind="dataset_summary")
        return {key: result[key] for key in ("bundle", "bundle_id", "bundle_dir")}

    return _execute(config, "training", operation, run_id=run_id, resume_from=resume_from)


def _write_request(run: RunContext, request: ForecastRequest) -> Path:
    files, manifest = serialize_request(request)
    for name, data in files.items():
        write_run_artifact(
            run,
            f"request/{name}",
            lambda path, data=data: path.write_bytes(data),
            kind="forecast_request_table",
        )
    write_run_artifact(
        run,
        f"request/{REQUEST_MANIFEST}",
        lambda path: path.write_bytes(canonical_json(manifest)),
        kind="forecast_request",
    )
    return run.root / "request"


def request(
    config: ProjectConfig,
    *,
    origin: date | None = None,
    run_id: str | None = None,
    resume_from: str | None = None,
) -> dict[str, Any]:
    """Resolve raw inputs once into a normalized, label-free request directory."""

    def operation(run: RunContext) -> dict[str, Any]:
        _inputs(run, config)
        frames = _verified_frames(run)
        resolved = pd.Timestamp(origin) if origin else frames["train.csv"]["date"].max()
        spec = FeatureSpec(config.data.availability_policy, horizon=config.forecast.horizon)
        projected = project_snapshot(
            frames,
            snapshot_id=run.manifest["snapshot_id"],
            data_source=config.data.source,
            origin=resolved,
            policy=spec.availability_policy,
            context_days=spec.context_days,
            supported_series=_supported_series(run),
        )
        return {"request_dir": str(_write_request(run, projected))}

    return _execute(config, "request", operation, run_id=run_id, resume_from=resume_from)


def _project_for_bundle(
    run: RunContext, loaded: LoadedModel, origin: date | None
) -> ForecastRequest:
    frames = _verified_frames(run)
    resolved = pd.Timestamp(origin) if origin else frames["train.csv"]["date"].max()
    return project_snapshot(
        frames,
        snapshot_id=run.manifest["snapshot_id"],
        data_source=run.manifest["data_source"],
        origin=resolved,
        policy=loaded.manifest["availability_policy"],
        context_days=loaded.fitted.request_context_days,
        series_keys=loaded.fitted.supported_series,
        supported_series=_supported_series(run),
    )


def _publish_forecast(
    run: RunContext,
    loaded: LoadedModel,
    forecast_request: ForecastRequest,
    *,
    relative: str = "predictions/forecast.csv",
) -> tuple[Path, pd.DataFrame]:
    context = PredictionContext(
        prediction_run_id=run.run_id,
        evaluation_run_id=None,
        data_source=run.manifest["data_source"],
        execution_mode=run.manifest["execution_mode"],
    )
    frame, diagnostics = predict_bundle(loaded, forecast_request, context=context)
    write_run_artifact(run, relative, lambda path: write_forecast_csv(frame, path), kind="forecast")
    write_json_artifact(
        run,
        relative.replace("predictions/", "checks/").replace(".csv", "-diagnostics.json"),
        diagnostics,
        kind="prediction_diagnostics",
    )
    return run.root / relative, frame


def _publish_submission(run: RunContext, frame: pd.DataFrame) -> Path:
    assert run.input_dir is not None
    test = pd.read_csv(safe_path(run.input_dir, "test.csv", regular=True))
    submission = export_kaggle_submission(frame, test)
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
    bundle: str | None = None,
    origin: date | None = None,
    request_dir: Path | None = None,
    kaggle_submission: bool = False,
    run_id: str | None = None,
    resume_from: str | None = None,
) -> dict[str, Any]:
    """Predict only from an explicitly selected verified bundle; never fit or fall back."""

    if bundle is None:
        raise ValueError("BUNDLE_REQUIRED: use predict --bundle <run_id>:models/<model_id>/<slot>")
    if request_dir is not None and (origin is not None or kaggle_submission):
        raise ValueError("--request already fixes the origin and has no Kaggle test table")
    ref = parse_bundle_ref(bundle, artifacts_dir=config.paths.artifacts_dir)
    parents = (ref.training_run_id,) if ref.training_run_id else ()

    def operation(run: RunContext) -> dict[str, Any]:
        loaded = load_bundle(
            ref,
            expected_policy=config.data.availability_policy,
            expected_mode=config.mode,
            expected_source=config.data.source,
        )
        if request_dir is not None:
            contents = capture_request_inputs(run, Path(request_dir))
            set_global_seed(config.runtime.seed)
            forecast_request = request_from_files(contents)
        else:
            _inputs(run, config)
            forecast_request = _project_for_bundle(run, loaded, origin)
        path, frame = _publish_forecast(run, loaded, forecast_request)
        result = {"forecast": str(path), "bundle_id": loaded.bundle_id}
        if kaggle_submission:
            result["submission"] = str(_publish_submission(run, frame))
        return result

    return _execute(
        config,
        "prediction",
        operation,
        run_id=run_id,
        resume_from=resume_from,
        parent_run_ids=parents,
    )


def smoke(
    config: ProjectConfig, *, run_id: str | None = None, resume_from: str | None = None
) -> dict[str, Any]:
    """Exercise evaluation, a seasonal bundle, reload and keyed export on synthetic data."""

    if config.mode != "smoke" or config.data.source != "synthetic":
        raise ValueError("The smoke command requires the smoke synthetic configuration")

    def operation(run: RunContext) -> dict[str, Any]:
        isolated = _stage_inputs(run, config)
        evaluation_path = _evaluate(run, isolated)
        frames = _verified_frames(run)
        cutoff = frames["train.csv"]["date"].max()
        fitted = _fit_bundle(
            run, isolated, frames, model_id="seasonal_naive_7d", cutoff=cutoff, slot="final"
        )
        relative = bundle_relative_dir("seasonal_naive_7d", "final")
        loaded = load_bundle(
            BundleRef("running", run.root, relative),
            expected_policy=config.data.availability_policy,
            running=run,
        )
        forecast_path, frame = _publish_forecast(
            run, loaded, _project_for_bundle(run, loaded, None)
        )
        submission_path = _publish_submission(run, frame)
        summary = validate_dataset(isolated)
        write_json_artifact(run, "dataset_summary.json", summary.to_dict(), kind="dataset_summary")
        details = {
            "evaluation": str(evaluation_path.relative_to(run.root)),
            "bundle": f"{relative}/bundle.json",
            "forecast": str(forecast_path.relative_to(run.root)),
            "submission": str(submission_path.relative_to(run.root)),
            "test_rows": summary.test_rows,
            "bundle_id": fitted["bundle_id"],
        }
        write_json_artifact(run, "smoke_result.json", details, kind="smoke_summary")
        return {
            **{
                name: str(run.root / details[name])
                for name in ("evaluation", "bundle", "forecast", "submission")
            },
            "test_rows": details["test_rows"],
            "bundle_id": details["bundle_id"],
        }

    return _execute(config, "smoke", operation, run_id=run_id, resume_from=resume_from)


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


SANITY_MODEL_ID = "zero"
COMPLETE_LINEAGE = "complete"
RESEARCH_ONLY_LINEAGE = "research_only"


def _read_listed_forecast(root: Path, manifest: dict[str, Any], relative: str) -> pd.DataFrame:
    """Read a forecast artifact once, checking the completed run's listed hash."""

    import hashlib
    import io

    entries = [entry for entry in manifest["artifacts"] if entry["path"] == relative]
    if len(entries) != 1 or entries[0]["kind"] != "forecast":
        raise ValueError(f"Forecast {relative} is not listed by its prediction run")
    data = safe_path(root, relative, regular=True).read_bytes()
    if hashlib.sha256(data).hexdigest() != entries[0]["sha256"]:
        raise ValueError(f"Forecast {relative} failed integrity verification")
    return read_forecast_csv(io.BytesIO(data))


def compare(
    config: ProjectConfig, *, run_id: str | None = None, resume_from: str | None = None
) -> dict[str, Any]:
    """Re-emit serving-eligible comparators through verified bundles as linked runs.

    A training run fits and reloads one bundle per fold and model and completes;
    a separate prediction run then loads those completed bundles; finally one
    comparison run joins labels by key. Zero stays a research-only sanity control.
    """

    if resume_from is not None:
        resolve_run(config, resume_from)
        raise ValueError("RESUME_UNSUPPORTED: current adapters have no eligible checkpoint")

    def fit_folds(run: RunContext) -> dict[str, Any]:
        staged = _stage_inputs(run, config)
        frames = _verified_frames(run)
        folds = make_rolling_origin_folds(staged, frames["train.csv"]["date"])
        cache = FeatureBatchCache.empty()
        bundles: dict[str, dict[str, str]] = {}
        for fold in folds:
            bundles[fold.name] = {
                model_id: _fit_bundle(
                    run,
                    staged,
                    frames,
                    model_id=model_id,
                    cutoff=fold.origin,
                    slot=fold.name,
                    cache=cache,
                )["bundle"]
                for model_id in COMPARATOR_MODEL_IDS
            }
        summary = validate_dataset(staged)
        write_json_artifact(run, "dataset_summary.json", summary.to_dict(), kind="dataset_summary")
        return {"bundles": bundles, "folds": [fold.to_dict() for fold in folds]}

    trained = _execute(config, "training", fit_folds, run_id=None, resume_from=None)
    folds = {item["name"]: item for item in trained["folds"]}

    def predict_folds(run: RunContext) -> dict[str, Any]:
        _stage_inputs(run, config)
        outputs: dict[str, dict[str, str]] = {}
        for fold_name, references in trained["bundles"].items():
            origin = pd.Timestamp(folds[fold_name]["origin"]).date()
            outputs[fold_name] = {}
            for model_id, reference in references.items():
                loaded = load_bundle(
                    parse_bundle_ref(reference, artifacts_dir=config.paths.artifacts_dir),
                    expected_policy=config.data.availability_policy,
                    expected_mode=config.mode,
                    expected_source=config.data.source,
                )
                path, _ = _publish_forecast(
                    run,
                    loaded,
                    _project_for_bundle(run, loaded, origin),
                    relative=f"predictions/{fold_name}/{model_id}/forecast.csv",
                )
                outputs[fold_name][model_id] = str(path.relative_to(run.root))
        return {"forecasts": outputs}

    predicted = _execute(
        config,
        "prediction",
        predict_folds,
        run_id=None,
        resume_from=None,
        parent_run_ids=(trained["run_id"],),
    )
    parents = (trained["run_id"], predicted["run_id"])

    def evaluate_comparison(run: RunContext) -> dict[str, Any]:
        staged = _stage_inputs(run, config)
        frames = _verified_frames(run)
        snapshot_id = run.manifest["snapshot_id"]
        spec = FeatureSpec(config.data.availability_policy, horizon=config.forecast.horizon)
        supported = _supported_series(run)
        parts = []
        lineage = []
        prediction_root = Path(predicted["run_dir"])
        manifest = read_completed_run(
            prediction_root, expected_mode=config.mode, expected_source=config.data.source
        )
        training_manifest = read_completed_run(Path(trained["run_dir"]))
        for other in (manifest, training_manifest):
            if other["snapshot_id"] != snapshot_id:
                raise ValueError("Comparison children used different input snapshots")
        fold_records = make_rolling_origin_folds(staged, frames["train.csv"]["date"])
        if [fold.to_dict() for fold in fold_records] != trained["folds"]:
            raise ValueError("Comparison folds differ from the training run folds")
        for fold in fold_records:
            request_at_origin = project_snapshot(
                frames,
                snapshot_id=snapshot_id,
                data_source=config.data.source,
                origin=fold.origin,
                policy=spec.availability_policy,
                context_days=spec.context_days,
                supported_series=supported,
            )
            features = build_request_feature_batch(
                request_at_origin, spec, supported_series=supported
            ).frame
            labels = benchmark_labels(frames["train.csv"], fold.validation_dates)
            scored = attach_labels(
                features[["store_nbr", "family", "target_date", "onpromotion", "is_holiday_event"]],
                labels[["target_date", "store_nbr", "family", "sales"]],
            )
            scored["promotion_regime"] = np.where(
                scored["onpromotion"].gt(0), "promoted", "not_promoted"
            )
            scored["temporal_regime"] = temporal_regime(scored)
            scored = scored.drop(columns="is_holiday_event").rename(columns={"sales": "y_true"})
            for model_id, relative in predicted["forecasts"][fold.name].items():
                forecast = _read_listed_forecast(prediction_root, manifest, relative)
                if not forecast["training_run_id"].eq(trained["run_id"]).all():
                    raise ValueError("Forecast names another training run")
                forecast["evaluation_run_id"] = run.run_id
                forecast["lineage_status"] = COMPLETE_LINEAGE
                parts.append(_scored(forecast, scored, fold.name))
                lineage.append(
                    {
                        "fold": fold.name,
                        "model_id": model_id,
                        "bundle_id": forecast["bundle_id"].iloc[0],
                        "training_run_id": trained["run_id"],
                        "prediction_run_id": predicted["run_id"],
                    }
                )
            zero = scored[["store_nbr", "family", "target_date"]].copy()
            zero["forecast_origin"] = fold.origin
            zero["horizon"] = (zero["target_date"] - fold.origin).dt.days.astype("int64")
            zero["prediction"] = 0.0
            for column in FORECAST_COLUMNS:
                if column not in zero:
                    zero[column] = None
            zero["model_id"] = SANITY_MODEL_ID
            zero["evaluation_run_id"] = run.run_id
            zero["snapshot_id"] = snapshot_id
            zero["availability_policy"] = spec.availability_policy
            zero["input_contract_version"] = spec.policy.input_contract_version
            zero["data_source"] = config.data.source
            zero["execution_mode"] = config.mode
            zero = zero[list(FORECAST_COLUMNS)]
            zero["lineage_status"] = RESEARCH_ONLY_LINEAGE
            parts.append(_scored(zero, scored, fold.name))

        oof = pd.concat(parts, ignore_index=True)
        key_sets = {
            model: frame[["fold", "store_nbr", "family", "target_date"]]
            .sort_values(["fold", "store_nbr", "family", "target_date"])
            .reset_index(drop=True)
            for model, frame in oof.groupby("model_id", sort=False)
        }
        reference = next(iter(key_sets.values()))
        if any(not frame.equals(reference) for frame in key_sets.values()):
            raise ValueError("Comparators were not scored on identical target keys")
        results = segmented_rmsle(oof)
        results.insert(0, "availability_policy", spec.availability_policy)
        write_run_artifact(
            run,
            "metrics/comparison_results.csv",
            lambda path: results.to_csv(path, index=False),
            kind="segmented_metrics",
        )
        write_run_artifact(
            run,
            "predictions/oof.parquet",
            lambda path: oof.to_parquet(path, index=False),
            kind="oof",
        )
        summary = {
            "run_id": run.run_id,
            "availability_policy": spec.availability_policy,
            "input_contract_version": spec.policy.input_contract_version,
            "snapshot_id": snapshot_id,
            "folds": trained["folds"],
            "lineage": lineage,
            "complete_lineage_models": list(COMPARATOR_MODEL_IDS),
            "research_only_models": [SANITY_MODEL_ID],
            "oof_rows": len(oof),
            "global_rmsle": {
                row.model_id: float(row.rmsle)
                for row in results.loc[results["scope"].eq("global")].itertuples()
            },
            "config_sha256": run.manifest["config_sha256"],
            "empirical_results_are_synthetic": config.data.source == "synthetic",
        }
        write_json_artifact(
            run, "metrics/comparison_manifest.json", summary, kind="experiment_summary"
        )
        del staged
        return {
            "results": str(run.root / "metrics/comparison_results.csv"),
            "oof": str(run.root / "predictions/oof.parquet"),
            "summary": str(run.root / "metrics/comparison_manifest.json"),
            "child_run_ids": list(parents),
        }

    return _execute(
        config,
        "comparison",
        evaluate_comparison,
        run_id=run_id,
        resume_from=None,
        parent_run_ids=parents,
    )


def _scored(forecast: pd.DataFrame, scored: pd.DataFrame, fold: str) -> pd.DataFrame:
    joined = forecast.merge(
        scored, on=["store_nbr", "family", "target_date"], how="inner", validate="one_to_one"
    )
    if len(joined) != len(forecast) or len(joined) != len(scored):
        raise ValueError("Forecast and label keys differ")
    joined.insert(0, "fold", fold)
    joined["y_pred"] = joined["prediction"]
    joined["squared_log_error"] = np.square(np.log1p(joined["y_pred"]) - np.log1p(joined["y_true"]))
    return joined
