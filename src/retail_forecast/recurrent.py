"""Temporal recurrent training, checkpoint recovery and lineage-complete evaluation.

Each fold completes its own training run. Recovery can reuse verified earlier
folds and continue the interrupted phase in a new linked run. No outer actuals
enter selection, preprocessing or prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retail_forecast.artifacts import (
    read_completed_run,
    resolve_run,
    write_json_artifact,
    write_run_artifact,
)
from retail_forecast.bundles import load_bundle, parse_bundle_ref
from retail_forecast.config import ProjectConfig
from retail_forecast.data_contracts import benchmark_labels, project_snapshot
from retail_forecast.experiments import segmented_rmsle, temporal_regime
from retail_forecast.features import FeatureSpec, build_request_feature_batch
from retail_forecast.metrics import rmsle
from retail_forecast.provenance import digest_json, resolved_config
from retail_forecast.recurrent_training import _checkpoint_source, _recipe, _train_one
from retail_forecast.recurrent_training import train_recurrent as train_recurrent
from retail_forecast.splits import make_rolling_origin_folds


def recurrent_experiment(
    config: ProjectConfig, *, run_id: str | None = None, resume_from: str | None = None
) -> dict[str, Any]:
    """Train three folds, predict from completed bundles, then join actuals by key."""
    from retail_forecast.pipeline import (
        _execute,
        _project_for_bundle,
        _publish_forecast,
        _read_listed_forecast,
        _scored,
        _stage_inputs,
        _supported_series,
        _verified_frames,
    )

    _recipe(config)
    if len(config.validation.origins) != 3:
        raise ValueError("Recurrent study requires all three declared folds")
    completed = []
    source_slot = None
    if resume_from:
        _, info = _checkpoint_source(config, resume_from)
        completed = info["completed_folds"]
        source_slot = info["slot"]
        for item in completed:
            previous = read_completed_run(resolve_run(config, item["run_id"]))
            if previous["config_sha256"] != digest_json(resolved_config(config)):
                raise ValueError("Prior completed fold configuration differs")
    for i, origin in enumerate(config.validation.origins):
        slot = f"fold_{i + 1}"
        if i < len(completed):
            if completed[i]["slot"] != slot or completed[i]["cutoff"] != origin.isoformat():
                raise ValueError("Prior fold identity mismatch")
            continue
        if resume_from and slot != source_slot:
            raise ValueError("Resume source does not match next fold")
        completed.append(
            _train_one(
                config,
                cutoff=pd.Timestamp(origin),
                slot=slot,
                completed=list(completed),
                resume_from=resume_from,
            )
        )
        resume_from = None
    parents = tuple(x["run_id"] for x in completed)

    def predictions(run):
        _stage_inputs(run, config)
        paths = {}
        for item in completed:
            loaded = load_bundle(
                parse_bundle_ref(item["bundle"], artifacts_dir=config.paths.artifacts_dir),
                expected_policy="asof_v1",
                expected_mode=config.mode,
                expected_source=config.data.source,
            )
            if loaded.manifest["snapshot_id"] != run.manifest["snapshot_id"]:
                raise ValueError("Fold and prediction snapshots differ")
            path, _ = _publish_forecast(
                run,
                loaded,
                _project_for_bundle(run, loaded, pd.Timestamp(item["cutoff"]).date()),
                relative=f"predictions/{item['slot']}/gru_global/forecast.csv",
            )
            paths[item["slot"]] = str(path.relative_to(run.root))
        return {"forecasts": paths}

    predicted = _execute(
        config, "prediction", predictions, run_id=None, resume_from=None, parent_run_ids=parents
    )

    def evaluate(run):
        _stage_inputs(run, config)
        frames = _verified_frames(run)
        folds = make_rolling_origin_folds(config, frames["train.csv"]["date"])
        prediction_root = Path(predicted["run_dir"])
        manifest = read_completed_run(prediction_root)
        parts = []
        for fold in folds:
            req = project_snapshot(
                frames,
                snapshot_id=run.manifest["snapshot_id"],
                data_source=config.data.source,
                origin=fold.origin,
                policy="asof_v1",
                context_days=56,
                supported_series=_supported_series(run),
            )
            features = build_request_feature_batch(
                req, FeatureSpec("asof_v1"), supported_series=_supported_series(run)
            ).frame
            labels = benchmark_labels(frames["train.csv"], fold.validation_dates)
            scored = features[
                ["store_nbr", "family", "target_date", "onpromotion", "is_holiday_event"]
            ].merge(
                labels[["store_nbr", "family", "target_date", "sales"]],
                on=["store_nbr", "family", "target_date"],
                validate="one_to_one",
            )
            if len(scored) != len(req.series_keys) * 16:
                raise ValueError("Incomplete recurrent target grid")
            scored["promotion_regime"] = np.where(
                scored["onpromotion"].gt(0), "promoted", "not_promoted"
            )
            scored["temporal_regime"] = temporal_regime(scored)
            scored = scored.drop(columns="is_holiday_event").rename(columns={"sales": "y_true"})
            forecast = _read_listed_forecast(
                prediction_root, manifest, predicted["forecasts"][fold.name]
            )
            forecast["evaluation_run_id"] = run.run_id
            forecast["lineage_status"] = "complete"
            parts.append(_scored(forecast, scored, fold.name))
        oof = pd.concat(parts, ignore_index=True)
        expected = len(_supported_series(run)) * 16 * 3
        if (
            len(oof) != expected
            or oof.duplicated(["fold", "store_nbr", "family", "target_date"]).any()
        ):
            raise ValueError("Incomplete recurrent study")
        metrics = segmented_rmsle(oof)
        write_run_artifact(
            run,
            "metrics/recurrent_results.csv",
            lambda p: metrics.to_csv(p, index=False),
            kind="segmented_metrics",
        )
        write_run_artifact(
            run, "predictions/oof.parquet", lambda p: oof.to_parquet(p, index=False), kind="oof"
        )
        costs = {
            "primary_training_seconds": sum(x["training_seconds"] for x in completed),
            "fold_costs": [x["costs"] for x in completed],
            "seed_pilots_performed": False,
            "all_attempt_wall_seconds": sum(
                x["costs"]["attempt_wall_seconds"]
                + sum(a["wall_seconds"] or 0 for a in x["costs"]["predecessor_wall"])
                for x in completed
            ),
            "unmeasured_partial_epoch_possible": any(
                x["costs"]["predecessor_wall"] for x in completed
            ),
            "setup_download_included": False,
        }
        write_json_artifact(run, "metrics/recurrent_costs.json", costs, kind="costs")
        selections = [
            {"fold": x["slot"], "selected_epochs": x["selected_epochs"], "stages": x["stages"]}
            for x in completed
        ]
        write_json_artifact(run, "metrics/recurrent_selection.json", selections, kind="selection")
        true, pred = oof["y_true"].to_numpy(), oof["y_pred"].to_numpy()
        denominator = float(true.sum())
        summary = {
            "run_id": run.run_id,
            "model_id": "gru_global",
            "availability_policy": "asof_v1",
            "snapshot_id": run.manifest["snapshot_id"],
            "oof_rows": len(oof),
            "expected_rows": expected,
            "folds": [f.to_dict() for f in folds],
            "training_runs": list(parents),
            "prediction_run_id": predicted["run_id"],
            "empirical_results_are_synthetic": config.data.source == "synthetic",
            "rmsle": rmsle(true, pred),
            "mae": float(np.mean(np.abs(pred - true))),
            "signed_bias": float(np.mean(pred - true)),
            "wape": float(np.abs(pred - true).sum() / denominator) if denominator else None,
            "primary_training_seconds": costs["primary_training_seconds"],
            "limitations": [
                "Historical promotion and calendar vintage assumptions",
                "No calibrated intervals or inventory benefit evidence",
            ],
        }
        write_json_artifact(
            run, "metrics/recurrent_manifest.json", summary, kind="experiment_summary"
        )
        return {
            "summary": str(run.root / "metrics/recurrent_manifest.json"),
            "oof": str(run.root / "predictions/oof.parquet"),
            "training_runs": list(parents),
            "prediction_run_id": predicted["run_id"],
        }

    return _execute(
        config,
        "recurrent-study",
        evaluate,
        run_id=run_id,
        resume_from=None,
        parent_run_ids=(*parents, predicted["run_id"]),
    )
