"""Optional process-isolated local tracking; no inference-time MLflow import."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from retail_forecast.artifacts import RunContext
    from retail_forecast.config import ProjectConfig


def log_baseline_tracking(
    run: RunContext,
    *,
    enabled: bool,
    model_ids: tuple[str, ...],
    tree_model_id: str,
    config: ProjectConfig,
    folds: int,
    results: pd.DataFrame,
    costs: pd.DataFrame,
    paths: list[Path],
    models: list[Path],
    tree_parameters: dict[str, Any],
    versions: dict[str, Any],
) -> dict[str, Any]:
    """Return tracking status after every child-owned store handle has closed.

    A separate process avoids changing the notebook's active run, tracking URI or
    cached database engines. Synchronous child exit is the publication boundary.
    The parent never stores the subprocess request (which contains local paths).
    """

    if not enabled:
        return {"status": "disabled", "reason": "requested"}
    if importlib.util.find_spec("mlflow") is None:
        return {"status": "disabled", "reason": "not_installed"}

    from retail_forecast.artifacts import register_artifact

    tracking_root = run.path("tracking")
    tracking_root.mkdir(parents=True, exist_ok=False)
    request = {
        "root": str(tracking_root),
        "mode": config.mode,
        "run_id": run.run_id,
        "model_ids": model_ids,
        "tree_model_id": tree_model_id,
        "params": {
            "fold_count": folds,
            "horizon": config.forecast.horizon,
            "seed": config.runtime.seed,
            **{f"baseline_{key}": value for key, value in asdict(config.baseline).items()},
        },
        "tree_parameters": tree_parameters,
        "results": results.to_dict(orient="records"),
        "costs": costs.to_dict(orient="records"),
        "paths": [str(path) for path in paths],
        "models": [str(path) for path in models],
        "versions": versions,
    }
    try:
        child = subprocess.run(
            [sys.executable, "-m", "retail_forecast.tracking_runtime"],
            input=json.dumps(request, allow_nan=False).encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
            check=False,
            env={
                **os.environ,
                "MLFLOW_DISABLE_TELEMETRY": "true",
                "MLFLOW_DISABLE_AGENT_HINT": "true",
            },
        )
        status = (
            {"status": "enabled", "database": "tracking/mlflow.db"}
            if child.returncode == 0
            else {"status": "disabled", "reason": "tracking_failed"}
        )
    except (OSError, subprocess.TimeoutExpired):
        # subprocess.run kills and waits for a timed-out process before raising.
        status = {"status": "disabled", "reason": "tracking_failed"}
    for path in sorted(tracking_root.rglob("*")):
        if not path.is_dir():
            register_artifact(run, path.relative_to(run.root).as_posix(), kind="tracking")
    return status


def _track(request: dict[str, Any]) -> None:
    """Run only in the short-lived tracking process using synchronous client calls."""

    from mlflow.tracking import MlflowClient

    root = Path(request["root"])
    client = MlflowClient(tracking_uri=f"sqlite:///{root / 'mlflow.db'}")
    experiment_id = client.create_experiment(
        f"retail-baseline-{request['run_id']}",
        artifact_location=(root / "artifacts").as_uri(),
    )
    for model_id in request["model_ids"]:
        tracking_run = client.create_run(
            experiment_id,
            tags={
                "mlflow.runName": f"{request['mode']}-{model_id}",
                "execution_run_id": request["run_id"],
                "mode": request["mode"],
                "model_family": "tree" if model_id == request["tree_model_id"] else "baseline",
                "production_ready": "false",
            },
        )
        tracking_id = tracking_run.info.run_id
        try:
            parameters = {"model_id": model_id, **request["params"]}
            if model_id == request["tree_model_id"]:
                parameters.update(request["tree_parameters"])
            for name, value in parameters.items():
                client.log_param(tracking_id, name, value)
            scores = [row for row in request["results"] if row["model_id"] == model_id]
            costs = [row for row in request["costs"] if row["model_id"] == model_id]
            for row in scores:
                if row["scope"] in {"global", "fold"}:
                    metric = (
                        "rmsle_global" if row["scope"] == "global" else f"rmsle_{row['segment']}"
                    )
                    client.log_metric(tracking_id, metric, float(row["rmsle"]), synchronous=True)
            client.log_metric(
                tracking_id,
                "duration_seconds",
                sum(row["duration_seconds"] for row in costs),
                synchronous=True,
            )
            client.log_metric(
                tracking_id,
                "process_max_rss_kib",
                max(row["process_max_rss_kib"] for row in costs),
                synchronous=True,
            )
            client.log_dict(tracking_id, request["versions"], "runtime_versions.json")
            for path in request["paths"]:
                client.log_artifact(tracking_id, path)
            if model_id == request["tree_model_id"]:
                for path in request["models"]:
                    client.log_artifact(tracking_id, path, artifact_path="models")
            client.set_terminated(tracking_id)
        except Exception:
            client.set_terminated(tracking_id, status="FAILED")
            raise


if __name__ == "__main__":
    try:
        _track(json.load(sys.stdin))
    except Exception:
        # Raw MLflow errors can include local URIs; parent records a safe category.
        sys.exit(1)
