"""Resource-only internal training probe; it never scores an outer horizon."""

from __future__ import annotations

import resource
import time
from typing import Any

import pandas as pd

from retail_forecast.artifacts import write_json_artifact
from retail_forecast.config import ProjectConfig
from retail_forecast.features import FeatureSpec
from retail_forecast.recurrent_training import _recipe
from retail_forecast.sequences import SequencePreprocessor
from retail_forecast.training import build_training_set


def recurrent_feasibility(
    config: ProjectConfig, *, run_id: str | None = None, resume_from: str | None = None
) -> dict[str, Any]:
    """Two internal epochs on the full eligible grid, without selecting a recipe."""
    from retail_forecast.models.gru import initialize
    from retail_forecast.models.gru_training import fit_epochs
    from retail_forecast.pipeline import (
        _execute,
        _stage_inputs,
        _supported_series,
        _verified_frames,
    )

    params = _recipe(config)

    def operation(run):
        started = time.monotonic()
        _stage_inputs(run, config)
        frames = _verified_frames(run)
        cutoff = pd.Timestamp(config.validation.origins[0]) - pd.Timedelta(days=16)
        training, grid = build_training_set(
            frames,
            cutoff=cutoff,
            spec=FeatureSpec("asof_v1"),
            config=config,
            snapshot_id=run.manifest["snapshot_id"],
            data_source=config.data.source,
            supported_series=_supported_series(run),
            needs_batch=False,
            needs_sequences=True,
            params=params,
            name="resource-probe",
        )
        pre = SequencePreprocessor.fit(training.sequences, cutoff=cutoff)
        model = initialize(pre, params["hidden_size"], config.runtime.seed)
        try:
            result = fit_epochs(
                model,
                training.sequences,
                pre,
                params=params,
                seed=config.runtime.seed,
                epochs=min(2, params["max_epochs"]),
                identity={"phase": "resource-probe"},
            )
        except BaseException:
            import torch

            write_json_artifact(
                run,
                "metrics/failed-costs.json",
                {
                    "wall_seconds": time.monotonic() - started,
                    "host_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    "gpu_peak_bytes": torch.cuda.max_memory_allocated()
                    if params["device"] == "cuda"
                    else None,
                    "training_seconds": None,
                    "training_tail_unmeasured": True,
                    "outer_scores_computed": False,
                },
                kind="costs",
            )
            raise
        import torch

        output = {
            "phase": "internal_training_only",
            "outer_scores_computed": False,
            "grid": grid,
            "recipe": params,
            "preprocessor_sha256": pre.sha256,
            "timing": result,
            "wall_seconds": time.monotonic() - started,
            "host_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "device_name": torch.cuda.get_device_name() if params["device"] == "cuda" else "cpu",
            "torch_version": torch.__version__,
            "precision": "float32",
            "estimated_max_primary_seconds": result["training_seconds"]
            / len(result["history"])
            * params["max_epochs"]
            * 6,
            "estimate_limitations": "Two epochs; excludes setup and fold/selection variance",
        }
        if output["host_peak_bytes"] > params["host_memory_gib"] * 2**30:
            raise RuntimeError("GRU_HOST_MEMORY_EXCEEDED")
        write_json_artifact(run, "metrics/feasibility.json", output, kind="experiment_summary")
        write_json_artifact(run, "metrics/costs.json", result, kind="costs")
        return {"summary": str(run.root / "metrics/feasibility.json")}

    return _execute(
        config, "recurrent-feasibility", operation, run_id=run_id, resume_from=resume_from
    )
