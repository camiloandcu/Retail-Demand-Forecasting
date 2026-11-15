"""Small run setup shared by the numbered notebook consumers."""

from __future__ import annotations

from dataclasses import replace

from retail_forecast.artifacts import RunContext, allocate_run
from retail_forecast.config import ProjectConfig
from retail_forecast.data import write_synthetic_dataset

NOTEBOOK_CONFIGS = {
    ("full", "competition_v1"): "configs/full.yaml",
    ("full", "asof_v1"): "configs/asof.yaml",
    ("smoke", "competition_v1"): "configs/smoke.yaml",
    ("smoke", "asof_v1"): "configs/smoke-asof.yaml",
}


def notebook_config_path(mode: str, policy: str) -> str:
    """Map an explicit mode and availability policy to their declared config."""

    try:
        return NOTEBOOK_CONFIGS[(mode, policy)]
    except KeyError as error:
        raise ValueError(
            "RETAIL_FORECAST_MODE must be full/smoke and RETAIL_FORECAST_POLICY "
            "must be competition_v1/asof_v1"
        ) from error


def prepare_notebook_run(config: ProjectConfig, *, kind: str) -> tuple[RunContext, ProjectConfig]:
    """Allocate a fresh notebook execution and isolate its synthetic input staging.

    Acquisition remains an explicit notebook step. Call ``capture_inputs`` after
    acquisition/validation and before reading frames for scientific computation.
    The context retains the original logical config for portable hashing.
    """
    run = allocate_run(config, kind=kind)
    active = config
    if config.data.source == "synthetic":
        raw_dir = run.path("inputs/synthetic")
        active = replace(config, paths=replace(config.paths, raw_data_dir=raw_dir))
        write_synthetic_dataset(active)
    return run, active
