"""Small run setup shared by the numbered notebook consumers."""

from __future__ import annotations

from dataclasses import replace

from retail_forecast.artifacts import RunContext, allocate_run
from retail_forecast.config import ProjectConfig
from retail_forecast.data import write_synthetic_dataset


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
