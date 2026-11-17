"""Shared deterministic test fixtures."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

from retail_forecast.config import PathsConfig, ProjectConfig, load_config
from retail_forecast.data import write_synthetic_dataset


@pytest.fixture
def synthetic_config(tmp_path: Path) -> ProjectConfig:
    """Return a smoke config whose generated files stay inside pytest temp space."""

    base = load_config("configs/smoke.yaml")
    config = replace(
        base,
        paths=PathsConfig(
            raw_data_dir=tmp_path / "data",
            artifacts_dir=tmp_path / "artifacts",
        ),
    )
    write_synthetic_dataset(config)
    return config


@pytest.fixture
def sequence_data(tmp_path):
    import pandas as pd

    from retail_forecast.data import load_frames
    from retail_forecast.training import prepare_feature_frames

    config = load_config("configs/gru-smoke.yaml")
    config = replace(config, paths=PathsConfig(tmp_path / "data", tmp_path / "runs"))
    write_synthetic_dataset(config)
    frames = prepare_feature_frames(load_frames(config.paths.raw_data_dir))
    series = tuple(sorted(set(map(tuple, frames["train.csv"][["store_nbr", "family"]].to_numpy()))))
    cutoff = pd.Timestamp(config.validation.origins[0])
    return config, frames, series, cutoff
