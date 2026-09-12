"""Shared deterministic test fixtures."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

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
