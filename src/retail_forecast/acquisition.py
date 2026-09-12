"""Safe, idempotent acquisition for the Kaggle competition files."""

from __future__ import annotations

import logging
from pathlib import Path

from retail_forecast.data import required_files_exist

LOGGER = logging.getLogger(__name__)
COMPETITION_SLUG = "store-sales-time-series-forecasting"


def download_competition_data(data_dir: Path) -> Path:
    """Download only when required CSVs are absent, using Kaggle's official client."""

    if required_files_exist(data_dir):
        LOGGER.info("All competition files already exist; skipping download")
        return data_dir

    try:
        import kagglehub
    except ImportError as error:  # pragma: no cover - exercised only outside notebook extras
        raise RuntimeError("Install the 'notebook' extra to download from Kaggle") from error

    data_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading Kaggle competition files to %s", data_dir)
    kagglehub.competition_download(COMPETITION_SLUG, output_dir=str(data_dir))
    if not required_files_exist(data_dir):
        raise RuntimeError(
            "Kaggle download completed but the seven expected CSV files were not found"
        )
    return data_dir
