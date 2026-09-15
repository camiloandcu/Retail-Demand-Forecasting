"""Safe, idempotent acquisition for the Kaggle competition files."""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from retail_forecast.data import FILE_COLUMNS, required_files_exist

LOGGER = logging.getLogger(__name__)
COMPETITION_SLUG = "store-sales-time-series-forecasting"
COMPETITION_ARCHIVE = f"{COMPETITION_SLUG}.zip"


def find_competition_archive(*directories: Path) -> Path | None:
    """Return the first expected competition ZIP found in the given directories."""

    for directory in directories:
        candidate = Path(directory) / COMPETITION_ARCHIVE
        if candidate.is_file():
            return candidate
    return None


def extract_competition_archive(archive_path: Path, data_dir: Path) -> Path:
    """Extract only the seven expected CSVs from a manually downloaded archive."""

    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Competition archive does not exist: {archive_path}")
    if required_files_exist(data_dir):
        LOGGER.info("All competition files already exist; skipping extraction")
        return data_dir

    try:
        with ZipFile(archive_path) as archive:
            members_by_name: dict[str, list[str]] = {name: [] for name in FILE_COLUMNS}
            for member in archive.namelist():
                basename = Path(member).name
                if basename in members_by_name and not member.endswith("/"):
                    members_by_name[basename].append(member)
            invalid = {
                name: members for name, members in members_by_name.items() if len(members) != 1
            }
            if invalid:
                raise RuntimeError(
                    "Competition ZIP must contain exactly one copy of every expected CSV; "
                    f"invalid entries: {invalid}"
                )

            data_dir.mkdir(parents=True, exist_ok=True)
            for filename, members in members_by_name.items():
                with archive.open(members[0]) as source, (data_dir / filename).open("wb") as target:
                    shutil.copyfileobj(source, target)
    except BadZipFile as error:
        raise RuntimeError(f"Invalid competition ZIP: {archive_path}") from error

    if not required_files_exist(data_dir):  # pragma: no cover - defensive assertion
        raise RuntimeError("Archive extraction did not produce every expected CSV")
    return data_dir


def download_competition_data(data_dir: Path, archive_path: Path | None = None) -> Path:
    """Prepare raw CSVs from an archive or Kaggle's official client."""

    if required_files_exist(data_dir):
        LOGGER.info("All competition files already exist; skipping download")
        return data_dir

    if archive_path is not None:
        return extract_competition_archive(archive_path, data_dir)

    try:
        import kagglehub
    except ImportError as error:  # pragma: no cover - exercised outside notebook extras
        raise RuntimeError("Install the 'notebook' extra to download from Kaggle") from error

    data_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading Kaggle competition files to %s", data_dir)
    kagglehub.competition_download(COMPETITION_SLUG, output_dir=str(data_dir))
    if not required_files_exist(data_dir):
        raise RuntimeError(
            "Kaggle download completed but the seven expected CSV files were not found"
        )
    return data_dir


def acquire_competition_data(
    data_dir: Path,
    search_directories: tuple[Path, ...],
    *,
    token: str | None = None,
    token_reader: Callable[[str], str] | None = None,
) -> Path:
    """Resolve existing CSVs, a token, or a local ZIP without exposing credentials."""

    if required_files_exist(data_dir):
        return data_dir

    archive_path = find_competition_archive(*search_directories)
    resolved_token = (token if token is not None else os.getenv("KAGGLE_API_TOKEN", "")).strip()
    if not resolved_token and archive_path is None and token_reader is not None:
        resolved_token = token_reader(
            "Kaggle API token (press Enter to continue without one): "
        ).strip()

    if resolved_token:
        previous_token = os.environ.get("KAGGLE_API_TOKEN")
        os.environ["KAGGLE_API_TOKEN"] = resolved_token
        try:
            return download_competition_data(data_dir)
        finally:
            if previous_token is None:
                os.environ.pop("KAGGLE_API_TOKEN", None)
            else:
                os.environ["KAGGLE_API_TOKEN"] = previous_token

    if archive_path is not None:
        return download_competition_data(data_dir, archive_path)

    raise FileNotFoundError(
        f"No Kaggle token, complete data/raw directory, or data/{COMPETITION_ARCHIVE} was found."
    )
