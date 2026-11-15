"""Safe, staged acquisition for the Kaggle competition files."""

from __future__ import annotations

import contextlib
import logging
import os
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

from retail_forecast.acquisition_security import (
    COMPETITION_MANIFEST,
    AcquisitionManifest,
    copy_verified,
    reject_symlink_path,
    validate_directory,
)
from retail_forecast.data import required_files_exist

LOGGER = logging.getLogger(__name__)
COMPETITION_SLUG = "store-sales-time-series-forecasting"
COMPETITION_ARCHIVE = f"{COMPETITION_SLUG}.zip"


def find_competition_archive(*directories: Path) -> Path | None:
    """Return the first expected competition ZIP found in the given directories."""

    for directory in directories:
        candidate = Path(directory) / COMPETITION_ARCHIVE
        reject_symlink_path(candidate)
        if candidate.is_file():
            return candidate
    return None


def _destination_ready(data_dir: Path, manifest: AcquisitionManifest) -> bool:
    reject_symlink_path(data_dir)
    if required_files_exist(data_dir):
        validate_directory(data_dir, manifest)
        return True
    if data_dir.exists() and (not data_dir.is_dir() or any(data_dir.iterdir())):
        raise RuntimeError("Incomplete acquisition destination; choose a new empty directory")
    return False


def _publish(stage: Path, data_dir: Path, manifest: AcquisitionManifest) -> None:
    validate_directory(stage, manifest)
    # Recheck before atomic rename; never merge staged files with partial raw data.
    if _destination_ready(data_dir, manifest):
        return
    os.replace(stage, data_dir)


def extract_competition_archive(
    archive_path: Path,
    data_dir: Path,
    *,
    manifest: AcquisitionManifest = COMPETITION_MANIFEST,
) -> Path:
    """Extract bounded regular CSV members, verify them and publish atomically."""

    archive_path, data_dir = Path(archive_path), Path(data_dir)
    reject_symlink_path(archive_path)
    if _destination_ready(data_dir, manifest):
        return data_dir
    if not archive_path.is_file():
        raise FileNotFoundError("Competition archive does not exist")
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ZipFile(archive_path) as archive:
            members = {}
            total = 0
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                kind = stat.S_IFMT(member.external_attr >> 16)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "\\" in member.filename
                    or kind not in {0, stat.S_IFREG, stat.S_IFDIR}
                    or (kind == stat.S_IFDIR and not member.is_dir())
                ):
                    raise RuntimeError("Competition ZIP contains an unsafe member")
                total += member.file_size
                if total > sum(manifest.sizes.values()):
                    raise RuntimeError("Competition ZIP exceeds trusted declared byte limit")
                name = path.name
                if name in manifest.sizes and not member.is_dir():
                    if name in members:
                        raise RuntimeError("Competition ZIP must contain exactly one copy of CSVs")
                    members[name] = member
            if members.keys() != manifest.sizes.keys():
                raise RuntimeError("Competition ZIP must contain exactly one copy of every CSV")
            if any(members[name].file_size != size for name, size in manifest.sizes.items()):
                raise RuntimeError("Competition ZIP differs from trusted declared sizes")
            with tempfile.TemporaryDirectory(prefix=".acquire-", dir=data_dir.parent) as temporary:
                stage = Path(temporary) / "raw"
                stage.mkdir()
                for name, member in members.items():
                    with archive.open(member) as source, (stage / name).open("xb") as target:
                        copy_verified(source, target, name, manifest)
                _publish(stage, data_dir, manifest)
    except (BadZipFile, OSError) as error:
        raise RuntimeError("Competition ZIP acquisition failed; destination unchanged") from error
    return data_dir


def download_competition_data(
    data_dir: Path,
    archive_path: Path | None = None,
    *,
    manifest: AcquisitionManifest = COMPETITION_MANIFEST,
) -> Path:
    """Prepare verified CSVs from an archive or a staged official-client download."""

    data_dir = Path(data_dir)
    if _destination_ready(data_dir, manifest):
        return data_dir
    if archive_path is not None:
        return extract_competition_archive(archive_path, data_dir, manifest=manifest)
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError("Install the 'notebook' extra to download from Kaggle") from error
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".acquire-", dir=data_dir.parent) as temporary:
        download = Path(temporary) / "download"
        download.mkdir()
        stage = Path(temporary) / "raw"
        stage.mkdir()
        # A third-party client may echo its environment or raw exception text.
        # Capture its output and suppress logging; expose only a safe fixed error.
        previous_disable = logging.root.manager.disable
        try:
            logging.disable(logging.CRITICAL)
            with (
                open(os.devnull, "w") as discarded,
                contextlib.redirect_stdout(discarded),
                contextlib.redirect_stderr(discarded),
            ):
                kagglehub.competition_download(COMPETITION_SLUG, output_dir=str(download))
            validate_directory(download, manifest)
            for name in manifest.sizes:
                with (download / name).open("rb") as source, (stage / name).open("xb") as target:
                    copy_verified(source, target, name, manifest)
            _publish(stage, data_dir, manifest)
        except Exception:
            raise RuntimeError("Competition download failed; destination unchanged") from None
        finally:
            logging.disable(previous_disable)
    return data_dir


def acquire_competition_data(
    data_dir: Path,
    search_directories: tuple[Path, ...],
    *,
    token: str | None = None,
    token_reader: Callable[[str], str] | None = None,
) -> Path:
    """Resolve a validated existing dataset, local ZIP or temporary secret token."""

    if _destination_ready(Path(data_dir), COMPETITION_MANIFEST):
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
        except Exception:
            raise RuntimeError(
                "Competition acquisition failed; check credentials and inputs"
            ) from None
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
