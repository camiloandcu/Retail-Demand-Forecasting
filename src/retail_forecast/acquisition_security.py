"""Trusted competition manifest and bounded, contained acquisition primitives."""

from __future__ import annotations

import hashlib
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from retail_forecast.data import EXPECTED_KAGGLE_SHA256

# Observed 2026-10-23 in local data/raw after matching every file against the
# existing EXPECTED_KAGGLE_SHA256 registry. Sizes are exact bytes, not ZIP claims.
EXPECTED_KAGGLE_BYTES = {
    "holidays_events.csv": 22309,
    "oil.csv": 20580,
    "sample_submission.csv": 342153,
    "stores.csv": 1387,
    "test.csv": 1022269,
    "train.csv": 121800373,
    "transactions.csv": 1552637,
}


@dataclass(frozen=True)
class AcquisitionManifest:
    """Operator-reviewed exact byte sizes and hashes; fixtures supply their own."""

    sizes: dict[str, int]
    hashes: dict[str, str]

    def __post_init__(self) -> None:
        if (
            not self.sizes
            or self.sizes.keys() != self.hashes.keys()
            or any(Path(name).name != name or name in {".", ".."} for name in self.sizes)
            or any(size < 0 for size in self.sizes.values())
        ):
            raise ValueError("Invalid trusted acquisition manifest")


COMPETITION_MANIFEST = AcquisitionManifest(EXPECTED_KAGGLE_BYTES, EXPECTED_KAGGLE_SHA256)


def reject_symlink_path(path: Path) -> Path:
    """Validate lexical ancestors before resolution can follow a symlink."""

    absolute = Path(path).absolute()
    if ".." in absolute.parts:
        raise RuntimeError("Acquisition paths must not contain traversal components")
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise RuntimeError("Acquisition paths must not contain symlinks")
    return absolute


def copy_verified(source: BinaryIO, target: BinaryIO, name: str, manifest: AcquisitionManifest):
    """Bound actual streamed bytes and verify exact size/hash before publication."""

    size = 0
    digest = hashlib.sha256()
    limit = manifest.sizes[name]
    while chunk := source.read(min(1024 * 1024, limit - size + 1)):
        size += len(chunk)
        if size > limit:
            raise RuntimeError("Acquisition exceeded trusted byte limit")
        digest.update(chunk)
        target.write(chunk)
    if size != limit or digest.hexdigest() != manifest.hashes[name]:
        raise RuntimeError("Acquisition content does not match trusted manifest")


def validate_directory(directory: Path, manifest: AcquisitionManifest) -> None:
    """Verify files without accepting symlink escapes or unbounded reads."""

    reject_symlink_path(directory)
    for name in manifest.sizes:
        path = directory / name
        reject_symlink_path(path)
        if not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
            raise RuntimeError("Acquisition is missing an expected regular file")
        if path.stat().st_size != manifest.sizes[name]:
            raise RuntimeError("Acquisition file exceeds or differs from trusted size")
        with path.open("rb") as source:
            # Do not materialize content or produce another copy while hashing.
            digest = hashlib.sha256()
            remaining = manifest.sizes[name]
            while chunk := source.read(min(1024 * 1024, remaining + 1)):
                remaining -= len(chunk)
                if remaining < 0:
                    raise RuntimeError("Acquisition exceeded trusted byte limit")
                digest.update(chunk)
            if remaining or digest.hexdigest() != manifest.hashes[name]:
                raise RuntimeError("Acquisition content does not match trusted manifest")
