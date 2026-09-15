"""Leakage-safe rolling-origin split contracts for multi-horizon forecasting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from retail_forecast.config import PROJECT_HORIZON, ProjectConfig
from retail_forecast.io_utils import write_json

MIN_FOLDS: Final = 2
MAX_FOLDS: Final = 3


class SplitContractError(ValueError):
    """Raised when a temporal split could overlap or miss the forecast contract."""


@dataclass(frozen=True)
class RollingOriginFold:
    """One expanding-window split whose validation starts after the forecast origin."""

    name: str
    origin: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    horizon: int

    def __post_init__(self) -> None:
        timestamps = (
            self.origin,
            self.train_start,
            self.train_end,
            self.validation_start,
            self.validation_end,
        )
        if any(timestamp.tz is not None for timestamp in timestamps):
            raise SplitContractError("Split timestamps must be timezone-naive")
        if self.horizon != PROJECT_HORIZON:
            raise SplitContractError(
                f"Validation must match the real {PROJECT_HORIZON}-day test horizon"
            )
        if self.train_end != self.origin:
            raise SplitContractError("The training window must end at the forecast origin")
        if self.train_start > self.train_end:
            raise SplitContractError("Training start must not follow training end")
        if self.validation_start != self.origin + pd.offsets.Day(1):
            raise SplitContractError("Validation must start one day after the origin")
        expected_end = self.origin + pd.offsets.Day(self.horizon)
        if self.validation_end != expected_end:
            raise SplitContractError("Validation end does not match the declared horizon")
        if self.train_end >= self.validation_start:
            raise SplitContractError("Training and validation windows overlap")

    @property
    def validation_dates(self) -> pd.DatetimeIndex:
        """Return the exact contiguous dates scored by this fold."""

        return pd.date_range(self.validation_start, self.validation_end, freq="D")

    def to_dict(self) -> dict[str, str | int]:
        """Serialize the split without implementation-specific timestamp objects."""

        raw = asdict(self)
        return {
            key: value.date().isoformat() if isinstance(value, pd.Timestamp) else value
            for key, value in raw.items()
        }


def _normalize_dates(values: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    parsed = pd.DatetimeIndex(pd.to_datetime(values, errors="raise")).normalize()
    if parsed.has_duplicates:
        parsed = pd.DatetimeIndex(parsed.unique())
    return parsed.sort_values()


def make_rolling_origin_folds(
    config: ProjectConfig, available_dates: pd.Series | pd.DatetimeIndex
) -> tuple[RollingOriginFold, ...]:
    """Build two or three validated expanding folds from pre-registered origins."""

    if not MIN_FOLDS <= len(config.validation.origins) <= MAX_FOLDS:
        raise SplitContractError(
            f"Rolling-origin validation requires {MIN_FOLDS} to {MAX_FOLDS} folds"
        )
    if config.forecast.horizon != PROJECT_HORIZON:
        raise SplitContractError("Configured horizon does not match the Kaggle test grid")

    dates = _normalize_dates(available_dates)
    if dates.empty:
        raise SplitContractError("Available target dates cannot be empty")
    date_set = set(dates)
    folds: list[RollingOriginFold] = []
    for index, origin_value in enumerate(config.validation.origins, start=1):
        origin = pd.Timestamp(origin_value)
        fold = RollingOriginFold(
            name=f"fold_{index}",
            origin=origin,
            train_start=dates.min(),
            train_end=origin,
            validation_start=origin + pd.offsets.Day(1),
            validation_end=origin + pd.offsets.Day(config.forecast.horizon),
            horizon=config.forecast.horizon,
        )
        missing = fold.validation_dates.difference(date_set)
        if len(missing):
            formatted = [value.date().isoformat() for value in missing]
            raise SplitContractError(f"{fold.name} validation dates are unavailable: {formatted}")
        if origin not in date_set:
            raise SplitContractError(f"{fold.name} origin is not an observed target date")
        folds.append(fold)

    if any(left.origin >= right.origin for left, right in zip(folds, folds[1:], strict=False)):
        raise SplitContractError("Fold origins must be strictly chronological")
    return tuple(folds)


def validate_frame_against_fold(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    fold: RollingOriginFold,
    date_column: str = "date",
) -> None:
    """Reject future rows in train and any train/validation temporal overlap."""

    train_dates = pd.to_datetime(train_frame[date_column], errors="raise")
    validation_dates = pd.to_datetime(validation_frame[date_column], errors="raise")
    if train_dates.max() > fold.origin:
        raise SplitContractError("Training data contains rows after the forecast origin")
    if validation_dates.min() <= fold.origin:
        raise SplitContractError("Validation contains rows at or before the forecast origin")
    if set(train_dates).intersection(set(validation_dates)):
        raise SplitContractError("Training and validation dates overlap")
    if set(validation_dates.unique()) != set(fold.validation_dates):
        raise SplitContractError("Validation dates do not equal the fold horizon")


def split_manifest(config: ProjectConfig, folds: tuple[RollingOriginFold, ...]) -> dict[str, Any]:
    """Create a small, hash-addressed manifest for the split protocol."""

    payload: dict[str, Any] = {
        "horizon": config.forecast.horizon,
        "fold_count": len(folds),
        "strategy": "expanding_rolling_origin",
        "train_includes_origin": True,
        "validation_starts_next_day": True,
        "config_file": config.config_path.name,
        "config_sha256": hashlib.sha256(config.config_path.read_bytes()).hexdigest(),
        "folds": [fold.to_dict() for fold in folds],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {**payload, "sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def export_split_manifest(
    config: ProjectConfig, folds: tuple[RollingOriginFold, ...], path: Path
) -> Path:
    """Write only split metadata, never processed observations."""

    write_json(path, split_manifest(config, folds))
    return path
