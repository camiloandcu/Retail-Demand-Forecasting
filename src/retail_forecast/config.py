"""Small, validated configuration objects loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

PROJECT_HORIZON = 16


class ConfigError(ValueError):
    """Raised when a project configuration violates its schema or contract."""


@dataclass(frozen=True)
class PathsConfig:
    """Resolved input and output paths."""

    raw_data_dir: Path
    artifacts_dir: Path


@dataclass(frozen=True)
class DataConfig:
    """Dataset source and expected panel dimensions."""

    source: str
    expected_stores: int
    expected_families: int
    fixture_start_date: date
    fixture_history_days: int


@dataclass(frozen=True)
class ForecastConfig:
    """Forecast horizon and baseline seasonality."""

    horizon: int
    seasonal_period: int


@dataclass(frozen=True)
class ValidationConfig:
    """Pre-registered rolling-origin cutoffs."""

    origins: tuple[date, ...]


@dataclass(frozen=True)
class BaselineConfig:
    """Pre-registered baseline and single-tree experiment budget."""

    sanity_model: str
    seasonal_average_weeks: int
    training_origin_stride_days: int
    max_training_origins: int
    neural_min_relative_improvement: float
    oof_git_limit_mb: int
    tree_model: str
    tree_n_estimators: int
    tree_learning_rate: float
    tree_num_leaves: int
    tree_min_child_samples: int
    tree_feature_fraction: float
    tree_l2: float


@dataclass(frozen=True)
class RuntimeConfig:
    """Cross-cutting reproducibility settings."""

    seed: int
    log_level: str


@dataclass(frozen=True)
class ProjectConfig:
    """Complete project configuration shared by all entry points."""

    mode: str
    paths: PathsConfig
    data: DataConfig
    forecast: ForecastConfig
    validation: ValidationConfig
    baseline: BaselineConfig
    runtime: RuntimeConfig
    config_path: Path
    project_root: Path


SCHEMA: dict[str, set[str]] = {
    "project": {"mode"},
    "paths": {"raw_data_dir", "artifacts_dir"},
    "data": {
        "source",
        "expected_stores",
        "expected_families",
        "fixture_start_date",
        "fixture_history_days",
    },
    "forecast": {"horizon", "seasonal_period"},
    "validation": {"origins"},
    "baseline": {
        "sanity_model",
        "seasonal_average_weeks",
        "training_origin_stride_days",
        "max_training_origins",
        "neural_min_relative_improvement",
        "oof_git_limit_mb",
        "tree_model",
        "tree_n_estimators",
        "tree_learning_rate",
        "tree_num_leaves",
        "tree_min_child_samples",
        "tree_feature_fraction",
        "tree_l2",
    },
    "runtime": {"seed", "log_level"},
}


def _mapping(value: Any, section: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"Section '{section}' must be a mapping")
    expected = SCHEMA[section]
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ConfigError(f"Invalid keys in '{section}': missing={missing}, extra={extra}")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"'{name}' must be a positive integer")
    return value


def _bounded_float(value: Any, name: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{name}' must be numeric")
    result = float(value)
    if not lower < result <= upper:
        raise ConfigError(f"'{name}' must be in ({lower}, {upper}]")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{name}' must be numeric")
    result = float(value)
    if result < 0:
        raise ConfigError(f"'{name}' must be non-negative")
    return result


def _parse_date(value: Any, name: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as error:
        raise ConfigError(f"'{name}' must use ISO date format YYYY-MM-DD") from error


def load_config(path: str | Path) -> ProjectConfig:
    """Load a YAML file, reject unknown keys, and validate project invariants."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != set(SCHEMA):
        raise ConfigError(f"Top-level sections must be exactly {sorted(SCHEMA)}")

    sections = {name: _mapping(raw[name], name) for name in SCHEMA}
    project_root = config_path.parent.parent
    mode = str(sections["project"]["mode"])
    source = str(sections["data"]["source"])
    log_level = str(sections["runtime"]["log_level"]).upper()

    if mode not in {"smoke", "full"}:
        raise ConfigError("'project.mode' must be 'smoke' or 'full'")
    if source not in {"synthetic", "kaggle"}:
        raise ConfigError("'data.source' must be 'synthetic' or 'kaggle'")
    if mode == "smoke" and source != "synthetic":
        raise ConfigError("Smoke mode must use the synthetic source")
    if mode == "full" and source != "kaggle":
        raise ConfigError("Full mode must use the Kaggle source")
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigError("'runtime.log_level' is not a standard logging level")
    baseline = sections["baseline"]
    if baseline["sanity_model"] != "zero":
        raise ConfigError("This experiment pre-registers zero as the only sanity model")
    if baseline["tree_model"] != "lightgbm":
        raise ConfigError("Exactly one tree model is allowed: lightgbm")
    if baseline["seasonal_average_weeks"] != 4:
        raise ConfigError("The seasonal-average variant is pre-registered at four weeks")

    horizon = _positive_int(sections["forecast"]["horizon"], "forecast.horizon")
    if horizon != PROJECT_HORIZON:
        raise ConfigError(f"This project requires a {PROJECT_HORIZON}-day horizon")
    if baseline["training_origin_stride_days"] != horizon:
        raise ConfigError("Training origins must use the same stride as the forecast horizon")

    origins_raw = sections["validation"]["origins"]
    if not isinstance(origins_raw, list) or not origins_raw:
        raise ConfigError("'validation.origins' must be a non-empty list")
    origins = tuple(_parse_date(value, "validation.origins") for value in origins_raw)
    if tuple(sorted(set(origins))) != origins:
        raise ConfigError("'validation.origins' must be unique and chronological")

    def resolve_project_path(value: Any) -> Path:
        candidate = Path(str(value)).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return (project_root / candidate).resolve()

    return ProjectConfig(
        mode=mode,
        paths=PathsConfig(
            raw_data_dir=resolve_project_path(sections["paths"]["raw_data_dir"]),
            artifacts_dir=resolve_project_path(sections["paths"]["artifacts_dir"]),
        ),
        data=DataConfig(
            source=source,
            expected_stores=_positive_int(
                sections["data"]["expected_stores"], "data.expected_stores"
            ),
            expected_families=_positive_int(
                sections["data"]["expected_families"], "data.expected_families"
            ),
            fixture_start_date=_parse_date(
                sections["data"]["fixture_start_date"], "data.fixture_start_date"
            ),
            fixture_history_days=_positive_int(
                sections["data"]["fixture_history_days"], "data.fixture_history_days"
            ),
        ),
        forecast=ForecastConfig(
            horizon=horizon,
            seasonal_period=_positive_int(
                sections["forecast"]["seasonal_period"], "forecast.seasonal_period"
            ),
        ),
        validation=ValidationConfig(origins=origins),
        baseline=BaselineConfig(
            sanity_model=str(baseline["sanity_model"]),
            seasonal_average_weeks=_positive_int(
                baseline["seasonal_average_weeks"], "baseline.seasonal_average_weeks"
            ),
            training_origin_stride_days=_positive_int(
                baseline["training_origin_stride_days"],
                "baseline.training_origin_stride_days",
            ),
            max_training_origins=_positive_int(
                baseline["max_training_origins"], "baseline.max_training_origins"
            ),
            neural_min_relative_improvement=_bounded_float(
                baseline["neural_min_relative_improvement"],
                "baseline.neural_min_relative_improvement",
                0.0,
                1.0,
            ),
            oof_git_limit_mb=_positive_int(
                baseline["oof_git_limit_mb"], "baseline.oof_git_limit_mb"
            ),
            tree_model=str(baseline["tree_model"]),
            tree_n_estimators=_positive_int(
                baseline["tree_n_estimators"], "baseline.tree_n_estimators"
            ),
            tree_learning_rate=_bounded_float(
                baseline["tree_learning_rate"], "baseline.tree_learning_rate", 0.0, 1.0
            ),
            tree_num_leaves=_positive_int(baseline["tree_num_leaves"], "baseline.tree_num_leaves"),
            tree_min_child_samples=_positive_int(
                baseline["tree_min_child_samples"], "baseline.tree_min_child_samples"
            ),
            tree_feature_fraction=_bounded_float(
                baseline["tree_feature_fraction"],
                "baseline.tree_feature_fraction",
                0.0,
                1.0,
            ),
            tree_l2=_nonnegative_float(baseline["tree_l2"], "baseline.tree_l2"),
        ),
        runtime=RuntimeConfig(
            seed=_positive_int(sections["runtime"]["seed"], "runtime.seed"),
            log_level=log_level,
        ),
        config_path=config_path,
        project_root=project_root,
    )
