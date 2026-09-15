"""Command-line interface for reproducible project workflows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from retail_forecast.config import ProjectConfig, load_config
from retail_forecast.data import validate_dataset, write_synthetic_dataset
from retail_forecast.logging_utils import configure_logging
from retail_forecast.pipeline import baseline_experiment, evaluate, predict, smoke, train
from retail_forecast.versions import capture_versions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retail-forecast")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ["fixture", "validate", "train", "evaluate", "predict", "smoke", "baseline"]:
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    subparsers.add_parser("versions")
    return parser


def _run_with_config(function: Callable[[ProjectConfig], Any], config_path: Path) -> Any:
    config = load_config(config_path)
    configure_logging(config.runtime.log_level)
    return function(config)


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one pipeline command and return a process exit code."""

    args = _parser().parse_args(argv)
    if args.command == "versions":
        result: Any = capture_versions()
    else:
        functions: dict[str, Callable[[ProjectConfig], Any]] = {
            "fixture": write_synthetic_dataset,
            "validate": lambda config: validate_dataset(config).to_dict(),
            "train": train,
            "evaluate": evaluate,
            "predict": predict,
            "smoke": smoke,
            "baseline": baseline_experiment,
        }
        result = _run_with_config(functions[args.command], args.config)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
