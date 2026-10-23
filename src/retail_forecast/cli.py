"""Command-line interface for explicit reproducible execution records."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from retail_forecast.config import load_config
from retail_forecast.data import validate_dataset, write_synthetic_dataset
from retail_forecast.logging_utils import configure_logging
from retail_forecast.pipeline import baseline_experiment, evaluate, predict, smoke, train
from retail_forecast.versions import capture_versions

PRODUCERS = {"train", "evaluate", "predict", "smoke", "baseline"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retail-forecast")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ["fixture", "validate", "train", "evaluate", "predict", "smoke", "baseline"]:
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name in PRODUCERS:
            command.add_argument(
                "--run-id", help="Allocate this new execution ID; never reuse a run"
            )
            command.add_argument(
                "--resume-from",
                help="Explicit checkpoint source; current adapters reject continuation",
            )
        if name == "predict":
            command.add_argument("--training-run", required=True, help="Completed training run ID")
        if name == "baseline":
            command.add_argument("--no-tracking", action="store_true", help="Disable local MLflow")
    subparsers.add_parser("versions")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one command and emit machine-readable explicit output paths."""

    args = _parser().parse_args(argv)
    if args.command == "versions":
        result: Any = capture_versions()
    else:
        config = load_config(args.config)
        configure_logging(config.runtime.log_level)
        if args.command == "fixture":
            result = {"fixture_dir": str(write_synthetic_dataset(config))}
        elif args.command == "validate":
            result = validate_dataset(config).to_dict()
        else:
            functions = {
                "train": train,
                "evaluate": evaluate,
                "predict": predict,
                "smoke": smoke,
                "baseline": baseline_experiment,
            }
            options: dict[str, Any] = {"run_id": args.run_id, "resume_from": args.resume_from}
            if args.command == "predict":
                options["training_run"] = args.training_run
            elif args.command == "baseline":
                options["tracking"] = not args.no_tracking
            result = functions[args.command](config, **options)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
