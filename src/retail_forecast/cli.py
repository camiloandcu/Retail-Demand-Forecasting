"""Command-line interface for explicit reproducible execution records."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from retail_forecast.artifacts import record_interruption, resolve_run
from retail_forecast.bundles import parse_bundle_ref
from retail_forecast.capsules import export_inference_capsule
from retail_forecast.config import load_config
from retail_forecast.data import validate_dataset, write_synthetic_dataset
from retail_forecast.logging_utils import configure_logging
from retail_forecast.models import SERVING_MODEL_IDS
from retail_forecast.pipeline import (
    baseline_experiment,
    compare,
    evaluate,
    predict,
    request,
    smoke,
    train,
)
from retail_forecast.versions import capture_versions

PRODUCERS = {
    "train",
    "evaluate",
    "predict",
    "smoke",
    "baseline",
    "request",
    "compare",
    "recurrent",
    "recurrent-feasibility",
}
COMMANDS = [
    "fixture",
    "validate",
    "export-capsule",
    "record-interruption",
    *sorted(PRODUCERS),
]


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use an ISO date, for example 2017-08-15") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retail-forecast")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        if name in PRODUCERS:
            command.add_argument(
                "--run-id", help="Allocate this new execution ID; never reuse a run"
            )
            command.add_argument(
                "--resume-from",
                help="Explicit epoch checkpoint source; GRU training supports continuation",
            )
        if name == "record-interruption":
            command.add_argument("--run", required=True, help="Abandoned RUNNING run ID")
            command.add_argument(
                "--min-idle-minutes",
                type=float,
                default=60.0,
                help="Refuse records changed more recently than this",
            )
        if name == "export-capsule":
            command.add_argument(
                "--bundle", required=True, help="<training_run_id>:models/<model_id>/<slot>"
            )
            command.add_argument(
                "--destination", type=Path, required=True, help="Capsule destination root"
            )
        if name == "train":
            command.add_argument("--model-id", required=True, choices=SERVING_MODEL_IDS)
            command.add_argument(
                "--origin", type=_date, help="Training cutoff; default: last training date"
            )
        if name == "request":
            command.add_argument(
                "--origin", type=_date, help="Forecast origin; default: last training date"
            )
        if name == "predict":
            command.add_argument(
                "--bundle",
                required=True,
                help="<training_run_id>:models/<model_id>/<slot> or capsule:<path>",
            )
            source = command.add_mutually_exclusive_group()
            source.add_argument("--origin", type=_date, help="Project raw inputs at this origin")
            source.add_argument(
                "--request", type=Path, help="Normalized request directory (no raw files)"
            )
            command.add_argument(
                "--kaggle-submission",
                action="store_true",
                help="Also export a keyed Kaggle submission from raw test.csv",
            )
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
        elif args.command == "record-interruption":
            manifest = record_interruption(
                resolve_run(config, args.run), min_idle_seconds=args.min_idle_minutes * 60
            )
            result = {"run_id": manifest["run_id"], "state": manifest["state"]}
        elif args.command == "export-capsule":
            ref = parse_bundle_ref(args.bundle, artifacts_dir=config.paths.artifacts_dir)
            if ref.kind != "run":
                raise ValueError("Export a capsule from a completed training run bundle")
            capsule = export_inference_capsule(
                ref.root, bundle_relative_path=ref.relative, destination_root=args.destination
            )
            result = {
                "capsule_id": capsule["capsule_id"],
                "bundle_id": capsule["bundle_id"],
                "capsule_dir": str(args.destination.absolute() / capsule["capsule_id"]),
            }
        else:
            from retail_forecast.recurrent import recurrent_experiment
            from retail_forecast.recurrent_feasibility import recurrent_feasibility

            functions = {
                "recurrent": recurrent_experiment,
                "recurrent-feasibility": recurrent_feasibility,
                "train": train,
                "evaluate": evaluate,
                "predict": predict,
                "smoke": smoke,
                "baseline": baseline_experiment,
                "request": request,
                "compare": compare,
            }
            options: dict[str, Any] = {"run_id": args.run_id, "resume_from": args.resume_from}
            if args.command == "train":
                options.update(model_id=args.model_id, origin=args.origin)
            elif args.command == "request":
                options["origin"] = args.origin
            elif args.command == "predict":
                options.update(
                    bundle=args.bundle,
                    origin=args.origin,
                    request_dir=args.request,
                    kaggle_submission=args.kaggle_submission,
                )
            elif args.command == "baseline":
                options["tracking"] = not args.no_tracking
            result = functions[args.command](config, **options)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
