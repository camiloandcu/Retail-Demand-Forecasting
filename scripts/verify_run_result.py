"""Verify the exact execution selected by a command's JSON result."""

from __future__ import annotations

import argparse
from pathlib import Path

from retail_forecast.artifacts import read_completed_run
from retail_forecast.io_utils import read_json


def verify_result(result_path: Path, *, mode: str, source: str) -> Path:
    """Reject incomplete/corrupt records before returning their declared submission."""

    result = read_json(result_path)
    root = Path(result["run_dir"])
    manifest = read_completed_run(root, expected_mode=mode, expected_source=source)
    if result["run_id"] != manifest["run_id"]:
        raise ValueError("RUN_RESULT_IDENTITY_MISMATCH")
    outputs = [entry for entry in manifest["artifacts"] if entry["kind"] == "submission"]
    if len(outputs) != 1 or outputs[0]["bytes"] <= 0:
        raise ValueError("SUBMISSION_MISSING")
    path = root / outputs[0]["path"]
    if Path(result["submission"]) != path:
        raise ValueError("RUN_RESULT_PATH_MISMATCH")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--source", choices=("synthetic", "kaggle"), required=True)
    args = parser.parse_args()
    verify_result(args.result, mode=args.mode, source=args.source)


if __name__ == "__main__":
    main()
