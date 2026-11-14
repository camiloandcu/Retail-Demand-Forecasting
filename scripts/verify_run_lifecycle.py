"""Bounded synthetic process-interruption proof for local or Colab-local storage."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from retail_forecast import artifacts
from retail_forecast.config import load_config
from retail_forecast.pipeline import smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--interrupt-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.interrupt_child:
        if args.root is None:
            parser.error("Internal child requires a root")
        root = args.root
    elif args.root is None:
        root = Path(tempfile.mkdtemp(prefix="retail-run-lifecycle-"))
    else:
        root = args.root.absolute()
        root.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config.absolute())
    config = replace(config, paths=replace(config.paths, artifacts_dir=root / "artifacts/smoke"))
    if args.interrupt_child:
        original = artifacts.write_json

        def interrupt_before_completion(path: Path, payload: object) -> None:
            if path.name == "COMPLETE.json":
                os._exit(23)
            original(path, payload)

        artifacts.write_json = interrupt_before_completion
        smoke(config, run_id="interrupted")
        raise AssertionError("Interruption hook did not execute")
    result = smoke(config, run_id="successful")
    verified = artifacts.read_completed_run(
        Path(result["run_dir"]), expected_mode="smoke", expected_source="synthetic"
    )
    child = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).absolute()),
            "--config",
            str(args.config.absolute()),
            "--root",
            str(root),
            "--interrupt-child",
        ],
        check=False,
        capture_output=True,
        timeout=60,
    )
    if child.returncode != 23:
        raise RuntimeError("Synthetic interruption child failed before the publication boundary")
    interrupted = root / "artifacts/smoke/interrupted"
    if not (interrupted / "run.json").is_file() or (interrupted / "COMPLETE.json").exists():
        raise AssertionError("Incorrect interrupted publication state")
    try:
        artifacts.read_completed_run(interrupted)
    except (ValueError, FileNotFoundError):
        pass
    else:
        raise AssertionError("Interrupted result became consumable")
    print(
        json.dumps(
            {
                "status": "PASS",
                "evidence": "synthetic process interruption; not power-loss durability",
                "platform": platform.system(),
                "python": platform.python_version(),
                "successful_run": result["run_id"],
                "root": str(root),
                "code": verified["code"],
                "interrupted_rejected": True,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
