"""Execute numbered notebooks in fresh kernels, preserving partial and full evidence.

Run in Colab after preparing the locked environment. This script does not assert
Colab success when executed locally; its receipt records the actual environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--repo-url", default="https://github.com/camiloandcu/Retail-Demand-Forecasting.git"
    )
    parser.add_argument("--mode", choices=["full", "smoke"], default="full")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = (
        [root / n for n in args.notebooks]
        if args.notebooks
        else sorted(root.glob("0[1-5]_*.ipynb"))
    )
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    receipt = {
        "notebook_revision": revision,
        "mode": args.mode,
        "python": args.python,
        "driver_python": sys.version,
        "platform": platform.platform(),
        "colab_release_tag": os.getenv("COLAB_RELEASE_TAG"),
        "notebooks": [],
    }
    with tempfile.TemporaryDirectory(prefix="academic-kernel-") as temporary:
        kernels = Path(temporary)
        spec = kernels / "academic"
        spec.mkdir()
        (spec / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [args.python, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "Academic verification",
                    "language": "python",
                    "env": {
                        "RETAIL_FORECAST_MODE": args.mode,
                        "RETAIL_FORECAST_REPO_URL": args.repo_url,
                        "RETAIL_FORECAST_NOTEBOOK_REVISION": revision,
                    },
                }
            )
        )
        for path in paths:
            notebook = nbformat.read(path, as_version=4)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    cell.outputs, cell.execution_count = [], None
            manager = KernelManager(
                kernel_name="academic",
                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]),
            )
            client = NotebookClient(
                notebook, km=manager, timeout=7200, resources={"metadata": {"path": str(root)}}
            )
            started = time.monotonic()
            status, error = "completed", None
            try:
                client.execute(cleanup_kc=True)
            except Exception as failure:
                status, error = "incomplete", str(failure)[-1500:]
            finally:
                saved = output / path.name
                nbformat.write(notebook, saved)
                errors = sum(
                    o.get("output_type") == "error"
                    for c in notebook.cells
                    if c.cell_type == "code"
                    for o in c.outputs
                )
                record = {
                    "notebook": path.name,
                    "input_sha256": sha(path),
                    "output_sha256": sha(saved),
                    "status": status,
                    "cell_errors": errors,
                    "error": error,
                    "executed_code_cells": sum(
                        c.execution_count is not None
                        for c in notebook.cells
                        if c.cell_type == "code"
                    ),
                    "wall_seconds": time.monotonic() - started,
                }
                receipt["notebooks"].append(record)
                (output / "execution.json").write_text(json.dumps(receipt, indent=2))
                print(json.dumps(record), flush=True)
            if status != "completed" or errors:
                raise SystemExit("Notebook execution incomplete; inspect saved outputs")


if __name__ == "__main__":
    main()
