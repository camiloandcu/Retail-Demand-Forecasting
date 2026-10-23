"""Run pinned notebook cells twice in a disposable checkout and a real kernel."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from jupyter_client import KernelManager


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python", required=True, help="Disposable interpreter with ipykernel installed"
    )
    args = parser.parse_args()
    subprocess.run(
        [
            args.python,
            "-c",
            "import importlib.util; "
            "assert importlib.util.find_spec('retail_forecast') is None, "
            "'Use a fresh dependency-only environment without retail-forecast installed'",
        ],
        check=True,
    )
    source = Path(__file__).resolve().parents[1]
    notebooks = []
    for path in sorted(source.glob("0*.ipynb")):
        notebook = json.loads(path.read_text())
        cell = next(
            "".join(c["source"])
            for c in notebook["cells"]
            if c["cell_type"] == "code" and "CODE_REVISION =" in "".join(c["source"])
        )
        revision = re.search(r'CODE_REVISION = "([a-f0-9]{40})"', cell)[1]
        origin = re.search(r'REPO_URL = "([^"]+)"', cell)[1]
        notebooks.append((path.name, cell, revision, origin))
    if len({(n[2], n[3]) for n in notebooks}) != 1:
        raise RuntimeError("All existing notebooks must select the same package identity")
    _, _, revision, origin = notebooks[0]
    with tempfile.TemporaryDirectory(prefix="retail-bootstrap-") as temporary:
        root = Path(temporary) / "checkout"
        for command in (
            ["git", "clone", "-q", str(source), str(root)],
            ["git", "-C", str(root), "checkout", "--detach", revision],
            ["git", "-C", str(root), "remote", "set-url", "origin", origin],
        ):
            subprocess.run(command, check=True, capture_output=True, text=True)
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [
            str(Path(args.python).absolute()),
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        manager.start_kernel(
            cwd=str(root),
            env=dict(os.environ, RETAIL_FORECAST_CHECKOUT=str(root), RETAIL_FORECAST_MODE="smoke"),
        )
        client = manager.client()
        client.start_channels()
        try:
            client.wait_for_ready(timeout=30)
            for name, code, _, _ in notebooks:
                for _ in range(2):
                    client.execute(code)
                    while True:
                        message = client.get_iopub_msg(timeout=240)
                        kind, content = message["header"]["msg_type"], message["content"]
                        if kind == "error":
                            raise RuntimeError("\n".join(content["traceback"]))
                        if kind == "status" and content["execution_state"] == "idle":
                            break
                print(f"PASS {name}: initial setup and same-kernel rerun at {revision}")
        finally:
            client.stop_channels()
            manager.shutdown_kernel(now=True)
    # Installation was editable: do not leave an environment pointing to deleted code.
    print("Disposable interpreter now references the removed fixture; recreate it before reuse.")


if __name__ == "__main__":
    main()
