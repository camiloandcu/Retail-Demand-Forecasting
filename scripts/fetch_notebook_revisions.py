"""Fetch each notebook's pinned package revision by SHA when it is not available locally.

A squash merge keeps the reviewed pinned commit only under its pull-request ref, which
neither a normal clone nor CI checkout fetches. Fetching the exact SHA is the same
operation the Colab bootstrap performs, so a failure here also means a fresh Colab
runtime cannot obtain the pin.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

PIN = re.compile(r'^CODE_REVISION = "([0-9a-f]{40})"$', re.MULTILINE)


def pinned_revisions(root: Path) -> list[str]:
    """Return the sorted distinct `CODE_REVISION` values of notebooks 0*.ipynb."""
    revisions = set()
    for path in sorted(root.glob("0*.ipynb")):
        cells = json.loads(path.read_text())["cells"]
        found = {
            match
            for cell in cells
            if cell["cell_type"] == "code"
            for match in PIN.findall("".join(cell["source"]))
        }
        if len(found) != 1:
            raise SystemExit(f"{path.name}: expected exactly one CODE_REVISION, found {len(found)}")
        revisions |= found
    if not revisions:
        raise SystemExit(f"No pinned notebooks found under {root}")
    return sorted(revisions)


def is_local_commit(root: Path, revision: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", f"{revision}^{{commit}}"],
        capture_output=True,
    )
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--remote", default="origin")
    arguments = parser.parse_args()
    for revision in pinned_revisions(arguments.root):
        if is_local_commit(arguments.root, revision):
            print(f"{revision} already available")
            continue
        subprocess.run(
            ["git", "-C", str(arguments.root), "fetch", "--no-tags", arguments.remote, revision],
            check=True,
        )
        if not is_local_commit(arguments.root, revision):
            raise SystemExit(f"{revision} was not obtained from {arguments.remote}")
        print(f"{revision} fetched from {arguments.remote}")


if __name__ == "__main__":
    main()
