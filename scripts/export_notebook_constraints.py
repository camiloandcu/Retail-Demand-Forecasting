"""Regenerate hashed notebook dependency profiles from the reviewed uv lock."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

UV_VERSION = "0.12.0"
PROFILES = {
    "01-notebook.txt": ("dev", "notebook"),
    "02-notebook-models.txt": ("dev", "notebook", "models"),
}


def export_profile(root: Path, extras: tuple[str, ...]) -> str:
    """Export the universal lock with runtime markers and artifact hashes intact."""
    command = ["uv", "export", "--frozen", "--no-emit-project", "--no-header", "--no-annotate"]
    for extra in extras:
        command.extend(("--extra", extra))
    result = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    lock_hash = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    return (
        "# Generated; do not edit. Regenerate: python scripts/export_notebook_constraints.py\n"
        f"# Generator: uv {UV_VERSION}; supported Python: >=3.10,<3.14\n"
        f"# uv.lock SHA256: {lock_hash}\n"
        f"# Export command: {' '.join(command)}\n"
        "# Markers select the locked dependency fork for the active interpreter.\n"
        "# Install dependencies with --require-hashes, then project with --no-deps "
        "--no-build-isolation.\n" + result.stdout
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if generated files differ")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version = subprocess.run(
        ["uv", "--version"], check=True, capture_output=True, text=True
    ).stdout.split()[1]
    if version != UV_VERSION:
        raise SystemExit(f"Regeneration requires uv {UV_VERSION}; found {version}")
    for filename, extras in PROFILES.items():
        path = root / "constraints" / filename
        expected = export_profile(root, extras)
        if args.check:
            if not path.is_file() or path.read_text() != expected:
                raise SystemExit(f"Stale generated constraints: {path.relative_to(root)}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected)
        print(f"{'Verified' if args.check else 'Generated'} {path.relative_to(root)}")


if __name__ == "__main__":
    main()
