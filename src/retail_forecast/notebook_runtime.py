"""Immutable notebook setup using only the standard library until installation."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import re
import shutil
import site
import subprocess
import sys
from pathlib import Path

SUPPORTED_PYTHON = ((3, 10), (3, 11), (3, 12), (3, 13))


def check_python(version: tuple[int, int] | None = None) -> None:
    """Reject unsupported interpreters before installation or data acquisition."""
    if (version or sys.version_info[:2]) not in SUPPORTED_PYTHON:
        raise RuntimeError("Use Python 3.10–3.13; this runtime is unsupported. Restart with it.")


def git(root: Path, *args: str) -> str:
    """Read Git identity without changing contributor branches."""
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def validate_checkout(root: Path, revision: str, origin: str) -> Path:
    """Refuse incomplete, mismatched or dirty repositories without mutation."""
    check_python()
    root = root.resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("CODE_REVISION must be a reviewed 40-character commit SHA.")
    try:
        if git(root, "rev-parse", "--show-toplevel") != str(root):
            raise RuntimeError("Select the checkout root, not a nested directory.")
        if git(root, "remote", "get-url", "origin") != origin:
            raise RuntimeError("Checkout origin mismatch; use a new directory.")
        if git(root, "rev-parse", "HEAD") != revision:
            raise RuntimeError("Checkout revision mismatch; use the pinned checkout separately.")
        if git(root, "status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError("Checkout has user changes; preserve it and use a new directory.")
    except subprocess.CalledProcessError:
        raise RuntimeError("Incomplete checkout; preserve it and use a new directory.") from None
    for filename in ("pyproject.toml", "uv.lock", "src/retail_forecast/notebook_runtime.py"):
        if not (root / filename).is_file():
            raise RuntimeError("Incomplete checkout; use a new directory.")
    return root


def verify_loaded_source(root: Path) -> None:
    """An already imported foreign package cannot be repaired by changing paths."""
    for name, module in tuple(sys.modules.items()):
        if name == "retail_forecast" or name.startswith("retail_forecast."):
            source = getattr(module, "__file__", None)
            if source and not Path(source).resolve().is_relative_to(root / "src"):
                raise RuntimeError(
                    "Package already imported from another checkout. Restart kernel."
                )


def install(root: Path, *, models: bool = False) -> None:
    """Install hashed dependencies into this interpreter, then the pinned project."""
    profile = root / "constraints" / ("02-notebook-models.txt" if models else "01-notebook.txt")
    lock_hash = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
    if f"# uv.lock SHA256: {lock_hash}" not in profile.read_text():
        raise RuntimeError("Constraints do not match uv.lock; regenerate before publishing.")
    if importlib.util.find_spec("pip") is not None:
        command = [sys.executable, "-m", "pip", "install"]
    elif shutil.which("uv"):
        command = ["uv", "pip", "install", "--python", sys.executable]
    else:
        raise RuntimeError("Install pip or uv in this Python runtime, then retry.")
    # Output is bounded and never includes environment variables or credentials.
    for arguments in (
        ["--require-hashes", "-r", str(profile)],
        ["--no-deps", "--no-build-isolation", "-e", str(root)],
    ):
        result = subprocess.run(command + arguments, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(
                "Locked installation failed. Check connectivity/runtime; retry setup."
            )
    # Process newly installed editable .pth files in the already running kernel.
    for directory in site.getsitepackages():
        site.addsitedir(directory)
    importlib.invalidate_caches()


def bootstrap(root: Path, revision: str, origin: str, *, models: bool = False) -> dict:
    """Validate, install and report identities; refuse mixed-source kernel state."""
    root = validate_checkout(root, revision, origin)
    verify_loaded_source(root)
    # Dependencies imported before an upgrade remain resident. Detect disagreement
    # after installation and require restart instead of running mixed versions.
    loaded = {}
    distribution_map = importlib.metadata.packages_distributions()
    for name, module in tuple(sys.modules.items()):
        if "." in name:
            continue
        for distribution in distribution_map.get(name, []):
            version = importlib.metadata.version(distribution)
            resident = getattr(module, "__version__", None)
            # Numeric module versions can retain zero padding removed from wheel
            # metadata (certifi 2026.07.22 == 2026.7.22). Compare release integers;
            # non-release module strings have no general distribution contract.
            if (
                isinstance(resident, str)
                and re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", resident)
                and re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,3}", version)
            ):
                resident_parts = [int(part) for part in resident.split(".")]
                version_parts = [int(part) for part in version.split(".")]
                while resident_parts and resident_parts[-1] == 0:
                    resident_parts.pop()
                while version_parts and version_parts[-1] == 0:
                    version_parts.pop()
                if resident_parts != version_parts:
                    raise RuntimeError(f"{name} is already a different version. Restart kernel.")
            loaded[distribution] = version
    install(root, models=models)
    for distribution, old in loaded.items():
        if old != importlib.metadata.version(distribution):
            raise RuntimeError(f"{distribution} changed during install. Restart kernel.")
    package = importlib.import_module("retail_forecast")
    source = Path(package.__file__).resolve()
    if not source.is_relative_to(root / "src"):
        raise RuntimeError("Installed package resolves another checkout. Restart kernel.")
    distribution = importlib.metadata.distribution("retail-forecast")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct.get("url") != root.as_uri():
        raise RuntimeError("Installed distribution identity differs from checkout. Restart kernel.")
    report = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "code_revision": git(root, "rev-parse", "HEAD"),
        "package_source": str(source),
        "lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
        "dependency_profile": "models" if models else "notebook",
        "installer": (
            f"pip {importlib.metadata.version('pip')}"
            if importlib.util.find_spec("pip") is not None
            else subprocess.run(
                ["uv", "--version"], check=True, capture_output=True, text=True
            ).stdout.strip()
        ),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("retail-forecast", "numpy", "pandas", "ipykernel", "setuptools")
        },
    }
    print(json.dumps(report, indent=2))
    return report
