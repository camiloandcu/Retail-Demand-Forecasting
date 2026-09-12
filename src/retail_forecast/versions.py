"""Environment metadata for reproducible artifacts."""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any


def capture_versions() -> dict[str, Any]:
    """Return runtime and dependency versions without importing optional models."""

    packages = ["numpy", "pandas", "PyYAML", "retail-forecast", "tensorflow", "mlflow"]
    installed: dict[str, str | None] = {}
    for package in packages:
        try:
            installed[package] = version(package)
        except PackageNotFoundError:
            installed[package] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": installed,
    }
