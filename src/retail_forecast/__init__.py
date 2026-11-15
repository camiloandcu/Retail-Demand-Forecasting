"""Reusable components for the retail forecasting project."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("retail-forecast")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

__all__ = ["__version__"]
