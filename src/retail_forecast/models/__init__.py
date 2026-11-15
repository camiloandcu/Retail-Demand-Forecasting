"""Registered model adapters; optional frameworks load only when an adapter is used."""

from __future__ import annotations

from importlib import import_module
from typing import Final

from retail_forecast.models.base import Adapter

# adapter_id -> (module, class, supported major versions)
ADAPTERS: Final[dict[str, tuple[str, str, frozenset[int]]]] = {
    "seasonal_naive_7d": (
        "retail_forecast.models.seasonal",
        "SeasonalNaiveAdapter",
        frozenset({1}),
    ),
    "seasonal_average_4w": (
        "retail_forecast.models.seasonal",
        "SeasonalAverageAdapter",
        frozenset({1}),
    ),
    "lightgbm_global": ("retail_forecast.models.lightgbm", "LightGBMAdapter", frozenset({1})),
}
COMPARATOR_MODEL_IDS: Final = tuple(ADAPTERS)
ADAPTERS["gru_global"] = ("retail_forecast.models.gru", "GRUAdapter", frozenset({1}))
SERVING_MODEL_IDS: Final = tuple(ADAPTERS)


def get_adapter(adapter_id: str) -> Adapter:
    """Return a registered adapter instance or fail before any native work."""

    if adapter_id not in ADAPTERS:
        raise ValueError(f"Unknown model adapter: {adapter_id!r}")
    module, name, _ = ADAPTERS[adapter_id]
    return getattr(import_module(module), name)()


def supports_version(adapter_id: str, version: object) -> bool:
    """True when a saved adapter version's major is in the loader allowlist."""

    if adapter_id not in ADAPTERS or not isinstance(version, str):
        return False
    major, _, minor = version.partition(".")
    if not major.isdecimal() or not minor.isdecimal():
        return False
    return int(major) in ADAPTERS[adapter_id][2]
