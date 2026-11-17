"""Resolve training hardware and optional elapsed-time limits."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def resolve_device(requested: str) -> str:
    """Use CUDA when available for auto; preserve explicit device requests."""
    if requested == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("Training device must be auto, cpu or cuda")
    return requested


def training_time_limit(params: Mapping[str, Any]) -> float | None:
    """Return seconds, or None when training has no elapsed-time cutoff."""
    minutes = params["training_minutes"]
    return None if minutes is None else minutes * 60
