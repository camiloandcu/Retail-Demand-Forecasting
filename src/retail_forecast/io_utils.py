"""Typed, canonical JSON serialization shared by artifact producers."""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_value(item) for key, item in value.items()}
    raise TypeError("Unsupported JSON value type")


def canonical_json(payload: Any) -> bytes:
    """Serialize supported values deterministically, rejecting nonfinite numbers."""
    return (
        json.dumps(
            _json_value(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        + b"\n"
    )


def write_json(path: Path, payload: Any) -> None:
    """Atomically write canonical JSON within the destination filesystem."""
    data = canonical_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".json-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError("Nonfinite JSON value")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite JSON number")
    return result


def read_json(path: Path) -> Any:
    """Read unambiguous finite JSON; path trust belongs to the caller."""
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_pairs,
        parse_constant=_invalid_constant,
        parse_float=_finite_float,
    )
