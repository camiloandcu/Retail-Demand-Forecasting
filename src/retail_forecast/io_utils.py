"""Small serialization helpers for pipeline artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    """Write a stable, human-readable JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def read_json(path: Path) -> Any:
    """Read a JSON artifact or fail with its concrete path."""

    if not path.is_file():
        raise FileNotFoundError(f"Artifact does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
