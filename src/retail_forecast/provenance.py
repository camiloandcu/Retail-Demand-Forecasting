"""Portable code, configuration and input identities without raw diagnostic data."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from retail_forecast.config import ProjectConfig
from retail_forecast.data import EXPECTED_KAGGLE_SHA256, FILE_COLUMNS, validate_dataset
from retail_forecast.io_utils import canonical_json

_OBSERVED_HASH_SEED = os.environ.get("PYTHONHASHSEED")
OBSERVED_HASH_SEED = (
    _OBSERVED_HASH_SEED
    if _OBSERVED_HASH_SEED == "random" or (_OBSERVED_HASH_SEED and _OBSERVED_HASH_SEED.isdecimal())
    else None
)


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_identity() -> dict[str, Any]:
    """Identify the imported package checkout, never the caller's working directory."""
    source = Path(__file__).resolve()
    try:
        root = subprocess.check_output(
            ["git", "-C", str(source.parent), "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        checkout_package = Path(root) / "src/retail_forecast"
        if source.parent != checkout_package:
            raise ValueError("Unidentified package source")
        subprocess.check_output(
            [
                "git",
                "-C",
                root,
                "ls-files",
                "--error-unmatch",
                "--",
                "src/retail_forecast/__init__.py",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        sha = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", root, "status", "--porcelain", "--untracked-files=normal"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return {"git_sha": sha, "dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError, ValueError):
        return {"git_sha": None, "dirty": True}


def resolved_config(config: ProjectConfig) -> dict[str, Any]:
    """Keep logical path identities while removing installation-specific roots."""
    result = asdict(config)
    result.pop("project_root")
    result.pop("config_path")
    paths = {}
    for key, value in result["paths"].items():
        path = Path(value)
        try:
            paths[key] = path.relative_to(config.project_root).as_posix()
        except ValueError:
            paths[key] = f"external:{key}"
    result["paths"] = paths
    # Return JSON-native values, including dates normalized by the shared serializer.
    import json

    return json.loads(canonical_json(result))


def hardware_context() -> dict[str, Any]:
    memory = None
    try:
        memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        pass
    return {
        "cpu_architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": memory,
        "memory_method": "sysconf" if memory else "unavailable",
        "gpu": None,
        "gpu_method": "not_probed",
        "pythonhashseed_observed_at_import": OBSERVED_HASH_SEED,
        "pythonhashseed_method": "environment_at_provenance_import_not_verified_launch_seed",
        "interpreter_hash_randomization": bool(sys.flags.hash_randomization),
    }


def dataset_snapshot(config: ProjectConfig, input_dir: Path) -> dict[str, Any]:
    """Capture validated raw-stage inputs; this does not certify as-of semantics."""
    from dataclasses import replace

    checked = replace(config, paths=replace(config.paths, raw_data_dir=input_dir))
    for ancestor in [*reversed(input_dir.absolute().parents), input_dir.absolute()]:
        if ancestor.is_symlink() or (ancestor.exists() and not ancestor.is_dir()):
            raise ValueError("Unsafe dataset root")
    for name in FILE_COLUMNS:
        path = input_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Unsafe dataset input")
    validate_dataset(checked)
    files = []
    for name in FILE_COLUMNS:
        path = input_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Unsafe dataset input")
        files.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    if config.data.source == "kaggle" and any(
        entry["sha256"] != EXPECTED_KAGGLE_SHA256[entry["path"]] for entry in files
    ):
        raise ValueError("Kaggle source integrity mismatch")
    train = pd.read_csv(input_dir / "train.csv", usecols=["date", "store_nbr", "family"])
    series = train[["store_nbr", "family"]].drop_duplicates().sort_values(["store_nbr", "family"])
    snapshot = {
        "schema_version": 1,
        "data_source": config.data.source,
        "execution_mode": config.mode,
        "input_contract_version": 1,
        "availability_policy": "competition_v1",
        "stage": "raw_benchmark",
        "files": files,
        "valid_date_range": {"start": str(train.date.min()), "end": str(train.date.max())},
        "supported_series": [
            {"store_nbr": int(row.store_nbr), "family": str(row.family)}
            for row in series.itertuples(index=False)
        ],
        "extraction_code": code_identity(),
        "generator": {"seed": config.runtime.seed, "config": asdict(config.data)}
        if config.data.source == "synthetic"
        else None,
    }
    snapshot["snapshot_id"] = digest_json(snapshot)
    return snapshot


def verify_inputs(snapshot: dict[str, Any], input_dir: Path) -> None:
    for entry in snapshot["files"]:
        path = input_dir / entry["path"]
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or sha256_file(path) != entry["sha256"]
        ):
            raise ValueError("Input snapshot changed")
