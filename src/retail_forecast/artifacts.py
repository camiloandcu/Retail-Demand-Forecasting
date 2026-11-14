"""Exclusive local execution records with contained, integrity-checked outputs."""

from __future__ import annotations

import os
import secrets
import stat
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from retail_forecast.artifact_schema import REQUIRED_KINDS, validate_manifest, validate_run_id
from retail_forecast.config import ProjectConfig
from retail_forecast.io_utils import read_json, write_json
from retail_forecast.provenance import (
    code_identity,
    dataset_snapshot,
    digest_json,
    hardware_context,
    resolved_config,
    sha256_file,
    verify_inputs,
)
from retail_forecast.versions import capture_versions


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trusted_path(path: Path) -> Path:
    path = path.absolute()
    for part in [*reversed(path.parents), path]:
        if part.is_symlink():
            raise ValueError("Symlink path is not supported")
        if part.exists() and part != path and not part.is_dir():
            raise ValueError("Invalid path ancestor")
    return path


def safe_path(root: Path, relative: str | Path, *, regular: bool = False) -> Path:
    """Resolve a relative path without trusting symlinks or special files."""
    root = _trusted_path(root)
    raw = str(relative)
    posix = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or posix.is_absolute()
        or ".." in posix.parts
        or raw != posix.as_posix()
        or posix == PurePosixPath(".")
    ):
        raise ValueError("Unsafe artifact path")
    target = _trusted_path(root.joinpath(*posix.parts))
    if not target.is_relative_to(root):
        raise ValueError("Artifact escaped run root")
    if target.exists() and not (target.is_dir() or stat.S_ISREG(target.stat().st_mode)):
        raise ValueError("Unsupported artifact type")
    if regular and (not target.exists() or not stat.S_ISREG(target.stat().st_mode)):
        raise ValueError("Artifact is not a regular file")
    return target


def resolve_run(config: ProjectConfig, run_id: str) -> Path:
    return safe_path(config.paths.artifacts_dir, validate_run_id(run_id))


@dataclass(frozen=True)
class ArtifactRecord:
    path: str
    bytes: int
    sha256: str
    kind: str


@dataclass
class RunContext:
    root: Path
    run_id: str
    config: ProjectConfig
    manifest: dict[str, Any]
    input_dir: Path | None = None
    snapshot: dict[str, Any] | None = None
    _terminal: bool = field(default=False, repr=False)

    def path(self, relative: str | Path) -> Path:
        ensure_running(self)
        return safe_path(self.root, relative)


def ensure_running(run: RunContext) -> None:
    _trusted_path(run.root)
    current = read_json(safe_path(run.root, "run.json", regular=True))
    validate_manifest(current)
    if (
        run._terminal
        or current["state"] != "RUNNING"
        or current["run_id"] != run.run_id
        or current != run.manifest
    ):
        raise ValueError("Run is immutable or context is stale; allocate a new run")


def _persist(run: RunContext) -> None:
    validate_manifest(run.manifest)
    write_json(safe_path(run.root, "run.json"), run.manifest)


def allocate_run(
    config: ProjectConfig,
    *,
    kind: str,
    run_id: str | None = None,
    parent_run_ids: tuple[str, ...] = (),
    resumed_from_run_id: str | None = None,
) -> RunContext:
    if kind not in REQUIRED_KINDS:
        raise ValueError("Unknown producer kind")
    run_id = validate_run_id(
        run_id
        or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + secrets.token_hex(4))
    )
    for parent in parent_run_ids:
        validate_run_id(parent)
    if resumed_from_run_id is not None:
        validate_run_id(resumed_from_run_id)
    root = resolve_run(config, run_id)
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    normalized = resolved_config(config)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "kind": kind,
        "execution_mode": config.mode,
        "data_source": config.data.source,
        "state": "RUNNING",
        "code": code_identity(),
        "config_sha256": digest_json(normalized),
        "input_contract_version": 1,
        "availability_policy": "competition_v1",
        "snapshot_id": None,
        "seed": config.runtime.seed,
        "started_at_utc": _now(),
        "finished_at_utc": None,
        "parent_run_ids": list(parent_run_ids),
        "resumed_from_run_id": resumed_from_run_id,
        "versions": capture_versions(),
        "hardware": hardware_context(),
        "artifacts": [],
        "error_code": None,
    }
    run = RunContext(root, run_id, config, manifest)
    _persist(run)
    write_json_artifact(run, "config.resolved.json", normalized, kind="resolved_config")
    return run


def register_artifact(run: RunContext, relative: str | Path, *, kind: str) -> ArtifactRecord:
    ensure_running(run)
    path = safe_path(run.root, relative, regular=True)
    name = path.relative_to(run.root).as_posix()
    if name in {"run.json", "COMPLETE.json"} or any(
        item["path"] == name for item in run.manifest["artifacts"]
    ):
        raise ValueError("Artifact already registered or reserved")
    record = ArtifactRecord(name, path.stat().st_size, sha256_file(path), kind)
    run.manifest["artifacts"].append(asdict(record))
    _persist(run)
    return record


def write_run_artifact(
    run: RunContext,
    relative: str | Path,
    writer: Callable[[Path], Any],
    *,
    kind: str,
) -> ArtifactRecord:
    ensure_running(run)
    path = safe_path(run.root, relative)
    name = path.relative_to(run.root).as_posix()
    if name in {"run.json", "COMPLETE.json"} or path.exists():
        raise ValueError("Artifact destination already exists or is reserved")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".artifact-",
        suffix="".join(path.suffixes),
        dir=path.parent,
    )
    os.close(descriptor)
    temp = Path(temporary)
    try:
        writer(temp)
        safe_path(run.root, temp.relative_to(run.root), regular=True)
        with temp.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temp, safe_path(run.root, relative))
        return register_artifact(run, relative, kind=kind)
    finally:
        temp.unlink(missing_ok=True)


def write_json_artifact(
    run: RunContext,
    relative: str | Path,
    payload: Any,
    *,
    kind: str,
) -> ArtifactRecord:
    return write_run_artifact(run, relative, lambda path: write_json(path, payload), kind=kind)


def capture_inputs(run: RunContext, input_dir: Path | None = None) -> dict[str, Any]:
    ensure_running(run)
    if run.snapshot is not None:
        raise ValueError("Input snapshot already captured")
    input_dir = _trusted_path(input_dir or run.config.paths.raw_data_dir)
    snapshot = dataset_snapshot(run.config, input_dir)
    run.input_dir, run.snapshot = input_dir, snapshot
    run.manifest["snapshot_id"] = snapshot["snapshot_id"]
    _persist(run)
    write_json_artifact(run, "inputs/dataset-manifest.json", snapshot, kind="dataset_manifest")
    return snapshot


def _verify_outputs(root: Path, manifest: dict[str, Any]) -> None:
    for entry in manifest["artifacts"]:
        if entry["path"] in {"run.json", "COMPLETE.json"}:
            raise ValueError("Reserved artifact path")
        path = safe_path(root, entry["path"], regular=True)
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError("Artifact integrity verification failed")


def finish_run(
    run: RunContext,
    *,
    state: str = "SUCCEEDED",
    error_code: str | None = None,
) -> dict[str, Any]:
    ensure_running(run)
    if state not in {"SUCCEEDED", "FAILED", "INTERRUPTED"}:
        raise ValueError("Invalid terminal state")
    if state == "SUCCEEDED":
        if run.snapshot is None or run.input_dir is None:
            raise ValueError("Run has no captured input snapshot")
        verify_inputs(run.snapshot, run.input_dir)
        _verify_outputs(run.root, run.manifest)
    terminal = {**run.manifest, "state": state, "finished_at_utc": _now(), "error_code": error_code}
    validate_manifest(terminal, successful=state == "SUCCEEDED")
    run.manifest = terminal
    _persist(run)
    run._terminal = True
    if state == "SUCCEEDED":
        marker = {
            "schema_version": 1,
            "run_id": run.run_id,
            "state": "SUCCEEDED",
            "run_sha256": digest_json(terminal),
            "completed_at_utc": _now(),
        }
        write_json(safe_path(run.root, "COMPLETE.json"), marker)
    return run.manifest


def read_completed_run(
    run_dir: Path,
    *,
    expected_mode: str | None = None,
    expected_source: str | None = None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Return fully verified metadata; hash skipping is deliberately unsupported."""
    if not verify_hashes:
        raise ValueError("Completed result consumption requires checksum verification")
    root = _trusted_path(Path(run_dir))
    manifest = validate_manifest(
        read_json(safe_path(root, "run.json", regular=True)), successful=True
    )
    if root.name != manifest["run_id"]:
        raise ValueError("Run directory identity mismatch")
    marker = read_json(safe_path(root, "COMPLETE.json", regular=True))
    if (
        not isinstance(marker, dict)
        or set(marker) != {"schema_version", "run_id", "state", "run_sha256", "completed_at_utc"}
        or type(marker["schema_version"]) is not int
        or marker["schema_version"] != 1
        or marker["run_id"] != manifest["run_id"]
        or marker["state"] != "SUCCEEDED"
        or marker["run_sha256"] != digest_json(manifest)
        or not isinstance(marker["completed_at_utc"], str)
    ):
        raise ValueError("Invalid completion witness")
    try:
        completed_at = datetime.fromisoformat(marker["completed_at_utc"])
        if completed_at.utcoffset() is None or completed_at.utcoffset().total_seconds() != 0:
            raise ValueError("Completion timestamp must use UTC")
    except ValueError as error:
        raise ValueError("Invalid completion timestamp") from error
    if (expected_mode is not None and manifest["execution_mode"] != expected_mode) or (
        expected_source is not None and manifest["data_source"] != expected_source
    ):
        raise ValueError("Run mode/source mismatch")
    _verify_outputs(root, manifest)
    configs = [e for e in manifest["artifacts"] if e["kind"] == "resolved_config"]
    snapshots = [e for e in manifest["artifacts"] if e["kind"] == "dataset_manifest"]
    if len(configs) != 1 or len(snapshots) != 1:
        raise ValueError("Ambiguous provenance artifacts")
    if (
        digest_json(read_json(safe_path(root, configs[0]["path"], regular=True)))
        != manifest["config_sha256"]
    ):
        raise ValueError("Config identity mismatch")
    snapshot = read_json(safe_path(root, snapshots[0]["path"], regular=True))
    snapshot_id = snapshot.pop("snapshot_id", None)
    if snapshot_id != manifest["snapshot_id"] or digest_json(snapshot) != snapshot_id:
        raise ValueError("Snapshot identity mismatch")
    return manifest


def run_result(run: RunContext) -> dict[str, Any]:
    read_completed_run(run.root)
    return {
        "run_id": run.run_id,
        "run_dir": str(run.root),
        "manifest": str(run.root / "run.json"),
        "artifacts": {
            entry["path"]: str(run.root / entry["path"]) for entry in run.manifest["artifacts"]
        },
    }
