"""Portable inference capsule v1: a verified bundle's closure plus its run witness.

Checksums and witnesses establish integrity and provenance consistency, not
authenticity: an attacker able to replace both files and manifests is out of scope.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from retail_forecast.artifacts import (
    _trusted_path,
    read_completed_run,
    read_regular_bytes,
    safe_path,
    tree_matches,
)
from retail_forecast.io_utils import canonical_json, parse_json_bytes
from retail_forecast.provenance import code_identity, digest_json

CAPSULE_SCHEMA_VERSION: Final = 1
CAPSULE_NAME: Final = "capsule.json"
MARKER_NAME: Final = "CAPSULE_COMPLETE.json"
CAPSULE_ID = re.compile(r"^[0-9a-f]{64}$")
CAPSULE_KEYS: Final = frozenset(
    {
        "schema_version",
        "capsule_id",
        "bundle_id",
        "source_training_run_id",
        "source_bundle_relative_path",
        "source_run_sha256",
        "source_complete_sha256",
        "data_source",
        "execution_mode",
        "availability_policy",
        "input_contract_version",
        "export_code",
        "exported_at_utc",
        "files",
    }
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def export_inference_capsule(
    source_run_dir: Path, *, bundle_relative_path: str, destination_root: Path
) -> dict[str, Any]:
    """Copy only a verified bundle's inference closure and witness; publish atomically."""

    from retail_forecast.bundles import BundleRef, load_bundle

    source = _trusted_path(Path(source_run_dir))
    manifest = read_completed_run(source)
    ref = BundleRef("run", source, bundle_relative_path)
    loaded = load_bundle(ref, expected_policy=manifest["availability_policy"])  # full checks
    bundle_dir = safe_path(source, bundle_relative_path)
    prefix = f"{bundle_relative_path}/"
    closure = {
        entry["path"][len(prefix) :]: entry["sha256"]
        for entry in manifest["artifacts"]
        if entry["path"].startswith(prefix)
    }
    files: dict[str, tuple[bytes, str]] = {}
    for name, sha in sorted(closure.items()):
        data = read_regular_bytes(bundle_dir, name)
        if _sha(data) != sha:
            raise ValueError("Bundle file changed during export")
        files[f"bundle/{name}"] = (data, "bundle")
    run_bytes = read_regular_bytes(source, "run.json")
    complete_bytes = read_regular_bytes(source, "COMPLETE.json")
    files["provenance/source-run.json"] = (run_bytes, "witness_run")
    files["provenance/source-complete.json"] = (complete_bytes, "witness_completion")
    capsule = {
        "schema_version": CAPSULE_SCHEMA_VERSION,
        "bundle_id": loaded.bundle_id,
        "source_training_run_id": manifest["run_id"],
        "source_bundle_relative_path": bundle_relative_path,
        "source_run_sha256": _sha(run_bytes),
        "source_complete_sha256": _sha(complete_bytes),
        "data_source": manifest["data_source"],
        "execution_mode": manifest["execution_mode"],
        "availability_policy": loaded.manifest["availability_policy"],
        "input_contract_version": loaded.manifest["input_contract_version"],
        "export_code": code_identity(),
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [
            {"path": path, "bytes": len(data), "sha256": _sha(data), "kind": kind}
            for path, (data, kind) in sorted(files.items())
        ],
    }
    capsule["capsule_id"] = digest_json(capsule)
    capsule_bytes = canonical_json(capsule)
    marker = {
        "schema_version": CAPSULE_SCHEMA_VERSION,
        "capsule_id": capsule["capsule_id"],
        "state": "SUCCEEDED",
        "capsule_sha256": _sha(capsule_bytes),
    }

    destination = _trusted_path(Path(destination_root))
    destination.mkdir(parents=True, exist_ok=True)
    final = destination / capsule["capsule_id"]
    if final.exists() or final.is_symlink():
        raise ValueError("Capsule identity already exists; capsules are never overwritten")
    staging = destination / f".staging-{secrets.token_hex(8)}"
    staging.mkdir()
    try:
        for path, (data, _) in files.items():
            target = staging / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (staging / CAPSULE_NAME).write_bytes(capsule_bytes)
        (staging / MARKER_NAME).write_bytes(canonical_json(marker))
        for path in staging.rglob("*"):
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        verify_inference_capsule(staging, staging=True)
        os.rename(staging, final)  # atomic publish of directory and marker together
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return capsule


def verify_inference_capsule(capsule_root: Path, *, staging: bool = False) -> dict[str, Any]:
    """Validate marker, manifest, closure, checksums and witness without original paths."""

    root = _trusted_path(Path(capsule_root))
    if not root.is_dir():
        raise FileNotFoundError("Capsule root is unavailable")
    marker = parse_json_bytes(read_regular_bytes(root, MARKER_NAME))
    capsule_bytes = read_regular_bytes(root, CAPSULE_NAME)
    if (
        not isinstance(marker, dict)
        or set(marker) != {"schema_version", "capsule_id", "state", "capsule_sha256"}
        or marker["schema_version"] != CAPSULE_SCHEMA_VERSION
        or marker["state"] != "SUCCEEDED"
        or marker["capsule_sha256"] != _sha(capsule_bytes)
    ):
        raise ValueError("Invalid capsule completion marker")
    capsule = parse_json_bytes(capsule_bytes)
    if not isinstance(capsule, dict) or set(capsule) != CAPSULE_KEYS:
        raise ValueError("Invalid capsule manifest schema")
    if capsule["schema_version"] != CAPSULE_SCHEMA_VERSION:
        raise ValueError("Unsupported capsule schema")
    body = {key: value for key, value in capsule.items() if key != "capsule_id"}
    if (
        capsule["capsule_id"] != digest_json(body)
        or marker["capsule_id"] != capsule["capsule_id"]
        or not CAPSULE_ID.fullmatch(capsule["capsule_id"])
        or (not staging and root.name != capsule["capsule_id"])
    ):
        raise ValueError("Capsule identity mismatch")

    declared = {}
    for entry in capsule["files"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256", "kind"}:
            raise ValueError("Invalid capsule file entry")
        data = read_regular_bytes(root, entry["path"])
        if len(data) != entry["bytes"] or _sha(data) != entry["sha256"]:
            raise ValueError("Capsule file failed integrity verification")
        declared[entry["path"]] = data
    if not tree_matches(root, set(declared) | {CAPSULE_NAME, MARKER_NAME}):
        raise ValueError("Capsule contains undeclared, special or missing entries")

    run_bytes = declared.get("provenance/source-run.json")
    complete_bytes = declared.get("provenance/source-complete.json")
    if run_bytes is None or complete_bytes is None or "bundle/bundle.json" not in declared:
        raise ValueError("Capsule witness or bundle manifest is missing")
    if (
        _sha(run_bytes) != capsule["source_run_sha256"]
        or _sha(complete_bytes) != capsule["source_complete_sha256"]
    ):
        raise ValueError("Capsule witness checksum mismatch")
    run = parse_json_bytes(run_bytes)
    complete = parse_json_bytes(complete_bytes)
    if (
        complete.get("run_sha256") != digest_json(run)
        or complete.get("run_id") != run.get("run_id")
        or complete.get("state") != "SUCCEEDED"
        or run.get("state") != "SUCCEEDED"
        or run.get("run_id") != capsule["source_training_run_id"]
        or run.get("data_source") != capsule["data_source"]
        or run.get("execution_mode") != capsule["execution_mode"]
    ):
        raise ValueError("Capsule witness is inconsistent")
    listed = {entry["path"]: entry["sha256"] for entry in run.get("artifacts", [])}
    prefix = f"{capsule['source_bundle_relative_path']}/"
    source_closure = {path[len(prefix) :] for path in listed if path.startswith(prefix)}
    copied = {path[len("bundle/") :] for path in declared if path.startswith("bundle/")}
    if copied != source_closure:
        raise ValueError("Capsule bundle closure differs from the source run listing")
    for name in copied:
        if listed[f"{prefix}{name}"] != _sha(declared[f"bundle/{name}"]):
            raise ValueError("Capsule bundle file differs from the source run listing")
    bundle = parse_json_bytes(declared["bundle/bundle.json"])
    if (
        bundle.get("bundle_id") != capsule["bundle_id"]
        or bundle.get("training_run_id") != capsule["source_training_run_id"]
    ):
        raise ValueError("Capsule bundle identity mismatch")
    return capsule
