"""Verified model-bundle publication and loading (run/bundle contract, bundle v1).

Every file is hashed when read and the same bytes are handed to the adapter, so a
file cannot change between verification and native load. No pickle loader exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, Final

import pandas as pd

from retail_forecast.artifacts import (
    RunContext,
    _trusted_path,
    ensure_running,
    read_completed_run,
    read_regular_bytes,
    safe_path,
    write_run_artifact,
)
from retail_forecast.availability import INPUT_CONTRACT_VERSION
from retail_forecast.io_utils import canonical_json, parse_json_bytes
from retail_forecast.models import get_adapter, supports_version
from retail_forecast.models.base import FitContext, FittedModel
from retail_forecast.provenance import digest_json

BUNDLE_SCHEMA_VERSION: Final = 1
SUPPORTED_BUNDLE_SCHEMAS: Final = frozenset({BUNDLE_SCHEMA_VERSION})
BUNDLE_CODES: Final = frozenset(
    {
        "BUNDLE_INVALID",
        "BUNDLE_MISSING",
        "BUNDLE_UNSUPPORTED",
        "ORIGIN_BEFORE_CUTOFF",
        "ORIGIN_UNSUPPORTED",
        "POLICY_MISMATCH",
    }
)
SLOT = re.compile(r"^(final|fold_[0-9]{1,2})$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
BUNDLE_KEYS: Final = frozenset(
    {
        "schema_version",
        "bundle_id",
        "model_id",
        "model_version",
        "adapter_id",
        "adapter_version",
        "training_run_id",
        "training_cutoff",
        "availability_policy",
        "input_contract_version",
        "data_source",
        "execution_mode",
        "snapshot_id",
        "signature_path",
        "signature_sha256",
        "preprocessing_path",
        "preprocessing_sha256",
        "state_files",
        "supported_series",
        "horizon",
        "min_context_days",
        "request_context_days",
        "code",
        "config_sha256",
        "versions",
        "seed",
        "pretrained",
        "evaluation_ref",
    }
)


class BundleError(ValueError):
    """Bundle failure with a contract category and no local path in the message."""

    def __init__(self, code: str, detail: str) -> None:
        if code not in BUNDLE_CODES:
            raise ValueError("Unknown bundle error code")
        self.code = code
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class LoadedModel:
    """A verified bundle bound to its identity; the only model prediction accepts."""

    fitted: FittedModel
    manifest: dict[str, Any]

    @property
    def bundle_id(self) -> str:
        return self.manifest["bundle_id"]


@dataclass(frozen=True)
class BundleRef:
    """Explicit bundle location: inside a completed run, or a verified capsule."""

    kind: str  # "run", "capsule", or "running" (owning process only)
    root: Path
    relative: str  # bundle directory relative to root

    @property
    def training_run_id(self) -> str | None:
        return self.root.name if self.kind in {"run", "running"} else None


def bundle_relative_dir(model_id: str, slot: str) -> str:
    if not SLOT.fullmatch(slot):
        raise ValueError("Invalid bundle slot")
    get_adapter(model_id)  # unknown IDs fail before any path is formed
    return f"models/{model_id}/{slot}"


def parse_bundle_ref(value: str, *, artifacts_dir: Path) -> BundleRef:
    """Parse `<training_run_id>:models/<model_id>/<slot>` or `capsule:<path>`."""

    from retail_forecast.artifact_schema import validate_run_id

    if not isinstance(value, str) or ":" not in value:
        raise ValueError("BUNDLE_REQUIRED: use --bundle <run_id>:models/<model_id>/<slot>")
    head, tail = value.split(":", 1)
    if head == "capsule":
        return BundleRef("capsule", Path(tail).absolute(), "bundle")
    run_id = validate_run_id(head)
    parts = PurePosixPath(tail).parts
    if len(parts) != 3 or parts[0] != "models":
        raise ValueError("BUNDLE_REQUIRED: bundle path must be models/<model_id>/<slot>")
    relative = bundle_relative_dir(parts[1], parts[2])
    return BundleRef("run", safe_path(artifacts_dir, run_id), relative)


def model_version(fitted: FittedModel) -> str:
    """Immutable recipe identity: adapter version plus signature and recipe digest."""

    recipe = {
        "signature": fitted.signature,
        "params": fitted.preprocessing.get("params"),
        "rule": fitted.preprocessing.get("missing_weekday_rule"),
    }
    return f"{fitted.adapter_id}-{fitted.adapter_version}-{digest_json(recipe)[:16]}"


def save_bundle(
    fitted: FittedModel,
    *,
    run: RunContext,
    context: FitContext,
    slot: str,
    evaluation_ref: str | None = None,
) -> dict[str, Any]:
    """Write state, preprocessing and signature, then the identity-binding manifest last."""

    if context.training_run_id != run.run_id or run.snapshot is None:
        raise ValueError("Bundle must be saved by its own training run with captured inputs")
    if fitted.availability_policy != run.manifest["availability_policy"]:
        raise BundleError("POLICY_MISMATCH", "fitted model and run policies differ")
    relative = bundle_relative_dir(fitted.adapter_id, slot)
    adapter = get_adapter(fitted.adapter_id)
    state = adapter.state_files(fitted)
    if not state or any(not item.data for item in state.values()):
        raise BundleError("BUNDLE_MISSING", "adapter produced no native state")
    if not fitted.preprocessing or not fitted.signature:
        raise BundleError("BUNDLE_MISSING", "adapter produced no learned preprocessing")

    def put(name: str, data: bytes, kind: str) -> dict[str, Any]:
        record = write_run_artifact(
            run, f"{relative}/{name}", lambda path: path.write_bytes(data), kind=kind
        )
        return {"path": name, "bytes": record.bytes, "sha256": record.sha256}

    state_files = []
    for name in sorted(state):
        if not re.fullmatch(r"[a-z0-9_]+\.[a-z]+", name):
            raise BundleError("BUNDLE_INVALID", "adapter state file name is not allowed")
        entry = put(f"state/{name}", state[name].data, "bundle_state")
        state_files.append({**entry, "format": state[name].format})
    signature = put("signature.json", canonical_json(fitted.signature), "bundle_signature")
    preprocessing = put(
        "preprocessing.json", canonical_json(fitted.preprocessing), "bundle_preprocessing"
    )
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "model_id": fitted.adapter_id,
        "model_version": model_version(fitted),
        "adapter_id": fitted.adapter_id,
        "adapter_version": fitted.adapter_version,
        "training_run_id": run.run_id,
        "training_cutoff": fitted.training_cutoff.date().isoformat(),
        "availability_policy": fitted.availability_policy,
        "input_contract_version": fitted.input_contract_version,
        "data_source": run.manifest["data_source"],
        "execution_mode": run.manifest["execution_mode"],
        "snapshot_id": run.manifest["snapshot_id"],
        "signature_path": "signature.json",
        "signature_sha256": signature["sha256"],
        "preprocessing_path": "preprocessing.json",
        "preprocessing_sha256": preprocessing["sha256"],
        "state_files": state_files,
        "supported_series": [[int(s), str(f)] for s, f in fitted.supported_series],
        "horizon": fitted.signature["horizon"],
        "min_context_days": fitted.min_context_days,
        "request_context_days": fitted.request_context_days,
        "code": run.manifest["code"],
        "config_sha256": run.manifest["config_sha256"],
        "versions": run.manifest["versions"],
        "seed": run.manifest["seed"],
        "pretrained": None,
        "evaluation_ref": evaluation_ref,
    }
    manifest["bundle_id"] = digest_json(manifest)
    write_run_artifact(
        run,
        f"{relative}/bundle.json",
        lambda path: path.write_bytes(canonical_json(manifest)),
        kind="bundle_manifest",
    )
    return manifest


def _read_verified(directory: Path, name: str, expected_sha: str | None) -> bytes:
    """Read one regular contained file once and check it against its declared hash."""

    import hashlib

    candidate = directory / name
    if not candidate.exists() and not candidate.is_symlink():
        raise BundleError("BUNDLE_MISSING", f"bundle file {name!r} is unavailable")
    try:
        data = read_regular_bytes(directory, name)
    except (ValueError, OSError) as error:
        raise BundleError("BUNDLE_INVALID", f"bundle file {name!r} is unsafe") from error
    if expected_sha is not None and hashlib.sha256(data).hexdigest() != expected_sha:
        raise BundleError("BUNDLE_INVALID", f"bundle file {name!r} failed integrity")
    return data


def _container_listing(
    ref: BundleRef,
    expected_mode: str | None,
    expected_source: str | None,
    running: RunContext | None,
):
    """Verify the container and return {bundle-relative path: sha256} it vouches for."""

    from retail_forecast.capsules import verify_inference_capsule

    prefix = f"{ref.relative}/"
    if ref.kind == "running":
        # Only the process that owns a RUNNING record may read back what it wrote.
        if running is None or running.root != ref.root:
            raise BundleError("BUNDLE_INVALID", "a running record is not a consumable bundle")
        ensure_running(running)
        listing = {
            entry["path"][len(prefix) :]: entry["sha256"]
            for entry in running.manifest["artifacts"]
            if entry["path"].startswith(prefix)
        }
        return safe_path(ref.root, ref.relative), listing, running.run_id, running.manifest
    if ref.kind == "run":
        if not ref.root.exists():
            raise BundleError("BUNDLE_MISSING", "training run is unavailable")
        try:
            manifest = read_completed_run(
                ref.root, expected_mode=expected_mode, expected_source=expected_source
            )
        except (ValueError, FileNotFoundError, OSError) as error:
            raise BundleError(
                "BUNDLE_INVALID", "training run is not a verified completed record"
            ) from error
        listing = {
            entry["path"][len(prefix) :]: entry["sha256"]
            for entry in manifest["artifacts"]
            if entry["path"].startswith(prefix)
        }
        return safe_path(ref.root, ref.relative), listing, manifest["run_id"], manifest
    if ref.kind == "capsule":
        try:
            capsule = verify_inference_capsule(ref.root)
        except FileNotFoundError as error:
            raise BundleError("BUNDLE_MISSING", "capsule is unavailable") from error
        except (ValueError, OSError) as error:
            raise BundleError("BUNDLE_INVALID", "capsule is not verified") from error
        listing = {
            entry["path"][len("bundle/") :]: entry["sha256"]
            for entry in capsule["files"]
            if entry["path"].startswith("bundle/")
        }
        from retail_forecast.io_utils import parse_json_bytes as _parse

        witness = _parse(read_regular_bytes(_trusted_path(ref.root), "provenance/source-run.json"))
        if (
            capsule["availability_policy"] != witness["availability_policy"]
            or capsule["execution_mode"] != witness["execution_mode"]
            or capsule["data_source"] != witness["data_source"]
        ):
            raise BundleError("BUNDLE_INVALID", "capsule metadata differs from its witness")
        bundle_dir = safe_path(_trusted_path(ref.root), "bundle")
        return bundle_dir, listing, witness["run_id"], witness
    raise BundleError("BUNDLE_INVALID", "unknown bundle location kind")


def validate_bundle_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict) or set(manifest) != BUNDLE_KEYS:
        raise BundleError("BUNDLE_INVALID", "bundle manifest schema mismatch")
    if type(manifest["schema_version"]) is not int or (
        manifest["schema_version"] not in SUPPORTED_BUNDLE_SCHEMAS
    ):
        raise BundleError("BUNDLE_UNSUPPORTED", "bundle schema version is not supported")
    body = {key: value for key, value in manifest.items() if key != "bundle_id"}
    if manifest["bundle_id"] != digest_json(body):
        raise BundleError("BUNDLE_INVALID", "bundle identity does not match its content")
    if manifest["adapter_id"] != manifest["model_id"] or not supports_version(
        manifest["adapter_id"], manifest["adapter_version"]
    ):
        raise BundleError("BUNDLE_UNSUPPORTED", "adapter or adapter major version is not supported")
    for key in ("signature_sha256", "preprocessing_sha256", "snapshot_id", "config_sha256"):
        if not isinstance(manifest[key], str) or not SHA256.fullmatch(manifest[key]):
            raise BundleError("BUNDLE_INVALID", f"invalid {key}")
    if manifest["signature_path"] != "signature.json" or (
        manifest["preprocessing_path"] != "preprocessing.json"
    ):
        raise BundleError("BUNDLE_INVALID", "unexpected bundle file layout")
    state = manifest["state_files"]
    if not isinstance(state, list) or not state:
        raise BundleError("BUNDLE_MISSING", "bundle declares no native state")
    for entry in state:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "bytes", "sha256", "format"}
            or not isinstance(entry["path"], str)
            or not re.fullmatch(r"state/[a-z0-9_]+\.[a-z]+", entry["path"])
            or not isinstance(entry["sha256"], str)
            or not SHA256.fullmatch(entry["sha256"])
        ):
            raise BundleError("BUNDLE_INVALID", "invalid state file entry")
    if (manifest["execution_mode"], manifest["data_source"]) not in {
        ("smoke", "synthetic"),
        ("full", "kaggle"),
    }:
        raise BundleError("BUNDLE_INVALID", "invalid mode/source identity")
    series = manifest["supported_series"]
    if (
        not isinstance(series, list)
        or not series
        or any(
            not isinstance(item, list)
            or len(item) != 2
            or type(item[0]) is not int
            or not isinstance(item[1], str)
            for item in series
        )
        or len({tuple(item) for item in series}) != len(series)
    ):
        raise BundleError("BUNDLE_INVALID", "invalid supported series")
    for key in ("horizon", "min_context_days", "request_context_days", "seed"):
        if type(manifest[key]) is not int or manifest[key] < (0 if key == "seed" else 1):
            raise BundleError("BUNDLE_INVALID", f"invalid {key}")
    cutoff = manifest["training_cutoff"]
    try:
        if not isinstance(cutoff, str) or date.fromisoformat(cutoff).isoformat() != cutoff:
            raise ValueError
    except ValueError as error:
        raise BundleError("BUNDLE_INVALID", "invalid training_cutoff") from error
    return manifest


def load_bundle(
    ref: BundleRef,
    *,
    expected_policy: str,
    expected_input_version: int = INPUT_CONTRACT_VERSION,
    expected_mode: str | None = None,
    expected_source: str | None = None,
    running: RunContext | None = None,
) -> LoadedModel:
    """Verify container, manifest, files, policy and adapter, then build the model."""

    directory, listing, container_run_id, run_identity = _container_listing(
        ref, expected_mode, expected_source, running
    )
    if "bundle.json" not in listing:
        raise BundleError("BUNDLE_MISSING", "bundle manifest is not listed by its container")
    raw = _read_verified(directory, "bundle.json", listing["bundle.json"])
    try:
        manifest = validate_bundle_manifest(parse_json_bytes(raw))
    except BundleError:
        raise
    except ValueError as error:
        raise BundleError("BUNDLE_INVALID", "bundle manifest is not valid JSON") from error
    if container_run_id is not None and manifest["training_run_id"] != container_run_id:
        raise BundleError("BUNDLE_INVALID", "bundle names another training run")
    if ref.kind == "capsule":
        capsule_manifest = parse_json_bytes(
            read_regular_bytes(_trusted_path(ref.root), "capsule.json")
        )
        if any(
            capsule_manifest[key] != manifest[key]
            for key in (
                "availability_policy",
                "input_contract_version",
                "execution_mode",
                "data_source",
                "bundle_id",
            )
        ):
            raise BundleError("BUNDLE_INVALID", "capsule metadata differs from its bundle")
    if run_identity is not None and any(
        manifest[key] != run_identity[key]
        for key in (
            "snapshot_id",
            "config_sha256",
            "code",
            "seed",
            "data_source",
            "execution_mode",
            "availability_policy",
        )
    ):
        raise BundleError("BUNDLE_INVALID", "bundle provenance differs from its training run")
    if manifest["availability_policy"] != expected_policy:
        raise BundleError("POLICY_MISMATCH", "bundle policy differs from the request policy")
    if manifest["input_contract_version"] != expected_input_version:
        raise BundleError("POLICY_MISMATCH", "bundle input contract version differs")
    if (expected_mode is not None and manifest["execution_mode"] != expected_mode) or (
        expected_source is not None and manifest["data_source"] != expected_source
    ):
        raise BundleError("BUNDLE_INVALID", "bundle mode/source differs from this execution")

    declared = {
        "signature.json": manifest["signature_sha256"],
        "preprocessing.json": manifest["preprocessing_sha256"],
        **{entry["path"]: entry["sha256"] for entry in manifest["state_files"]},
    }
    expected_listing = {**declared, "bundle.json": listing["bundle.json"]}
    if set(listing) != set(expected_listing):
        raise BundleError("BUNDLE_INVALID", "bundle file closure differs from its container")
    contents = {}
    for name, sha in declared.items():
        if listing.get(name) != sha:
            raise BundleError("BUNDLE_INVALID", f"container hash for {name!r} differs")
        contents[name] = _read_verified(directory, name, sha)
    try:
        fitted = _build_model(manifest, contents)
    except BundleError:
        raise
    except Exception as error:  # any parse/native failure is an invalid bundle
        # Inner messages can carry local paths or record values; never forward them.
        raise BundleError("BUNDLE_INVALID", "bundle content was rejected") from error
    return LoadedModel(fitted, manifest)


def _build_model(manifest: dict[str, Any], contents: dict[str, bytes]) -> FittedModel:
    signature = parse_json_bytes(contents["signature.json"])
    preprocessing = parse_json_bytes(contents["preprocessing.json"])
    if not isinstance(signature, dict) or not isinstance(preprocessing, dict):
        raise BundleError("BUNDLE_INVALID", "bundle JSON has the wrong shape")
    if (
        signature.get("adapter_id") != manifest["adapter_id"]
        or signature.get("adapter_version") != manifest["adapter_version"]
        or signature.get("availability_policy") != manifest["availability_policy"]
        or signature.get("horizon") != manifest["horizon"]
    ):
        raise BundleError("BUNDLE_INVALID", "signature differs from the bundle manifest")
    supported = tuple((store, family) for store, family in manifest["supported_series"])
    fitted = get_adapter(manifest["adapter_id"]).load(
        signature=signature,
        preprocessing=preprocessing,
        state={
            name.removeprefix("state/"): contents[name]
            for name in contents
            if name.startswith("state/")
        },
        identity={"supported_series": supported},
    )
    if fitted.training_cutoff != pd.Timestamp(manifest["training_cutoff"]):
        raise BundleError("BUNDLE_INVALID", "training cutoff differs from the manifest")
    return fitted
