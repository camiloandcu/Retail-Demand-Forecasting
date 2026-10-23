"""Strict run-record validation and producer output obligations."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_KINDS = {
    "training": {"model", "dataset_summary"},
    "evaluation": {"evaluation"},
    "prediction": {"submission"},
    "smoke": {"model", "dataset_summary", "evaluation", "submission", "smoke_summary"},
    "baseline": {
        "oof",
        "segmented_metrics",
        "costs",
        "selection",
        "experiment_summary",
        "figure",
        "native_model",
    },
    "audit": {"audit_manifest", "audit_table", "figure"},
    "features": {"split_manifest", "feature_manifest"},
}
MANIFEST_KEYS = {
    "schema_version",
    "run_id",
    "kind",
    "execution_mode",
    "data_source",
    "state",
    "code",
    "config_sha256",
    "input_contract_version",
    "availability_policy",
    "snapshot_id",
    "seed",
    "started_at_utc",
    "finished_at_utc",
    "parent_run_ids",
    "resumed_from_run_id",
    "versions",
    "hardware",
    "artifacts",
    "error_code",
}


def validate_run_id(value: Any) -> str:
    if not isinstance(value, str) or not RUN_ID.fullmatch(value):
        raise ValueError("Invalid run identity")
    return value


def _hash(value: Any) -> None:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ValueError("Invalid checksum")


def validate_manifest(value: Any, *, successful: bool = False) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != MANIFEST_KEYS:
        raise ValueError("Invalid run schema")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported run schema")
    validate_run_id(value["run_id"])
    if not isinstance(value["kind"], str) or value["kind"] not in REQUIRED_KINDS:
        raise ValueError("Invalid producer kind")
    if (
        not isinstance(value["execution_mode"], str)
        or not isinstance(value["data_source"], str)
        or (value["execution_mode"], value["data_source"])
        not in {
            ("smoke", "synthetic"),
            ("full", "kaggle"),
        }
    ):
        raise ValueError("Invalid mode/source identity")
    if not isinstance(value["state"], str) or value["state"] not in {
        "RUNNING",
        "SUCCEEDED",
        "FAILED",
        "INTERRUPTED",
    }:
        raise ValueError("Invalid run state")
    if successful and value["state"] != "SUCCEEDED":
        raise ValueError("Run is not successful")
    code = value["code"]
    if (
        not isinstance(code, dict)
        or set(code) != {"git_sha", "dirty"}
        or type(code["dirty"]) is not bool
    ):
        raise ValueError("Invalid code provenance")
    if code["git_sha"] is not None and (
        not isinstance(code["git_sha"], str) or not re.fullmatch(r"[0-9a-f]{40}", code["git_sha"])
    ):
        raise ValueError("Invalid code revision")
    if code["git_sha"] is None and not code["dirty"]:
        raise ValueError("Unidentified code cannot be clean")
    _hash(value["config_sha256"])
    if value["snapshot_id"] is not None:
        _hash(value["snapshot_id"])
    if type(value["input_contract_version"]) is not int or value["input_contract_version"] != 1:
        raise ValueError("Invalid input contract")
    if (
        not isinstance(value["availability_policy"], str)
        or value["availability_policy"] != "competition_v1"
    ):
        raise ValueError("Unsupported availability policy")
    if type(value["seed"]) is not int:
        raise ValueError("Invalid seed")
    if not isinstance(value["parent_run_ids"], list):
        raise ValueError("Invalid parents")
    for parent in value["parent_run_ids"]:
        validate_run_id(parent)
    if value["resumed_from_run_id"] is not None:
        validate_run_id(value["resumed_from_run_id"])
    if not isinstance(value["versions"], dict) or not isinstance(value["hardware"], dict):
        raise ValueError("Invalid measurement context")
    for timestamp in (value["started_at_utc"], value["finished_at_utc"]):
        if timestamp is None:
            continue
        if not isinstance(timestamp, str):
            raise ValueError("Invalid run timestamp")
        try:
            observed = datetime.fromisoformat(timestamp)
            if observed.utcoffset() is None or observed.utcoffset().total_seconds() != 0:
                raise ValueError("Timestamp must be UTC")
        except (ValueError, TypeError) as error:
            raise ValueError("Invalid run timestamp") from error
    if value["started_at_utc"] is None:
        raise ValueError("Missing start time")
    if successful and (value["snapshot_id"] is None or value["finished_at_utc"] is None):
        raise ValueError("Missing successful provenance")
    if not (value["error_code"] is None or isinstance(value["error_code"], str)):
        raise ValueError("Invalid error category")
    if value["error_code"] not in {
        None,
        "EXECUTION_FAILED",
        "INTERRUPTED",
        "INPUT_CHANGED",
        "RESUME_INVALID",
        "TRACKING_UNSAFE",
    }:
        raise ValueError("Invalid error category")
    if value["state"] == "SUCCEEDED" and value["error_code"] is not None:
        raise ValueError("Successful run cannot contain a failure category")
    if not isinstance(value["artifacts"], list):
        raise ValueError("Invalid artifacts")
    seen = set()
    for entry in value["artifacts"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256", "kind"}:
            raise ValueError("Invalid artifact schema")
        if not isinstance(entry["path"], str) or entry["path"] in seen:
            raise ValueError("Duplicate or invalid artifact")
        seen.add(entry["path"])
        if type(entry["bytes"]) is not int or entry["bytes"] < 0:
            raise ValueError("Invalid artifact size")
        _hash(entry["sha256"])
        if not isinstance(entry["kind"], str) or not entry["kind"]:
            raise ValueError("Invalid artifact kind")
    if successful:
        required = REQUIRED_KINDS[value["kind"]] | {"resolved_config", "dataset_manifest"}
        if not required.issubset({entry["kind"] for entry in value["artifacts"]}):
            raise ValueError("Missing producer-required artifacts")
    return value
