"""Model-owned checkpoint validation with immutable predecessor linkage."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from retail_forecast.artifact_schema import validate_manifest
from retail_forecast.artifacts import (
    RunContext,
    _verify_outputs,
    allocate_run,
    capture_inputs,
    read_completed_run,
    resolve_run,
    safe_path,
)
from retail_forecast.config import ProjectConfig
from retail_forecast.io_utils import read_json
from retail_forecast.provenance import (
    code_identity,
    dataset_snapshot,
    digest_json,
    resolved_config,
    sha256_file,
)

CheckpointValidator = Callable[[Path, dict[str, Any]], str]


def resume_run(
    config: ProjectConfig,
    *,
    source_run_id: str,
    kind: str,
    validate_checkpoint: CheckpointValidator,
    run_id: str | None = None,
    input_dir: Path | None = None,
) -> RunContext:
    """Validate a model-selected checkpoint, then allocate a distinct linked run.

    The validator returns the relative path of an eligible checkpoint already
    registered by its producer. It owns adapter/signature/preprocessor eligibility;
    the orchestration verifies source/checkpoint bytes and execution identities.
    """
    source = resolve_run(config, source_run_id)
    manifest = validate_manifest(read_json(safe_path(source, "run.json", regular=True)))
    if manifest["run_id"] != source_run_id:
        raise ValueError("Resume source identity mismatch")
    if manifest["state"] == "SUCCEEDED":
        manifest = read_completed_run(source)
    expected_snapshot = dataset_snapshot(config, input_dir or config.paths.raw_data_dir)
    if (
        manifest["execution_mode"] != config.mode
        or manifest["data_source"] != config.data.source
        or manifest["config_sha256"] != digest_json(resolved_config(config))
        or manifest["snapshot_id"] != expected_snapshot["snapshot_id"]
        or manifest["code"] != code_identity()
    ):
        raise ValueError("Resume source is incompatible")
    _verify_outputs(source, manifest)
    checkpoint = validate_checkpoint(source, manifest)
    path = safe_path(source, checkpoint, regular=True)
    records = [entry for entry in manifest["artifacts"] if entry["path"] == checkpoint]
    if (
        len(records) != 1
        or records[0]["kind"] != "checkpoint"
        or records[0]["bytes"] != path.stat().st_size
        or records[0]["sha256"] != sha256_file(path)
    ):
        raise ValueError("Resume checkpoint integrity is invalid")
    run = allocate_run(
        config,
        kind=kind,
        run_id=run_id,
        parent_run_ids=(source_run_id,),
        resumed_from_run_id=source_run_id,
    )
    capture_inputs(run, input_dir)
    return run
