"""Independent adversarial checks of the approved execution trust boundaries."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from retail_forecast.artifacts import (
    allocate_run,
    capture_inputs,
    finish_run,
    read_completed_run,
    write_json_artifact,
)
from retail_forecast.config import ProjectConfig
from retail_forecast.io_utils import read_json
from retail_forecast.run_resume import resume_run


def _evaluation(config: ProjectConfig):
    context = allocate_run(config, kind="evaluation")
    capture_inputs(context)
    write_json_artifact(context, "metrics/evaluation.json", {"rmsle": 0.25}, kind="evaluation")
    return context


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def test_failed_marker_publication_is_not_repaired_by_retry(
    synthetic_config: ProjectConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    import retail_forecast.artifacts as artifacts

    context = _evaluation(synthetic_config)
    real_write = artifacts.write_json

    def interrupted_marker(path, payload):
        if path.name == "COMPLETE.json":
            raise OSError("simulated process interruption")
        return real_write(path, payload)

    monkeypatch.setattr(artifacts, "write_json", interrupted_marker)
    with pytest.raises(OSError):
        finish_run(context)
    assert read_json(context.root / "run.json")["state"] == "SUCCEEDED"
    before = _tree_hashes(context.root)
    monkeypatch.setattr(artifacts, "write_json", real_write)
    with pytest.raises(ValueError):
        finish_run(context)
    assert _tree_hashes(context.root) == before
    with pytest.raises((ValueError, FileNotFoundError)):
        read_completed_run(context.root)


@pytest.mark.parametrize("target", ["run.json", "COMPLETE.json", "metrics/evaluation.json"])
def test_external_symlink_is_rejected_by_completed_reader(
    synthetic_config: ProjectConfig, tmp_path: Path, target: str
) -> None:
    context = _evaluation(synthetic_config)
    finish_run(context)
    outside = tmp_path / "outside-sentinel"
    outside.write_bytes((context.root / target).read_bytes())
    original = outside.read_bytes()
    (context.root / target).unlink()
    (context.root / target).symlink_to(outside)
    with pytest.raises(ValueError):
        read_completed_run(context.root)
    assert outside.read_bytes() == original


def test_fifo_artifact_is_rejected_before_attempting_to_read(
    synthetic_config: ProjectConfig,
) -> None:
    context = _evaluation(synthetic_config)
    finish_run(context)
    artifact = context.root / "metrics/evaluation.json"
    artifact.unlink()
    os.mkfifo(artifact)
    with pytest.raises(ValueError, match="type|regular"):
        read_completed_run(context.root)


def test_finite_json_rejects_numeric_overflow(tmp_path: Path) -> None:
    document = tmp_path / "overflow.json"
    document.write_text('{"measurement": 1e999}')
    with pytest.raises(ValueError):
        read_json(document)


def test_success_cannot_carry_an_execution_failure_category(
    synthetic_config: ProjectConfig,
) -> None:
    context = _evaluation(synthetic_config)
    with pytest.raises(ValueError):
        finish_run(context, error_code="EXECUTION_FAILED")
    assert not (context.root / "COMPLETE.json").exists()


def test_corrupted_checkpoint_is_rejected_without_mutating_predecessor_or_allocating_target(
    synthetic_config: ProjectConfig,
) -> None:
    source = allocate_run(synthetic_config, kind="training", run_id="checkpoint-source")
    capture_inputs(source)
    write_json_artifact(source, "state/checkpoint.json", {"step": 1}, kind="checkpoint")
    checkpoint = source.root / "state/checkpoint.json"
    checkpoint.write_text('{"step": 2}')
    before = _tree_hashes(source.root)
    with pytest.raises(ValueError, match="checkpoint.*invalid|integrity"):
        resume_run(
            synthetic_config,
            source_run_id=source.run_id,
            kind="training",
            run_id="must-not-be-allocated",
            validate_checkpoint=lambda *_: "state/checkpoint.json",
        )
    assert not (source.root.parent / "must-not-be-allocated").exists()
    assert _tree_hashes(source.root) == before


def test_pipeline_refuses_symlink_input_before_any_csv_read(
    synthetic_config: ProjectConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pandas as pd

    from retail_forecast.pipeline import train

    source = synthetic_config.paths.raw_data_dir / "train.csv"
    outside = tmp_path / "outside-training.csv"
    source.rename(outside)
    source.symlink_to(outside)
    real_read = pd.read_csv

    def guarded_read(path, *args, **kwargs):
        if Path(path).is_symlink():
            raise AssertionError("producer read the symlink before trust validation")
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded_read)
    with pytest.raises(ValueError, match="Unsafe dataset input"):
        train(synthetic_config)
