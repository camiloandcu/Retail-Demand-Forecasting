"""Behavioral run lifecycle, integrity and filesystem denial verification."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from retail_forecast.artifacts import (
    allocate_run,
    capture_inputs,
    finish_run,
    read_completed_run,
    resolve_run,
    run_result,
    write_json_artifact,
    write_run_artifact,
)
from retail_forecast.config import ProjectConfig
from retail_forecast.io_utils import canonical_json, read_json, write_json
from retail_forecast.provenance import digest_json
from retail_forecast.run_resume import resume_run


def completed(config: ProjectConfig, name: str = "test-run"):
    run = allocate_run(config, kind="evaluation", run_id=name)
    capture_inputs(run)
    write_json_artifact(run, "metrics/evaluation.json", {"score": 1.0}, kind="evaluation")
    finish_run(run)
    return run


def test_distinct_and_contended_allocation(synthetic_config):
    with ThreadPoolExecutor(2) as pool:
        runs = list(pool.map(lambda _: allocate_run(synthetic_config, kind="evaluation"), range(2)))
    assert runs[0].root != runs[1].root

    def allocate(_):
        try:
            return allocate_run(synthetic_config, kind="evaluation", run_id="same").root
        except FileExistsError:
            return None

    with ThreadPoolExecutor(2) as pool:
        roots = list(pool.map(allocate, range(2)))
    assert sum(root is not None for root in roots) == 1


def test_complete_and_immutable(synthetic_config):
    run = completed(synthetic_config)
    manifest = read_completed_run(run.root, expected_mode="smoke", expected_source="synthetic")
    assert manifest["state"] == "SUCCEEDED"
    assert run_result(run)["run_id"] == "test-run"
    with pytest.raises(ValueError, match="immutable"):
        write_json_artifact(run, "another.json", {}, kind="evaluation")
    with pytest.raises(ValueError):
        read_completed_run(run.root, expected_mode="full")
    with pytest.raises(ValueError):
        read_completed_run(run.root, verify_hashes=False)


def test_missing_required_output_and_input_mutation(synthetic_config):
    run = allocate_run(synthetic_config, kind="evaluation")
    capture_inputs(run)
    with pytest.raises(ValueError, match="required"):
        finish_run(run)
    write_json_artifact(run, "evaluation.json", {}, kind="evaluation")
    path = synthetic_config.paths.raw_data_dir / "train.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Input snapshot changed"):
        finish_run(run)
    assert not (run.root / "COMPLETE.json").exists()


@pytest.mark.parametrize("relative", ["../outside", "/tmp/outside", "a/../../outside", "a\\b"])
def test_path_denials(synthetic_config, tmp_path, relative):
    outside = tmp_path / "outside"
    outside.write_text("unchanged")
    run = allocate_run(synthetic_config, kind="evaluation")
    with pytest.raises(ValueError):
        write_json_artifact(run, relative, {}, kind="evaluation")
    assert outside.read_text() == "unchanged"


def test_symlinks_and_special_files_rejected(synthetic_config, tmp_path):
    run = allocate_run(synthetic_config, kind="evaluation")
    outside = tmp_path / "outside"
    outside.mkdir()
    (run.root / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        write_json_artifact(run, "escape/x.json", {}, kind="evaluation")
    (run.root / "run.json").unlink()
    (run.root / "run.json").symlink_to(outside / "sentinel")
    with pytest.raises(ValueError):
        finish_run(run, state="FAILED")
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("run_id", ["../bad", "/absolute", "bad/name", "", ".", "x" * 97])
def test_invalid_ids(synthetic_config, run_id):
    with pytest.raises(ValueError):
        resolve_run(synthetic_config, run_id)


def test_corruption_and_completion_mismatch(synthetic_config):
    run = completed(synthetic_config)
    path = run.root / "metrics/evaluation.json"
    original = path.read_bytes()
    path.write_bytes(original.replace(b"1.0", b"2.0"))
    with pytest.raises(ValueError, match="integrity"):
        read_completed_run(run.root)
    path.write_bytes(original)
    marker = read_json(run.root / "COMPLETE.json")
    marker["run_id"] = "other"
    write_json(run.root / "COMPLETE.json", marker)
    with pytest.raises(ValueError, match="witness"):
        read_completed_run(run.root)


@pytest.mark.parametrize("boundary", ["artifact", "terminal", "marker"])
def test_interrupted_publication(synthetic_config, monkeypatch, boundary):
    import retail_forecast.artifacts as artifacts

    run = allocate_run(synthetic_config, kind="evaluation")
    capture_inputs(run)
    if boundary == "artifact":

        def writer(path):
            path.write_text("partial")
            raise RuntimeError("interrupted")

        with pytest.raises(RuntimeError):
            write_run_artifact(run, "partial.json", writer, kind="evaluation")
    else:
        write_json_artifact(run, "evaluation.json", {}, kind="evaluation")
        actual = artifacts.write_json

        def interrupt(path, payload):
            if (boundary == "terminal" and path.name == "run.json") or path.name == "COMPLETE.json":
                raise RuntimeError("interrupted")
            actual(path, payload)

        monkeypatch.setattr(artifacts, "write_json", interrupt)
        with pytest.raises(RuntimeError):
            finish_run(run)
    with pytest.raises((ValueError, FileNotFoundError)):
        read_completed_run(run.root)
    assert not (run.root / "COMPLETE.json").exists()


def test_strict_json_and_schema(synthetic_config):
    for payload in ({"x": float("nan")}, {"x": object()}):
        with pytest.raises((ValueError, TypeError)):
            canonical_json(payload)
    run = completed(synthetic_config)
    (run.root / "run.json").write_text('{"x":1,"x":2}')
    with pytest.raises(ValueError, match="Duplicate"):
        read_completed_run(run.root)


def test_rebound_manifest_cannot_remove_required_outputs(synthetic_config):
    run = completed(synthetic_config)
    manifest = read_json(run.root / "run.json")
    manifest["artifacts"] = [a for a in manifest["artifacts"] if a["kind"] != "evaluation"]
    write_json(run.root / "run.json", manifest)
    marker = read_json(run.root / "COMPLETE.json")
    marker["run_sha256"] = digest_json(manifest)
    write_json(run.root / "COMPLETE.json", marker)
    with pytest.raises(ValueError, match="required"):
        read_completed_run(run.root)


def test_resume_interrupted_source_and_mismatch(synthetic_config):
    source = allocate_run(synthetic_config, kind="training", run_id="source")
    capture_inputs(source)
    write_json_artifact(source, "checkpoint.json", {"epoch": 1}, kind="checkpoint")
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.root.rglob("*")
        if p.is_file()
    }
    target = resume_run(
        synthetic_config,
        source_run_id="source",
        kind="training",
        validate_checkpoint=lambda *_: "checkpoint.json",
        run_id="resumed",
    )
    assert target.manifest["resumed_from_run_id"] == "source"
    with pytest.raises(ValueError):
        read_completed_run(source.root)
    wrong = replace(synthetic_config, runtime=replace(synthetic_config.runtime, seed=12))
    with pytest.raises(ValueError, match="incompatible"):
        resume_run(
            wrong,
            source_run_id="source",
            kind="training",
            validate_checkpoint=lambda *_: "checkpoint.json",
        )
    after = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.root.rglob("*")
        if p.is_file()
    }
    assert before == after


@pytest.mark.parametrize(
    "timestamp", ["invalid", "2026-10-23T20:00:00", "2026-10-23T20:00:00+02:00"]
)
def test_completion_timestamp_requires_utc(synthetic_config, timestamp):
    run = completed(synthetic_config)
    path = run.root / "COMPLETE.json"
    marker = read_json(path)
    marker["completed_at_utc"] = timestamp
    write_json(path, marker)
    with pytest.raises(ValueError, match="completion timestamp"):
        read_completed_run(run.root)
