"""Portable, truthful and minimized execution metadata."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from retail_forecast.artifacts import allocate_run, capture_inputs, finish_run, write_json_artifact
from retail_forecast.config import PathsConfig
from retail_forecast.io_utils import read_json
from retail_forecast.provenance import code_identity, dataset_snapshot, digest_json, resolved_config
from retail_forecast.reproducibility import set_global_seed


def test_relocated_external_roots_keep_logical_config_identity(synthetic_config, tmp_path):
    changed = replace(
        synthetic_config, paths=PathsConfig(tmp_path / "other-data", tmp_path / "other-runs")
    )
    assert digest_json(resolved_config(changed)) == digest_json(resolved_config(synthetic_config))
    altered = replace(changed, runtime=replace(changed.runtime, seed=44))
    assert digest_json(resolved_config(altered)) != digest_json(resolved_config(synthetic_config))


def test_metadata_has_no_absolute_root_or_exception_secret(synthetic_config, tmp_path, monkeypatch):
    sentinel = "private-secret-sentinel"
    monkeypatch.setenv("PRIVATE_SECRET", sentinel)
    run = allocate_run(synthetic_config, kind="evaluation")
    capture_inputs(run)
    write_json_artifact(run, "evaluation.json", {"rmsle": 1.0}, kind="evaluation")
    finish_run(run)
    for path in run.root.rglob("*.json"):
        text = path.read_text()
        assert sentinel not in text
        assert str(tmp_path) not in text
    assert run.manifest["availability_policy"] == "competition_v1"
    assert run.manifest["hardware"]["pythonhashseed_method"].endswith("not_verified_launch_seed")


def test_symlink_input_rejected_before_dataset_validation(synthetic_config, tmp_path, monkeypatch):
    import retail_forecast.provenance as provenance

    source = synthetic_config.paths.raw_data_dir / "train.csv"
    outside = tmp_path / "outside.csv"
    source.rename(outside)
    source.symlink_to(outside)

    def must_not_read(*args):
        raise AssertionError("validation read unsafe inputs")

    monkeypatch.setattr(provenance, "validate_dataset", must_not_read)
    with pytest.raises(ValueError, match="Unsafe dataset input"):
        dataset_snapshot(synthetic_config, synthetic_config.paths.raw_data_dir)


def test_kaggle_snapshot_checks_official_hash_registry(synthetic_config, monkeypatch):
    import retail_forecast.provenance as provenance

    config = replace(
        synthetic_config, mode="full", data=replace(synthetic_config.data, source="kaggle")
    )
    monkeypatch.setattr(provenance, "validate_dataset", lambda *_: None)
    with pytest.raises(ValueError, match="Kaggle source integrity mismatch"):
        dataset_snapshot(config, config.paths.raw_data_dir)


def test_missing_git_is_explicit_and_not_clean(monkeypatch):
    import retail_forecast.provenance as provenance

    def absent(*args, **kwargs):
        raise OSError("synthetic secret message")

    monkeypatch.setattr(provenance.subprocess, "check_output", absent)
    assert code_identity() == {"git_sha": None, "dirty": True}


def test_dirty_capture_omits_git_status_contents(monkeypatch):
    import retail_forecast.provenance as provenance

    actual_root = Path(provenance.__file__).resolve().parents[2]
    answers = iter(
        [str(actual_root), "src/retail_forecast/__init__.py", "a" * 40, " M private-secret-file\n"]
    )
    monkeypatch.setattr(provenance.subprocess, "check_output", lambda *_a, **_k: next(answers))
    assert code_identity() == {"git_sha": "a" * 40, "dirty": True}


def test_nonfinite_overflow_json_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"value":1e999}')
    with pytest.raises(ValueError, match="Nonfinite"):
        read_json(path)


def test_seed_helper_does_not_rewrite_hash_launch_environment(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "1234")
    set_global_seed(5678)
    import os

    assert os.environ["PYTHONHASHSEED"] == "1234"


def test_ignored_installed_package_cannot_inherit_checkout_revision(tmp_path, monkeypatch):
    import subprocess

    import retail_forecast.provenance as provenance

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text(".venv/\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", ".gitignore"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    installed = tmp_path / ".venv/lib/python/site-packages/retail_forecast/provenance.py"
    installed.parent.mkdir(parents=True)
    installed.write_text("# stale installed package\n")
    monkeypatch.setattr(provenance, "__file__", str(installed))
    assert provenance.code_identity() == {"git_sha": None, "dirty": True}
