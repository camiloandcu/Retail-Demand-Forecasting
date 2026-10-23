"""Actual tiny baseline verifies run-scoped scientific outputs and tracking."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from retail_forecast.artifacts import allocate_run, capture_inputs, finish_run, read_completed_run
from retail_forecast.config import ProjectConfig
from retail_forecast.experiments import MODEL_IDS, run_baseline_experiments
from retail_forecast.metrics import rmsle


def test_real_baseline_isolated_and_preserves_oof_scores(
    synthetic_config: ProjectConfig,
) -> None:
    pytest.importorskip("lightgbm")
    pytest.importorskip("matplotlib")
    config = replace(
        synthetic_config,
        baseline=replace(synthetic_config.baseline, tree_n_estimators=2),
    )
    results = []
    for _ in range(2):
        run = allocate_run(config, kind="baseline")
        capture_inputs(run)
        result = run_baseline_experiments(config, run=run, tracking=False)
        finish_run(run)
        completed = read_completed_run(run.root)
        assert completed["availability_policy"] == "competition_v1"
        assert {record["kind"] for record in completed["artifacts"]} >= {
            "oof",
            "segmented_metrics",
            "costs",
            "selection",
            "experiment_summary",
            "figure",
            "native_model",
        }
        assert result["manifest"]["tracking"] == {"status": "disabled", "reason": "requested"}
        assert str(config.project_root) not in json.dumps(result["manifest"])
        for model_id in MODEL_IDS:
            frame = result["oof"].loc[result["oof"]["model_id"].eq(model_id)]
            expected = (
                result["results"]
                .loc[
                    result["results"]["model_id"].eq(model_id)
                    & result["results"]["scope"].eq("global"),
                    "rmsle",
                ]
                .item()
            )
            assert rmsle(frame["y_true"], frame["y_pred"]) == pytest.approx(expected)
        results.append(result)
    assert results[0]["paths"]["manifest"] != results[1]["paths"]["manifest"]
    pd.testing.assert_frame_equal(results[0]["oof"], results[1]["oof"])
    assert np.isfinite(results[0]["oof"]["y_pred"]).all()


def test_baseline_wrapper_contains_enabled_tracking_and_closes_database(
    synthetic_config: ProjectConfig,
) -> None:
    pytest.importorskip("lightgbm")
    pytest.importorskip("matplotlib")
    import importlib.util
    import sqlite3
    import sys
    from pathlib import Path

    from retail_forecast.pipeline import baseline_experiment

    if importlib.util.find_spec("mlflow") is None:
        pytest.skip("optional MLflow is not installed")
    config = replace(
        synthetic_config,
        baseline=replace(synthetic_config.baseline, tree_n_estimators=2),
    )
    parent_mlflow_module = sys.modules.get("mlflow")
    result = baseline_experiment(config, tracking=True)
    root = Path(result["run_dir"])
    completed = read_completed_run(root)
    summary = json.loads(Path(result["experiment_manifest"]).read_text())
    assert summary["tracking"]["status"] == "enabled"
    assert sys.modules.get("mlflow") is parent_mlflow_module
    tracking_files = [entry for entry in completed["artifacts"] if entry["kind"] == "tracking"]
    assert any(entry["path"] == "tracking/mlflow.db" for entry in tracking_files)
    assert any(entry["path"].startswith("tracking/artifacts/") for entry in tracking_files)
    with sqlite3.connect(root / "tracking/mlflow.db", timeout=0) as connection:
        # Exclusive lock proves no live child-owned SQLite transaction remains.
        connection.execute("BEGIN EXCLUSIVE")
        uris = connection.execute(
            "SELECT artifact_location FROM experiments WHERE experiment_id != 0"
        ).fetchall()
        connection.rollback()
    assert all(str(root / "tracking/artifacts") in uri for (uri,) in uris)
    read_completed_run(root)


@pytest.mark.parametrize("availability", ["missing", "failed", "timeout"])
def test_optional_tracking_failure_is_safe_and_does_not_require_framework_import(
    synthetic_config: ProjectConfig,
    monkeypatch: pytest.MonkeyPatch,
    availability: str,
) -> None:
    from retail_forecast import tracking_runtime

    run = allocate_run(synthetic_config, kind="baseline")
    monkeypatch.setattr(
        tracking_runtime.importlib.util,
        "find_spec",
        lambda name: None if availability == "missing" else object(),
    )

    def failed_launch(*args: object, **kwargs: object) -> None:
        if availability == "timeout":
            raise tracking_runtime.subprocess.TimeoutExpired(
                "internal-tracking-worker", 120, output="SECRET_SENTINEL"
            )
        raise OSError("SECRET_SENTINEL /private/absolute/root")

    monkeypatch.setattr(tracking_runtime.subprocess, "run", failed_launch)
    status = tracking_runtime.log_baseline_tracking(
        run,
        enabled=True,
        model_ids=MODEL_IDS,
        tree_model_id="lightgbm_global",
        config=synthetic_config,
        folds=2,
        results=pd.DataFrame(),
        costs=pd.DataFrame(),
        paths=[],
        models=[],
        tree_parameters={},
        versions={},
    )
    assert status == {
        "status": "disabled",
        "reason": "not_installed" if availability == "missing" else "tracking_failed",
    }
    assert "SECRET_SENTINEL" not in json.dumps(run.manifest)
    assert "SECRET_SENTINEL" not in json.dumps(status)


_CONCURRENT_BASELINE = r"""
import json
import sys
from dataclasses import replace
from pathlib import Path
from retail_forecast.config import PathsConfig, load_config
from retail_forecast.pipeline import baseline_experiment
request = json.load(sys.stdin)
base = load_config('configs/smoke.yaml')
config = replace(base, project_root=Path(request['root']),
                 paths=PathsConfig(Path(request['raw']), Path(request['artifacts'])),
                 baseline=replace(base.baseline, tree_n_estimators=2))
print(json.dumps(baseline_experiment(config, tracking=False)))
"""


def test_two_baseline_processes_preserve_full_and_legacy_evidence(
    synthetic_config: ProjectConfig,
) -> None:
    import subprocess
    import sys
    from pathlib import Path

    from retail_forecast.provenance import sha256_file

    pytest.importorskip("lightgbm")
    pytest.importorskip("matplotlib")
    root = synthetic_config.paths.artifacts_dir.parent
    sentinels = [
        root / "artifacts/full/previous-run/metrics.json",
        root / "artifacts/metrics/baseline_results.csv",
        root / "artifacts/figures/baselines/legacy.png",
        root / "mlruns/legacy.txt",
    ]
    for path in sentinels:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unchanged full or historical evidence")
    before = {path: sha256_file(path) for path in sentinels}
    request = json.dumps(
        {
            "root": str(root),
            "raw": str(synthetic_config.paths.raw_data_dir),
            "artifacts": str(root / "artifacts/smoke"),
        }
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", _CONCURRENT_BASELINE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    results = []
    try:
        for process in processes:
            assert process.stdin is not None
            process.stdin.write(request)
            process.stdin.close()
            process.stdin = None
        for process in processes:
            output, errors = process.communicate(timeout=90)
            assert process.returncode == 0, errors[-2000:]
            result = json.loads(output.splitlines()[-1])
            read_completed_run(Path(result["run_dir"]))
            results.append(result)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    assert results[0]["run_id"] != results[1]["run_id"]
    assert {path: sha256_file(path) for path in sentinels} == before
    first = pd.read_csv(results[0]["oof"])
    second = pd.read_csv(results[1]["oof"])
    pd.testing.assert_frame_equal(first, second)


def test_baseline_refuses_uncaptured_or_changed_run_configuration(
    synthetic_config: ProjectConfig,
) -> None:
    run = allocate_run(synthetic_config, kind="baseline")
    with pytest.raises(ValueError, match="captured input identity"):
        run_baseline_experiments(synthetic_config, run=run, tracking=False)
    capture_inputs(run)
    changed = replace(
        synthetic_config,
        baseline=replace(synthetic_config.baseline, tree_n_estimators=1),
    )
    with pytest.raises(ValueError, match="config differs"):
        run_baseline_experiments(changed, run=run, tracking=False)


def test_baseline_uses_captured_disk_input_and_rejects_caller_frames(
    synthetic_config: ProjectConfig,
) -> None:
    import shutil

    from retail_forecast.data import load_frames

    pytest.importorskip("lightgbm")
    pytest.importorskip("matplotlib")
    config = replace(
        synthetic_config,
        baseline=replace(synthetic_config.baseline, tree_n_estimators=2),
    )
    run = allocate_run(config, kind="baseline")
    capture_inputs(run)
    modified = load_frames(config.paths.raw_data_dir)
    modified["train.csv"]["sales"] = 100_000_000.0
    modified["oil.csv"]["dcoilwtico"] = 100_000_000.0
    with pytest.raises(TypeError, match="frames"):
        run_baseline_experiments(config, frames=modified, run=run, tracking=False)

    alternate = config.paths.raw_data_dir.parent / "unbound-input"
    shutil.copytree(config.paths.raw_data_dir, alternate)
    modified["train.csv"].to_csv(alternate / "train.csv", index=False)
    modified["oil.csv"].to_csv(alternate / "oil.csv", index=False)
    alternate_config = replace(config, paths=replace(config.paths, raw_data_dir=alternate))
    result = run_baseline_experiments(alternate_config, run=run, tracking=False)
    original = pd.read_csv(config.paths.raw_data_dir / "train.csv").set_index("id")["sales"]
    expected = result["oof"]["id"].map(original).to_numpy()
    np.testing.assert_array_equal(result["oof"]["y_true"].to_numpy(), expected)
    assert not result["oof"]["y_true"].eq(100_000_000.0).any()
    finish_run(run)
    read_completed_run(run.root)


def test_baseline_rejects_source_changed_since_capture(
    synthetic_config: ProjectConfig,
) -> None:
    pytest.importorskip("lightgbm")
    pytest.importorskip("matplotlib")
    run = allocate_run(synthetic_config, kind="baseline")
    capture_inputs(run)
    source = synthetic_config.paths.raw_data_dir / "train.csv"
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed"):
        run_baseline_experiments(synthetic_config, run=run, tracking=False)
    assert not any(item["kind"] == "native_model" for item in run.manifest["artifacts"])
