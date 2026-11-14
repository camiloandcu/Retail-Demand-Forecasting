"""Execute the actual notebook consumers against isolated synthetic run stores."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from getpass import getpass
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "name",
    ["01_data_audit_eda.ipynb", "02_feature_pipeline.ipynb", "03_non_neural_baselines.ipynb"],
)
def test_notebook_consumers_publish_all_outputs_without_mutating_legacy(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("matplotlib")
    if name.startswith("03"):
        pytest.importorskip("lightgbm")
        pytest.importorskip("sklearn")
    from retail_forecast.artifacts import read_completed_run, write_json_artifact

    source_root = Path(__file__).resolve().parents[1]
    notebook = json.loads((source_root / name).read_text())
    (tmp_path / "configs").mkdir()
    shutil.copy(source_root / "configs/smoke.yaml", tmp_path / "configs/smoke.yaml")
    (tmp_path / "tests").mkdir()
    for test in ["conftest.py", "test_splits.py", "test_features.py"]:
        shutil.copy(source_root / "tests" / test, tmp_path / "tests" / test)
    legacy = tmp_path / "artifacts/metrics/historical.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"unchanged historical evidence")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RETAIL_FORECAST_MODE", "smoke")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace = {
        "REPO_ROOT": tmp_path,
        "os": os,
        "sys": sys,
        "subprocess": subprocess,
        "Path": Path,
        "getpass": getpass,
    }
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        code = "".join(cell["source"])
        if "# Immutable package setup" in code:
            continue  # The independent bootstrap suite verifies immutable installation.
        if name.startswith("03"):
            code = code.replace("tracking=True", "tracking=False")
        exec(compile(code, f"{name}:cell-{index}", "exec"), namespace)
    run = namespace["run_context"]
    manifest = read_completed_run(run.root, expected_mode="smoke", expected_source="synthetic")
    declared = {entry["path"] for entry in manifest["artifacts"]}
    produced = {
        path.relative_to(run.root).as_posix()
        for directory in ["metrics", "figures", "models", "predictions"]
        for path in (run.root / directory).rglob("*")
        if path.is_file()
    }
    assert produced <= declared
    assert manifest["availability_policy"] == "competition_v1"
    assert legacy.read_bytes() == b"unchanged historical evidence"
    with pytest.raises(ValueError):
        write_json_artifact(run, "metrics/late.json", {"late": True}, kind="metrics")
    if name.startswith("03"):
        assert "figures/baselines/02_horizon_rmsle_without_zero.png" in declared
