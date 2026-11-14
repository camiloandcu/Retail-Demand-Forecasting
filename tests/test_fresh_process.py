"""Separate-process and clean-root prediction from verified bundles (AC-2, AC-8).

Each step runs in its own interpreter through the installed CLI. Prediction runs with
sockets disabled, an unreachable MLflow URI and an empty HOME, so success shows the
bundle needs no fitting, network, tracking server or home model cache.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

BLOCKED = """
import json, socket, sys
def _denied(*args, **kwargs):
    raise OSError("network access denied by test")
socket.socket.connect = _denied
socket.socket.connect_ex = _denied
socket.create_connection = _denied
socket.getaddrinfo = _denied
from retail_forecast import pipeline
from retail_forecast.models import lightgbm as lgbm_adapter
from retail_forecast.models import seasonal as seasonal_adapter
def _no_fit(*args, **kwargs):
    raise AssertionError("prediction attempted to fit")
lgbm_adapter.LightGBMAdapter.fit = _no_fit
lgbm_adapter.fit_booster = _no_fit
seasonal_adapter.SeasonalNaiveAdapter.fit = _no_fit
seasonal_adapter.SeasonalAverageAdapter.fit = _no_fit
pipeline.build_training_set = _no_fit
from retail_forecast.cli import main
code = main(sys.argv[1:])
loaded = sorted(m for m in ("mlflow", "torch", "chronos") if m in sys.modules)
print("@@MODULES@@" + json.dumps(loaded))
sys.exit(code)
"""


def _config(tmp_path: Path, *, policy: str = "competition_v1") -> Path:
    base = yaml.safe_load(Path("configs/smoke.yaml").read_text())
    base["paths"] = {
        "raw_data_dir": str(tmp_path / "data"),
        "artifacts_dir": str(tmp_path / "artifacts"),
    }
    base["data"]["availability_policy"] = policy
    path = tmp_path / "configs" / "fresh.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(base))
    return path


def _cli(*args: str, blocked: bool = False, home: Path | None = None) -> dict:
    environment = {**os.environ, "MLFLOW_TRACKING_URI": "http://127.0.0.1:9", "MPLBACKEND": "Agg"}
    if home is not None:
        environment["HOME"] = str(home)
        environment.pop("XDG_CACHE_HOME", None)
    command = (
        [sys.executable, "-c", BLOCKED, *args]
        if blocked
        else [sys.executable, "-m", "retail_forecast.cli", *args]
    )
    completed = subprocess.run(command, capture_output=True, text=True, env=environment)
    assert completed.returncode == 0, completed.stderr[-3000:]
    text = completed.stdout
    modules = None
    if "@@MODULES@@" in text:
        text, marker = text.rsplit("@@MODULES@@", 1)
        modules = json.loads(marker)
    result = json.loads(text)
    if modules is not None:
        result["_modules"] = modules
    return result


def _fit_predictions(trained: dict, model_id: str) -> pd.DataFrame:
    path = Path(trained["run_dir"]) / f"checks/{model_id}-final-fit-predictions.csv"
    frame = pd.read_csv(path, dtype={"family": str})
    frame["prediction"] = np.maximum(frame["prediction"].to_numpy(dtype=np.float64), 0.0)
    return frame.sort_values(["store_nbr", "family", "target_date"]).reset_index(drop=True)


def _forecast(result: dict) -> pd.DataFrame:
    frame = pd.read_csv(result["forecast"], dtype={"family": str})
    return frame.sort_values(["store_nbr", "family", "target_date"]).reset_index(drop=True)


def _assert_parity(fit: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    key = ["store_nbr", "family", "target_date"]
    assert fit[key].equals(forecast[key])  # exact keys before any numeric tolerance
    np.testing.assert_allclose(forecast["prediction"], fit["prediction"], rtol=1e-10, atol=1e-8)
    return {"max_abs": float(np.abs(forecast["prediction"] - fit["prediction"]).max())}


@pytest.fixture
def prepared(tmp_path: Path):
    config = _config(tmp_path)
    _cli("fixture", "--config", str(config))
    return config


@pytest.mark.parametrize(
    "model_id",
    [
        pytest.param("lightgbm_global", marks=pytest.mark.models),
        "seasonal_naive_7d",
        "seasonal_average_4w",
    ],
)
def test_separate_process_prediction_reproduces_the_fit(prepared, tmp_path, model_id):
    if model_id == "lightgbm_global":
        pytest.importorskip("lightgbm")
    trained = _cli("train", "--config", str(prepared), "--model-id", model_id)
    request_dir = Path(_cli("request", "--config", str(prepared))["request_dir"])
    # Remove every competition file, including all labels, before the fresh process.
    shutil.rmtree(tmp_path / "data")
    predicted = _cli(
        "predict",
        "--config",
        str(prepared),
        "--bundle",
        trained["bundle"],
        "--request",
        str(request_dir),
        blocked=True,
        home=tmp_path / "empty-home",
    )
    assert predicted["_modules"] == []
    assert predicted["bundle_id"] == trained["bundle_id"]
    _assert_parity(_fit_predictions(trained, model_id), _forecast(predicted))


@pytest.mark.models
def test_capsule_in_clean_root_predicts_without_the_source_run(prepared, tmp_path):
    pytest.importorskip("lightgbm")
    trained = _cli("train", "--config", str(prepared), "--model-id", "lightgbm_global")
    request_dir = Path(_cli("request", "--config", str(prepared))["request_dir"])
    reference = _cli(
        "predict",
        "--config",
        str(prepared),
        "--bundle",
        trained["bundle"],
        "--request",
        str(request_dir),
    )
    exported = _cli(
        "export-capsule",
        "--config",
        str(prepared),
        "--bundle",
        trained["bundle"],
        "--destination",
        str(tmp_path / "exports"),
    )
    capsule = Path(exported["capsule_dir"])
    contents = sorted(p.relative_to(capsule).as_posix() for p in capsule.rglob("*") if p.is_file())
    assert contents == sorted(
        [
            "CAPSULE_COMPLETE.json",
            "capsule.json",
            "bundle/bundle.json",
            "bundle/preprocessing.json",
            "bundle/signature.json",
            "bundle/state/booster.txt",
            "provenance/source-complete.json",
            "provenance/source-run.json",
        ]
    )
    expected = _forecast(reference)
    original_bundle = Path(trained["bundle_dir"]) / "bundle.json"
    assert (capsule / "bundle/bundle.json").read_bytes() == original_bundle.read_bytes()

    clean = tmp_path / "clean-root"
    clean.mkdir()
    shutil.copytree(capsule, clean / capsule.name)
    shutil.copytree(request_dir, clean / "request")
    shutil.rmtree(tmp_path / "artifacts")  # no original run tree, labels or tracking state
    shutil.rmtree(tmp_path / "exports")
    shutil.rmtree(tmp_path / "data")
    predicted = _cli(
        "predict",
        "--config",
        str(prepared),
        "--bundle",
        f"capsule:{clean / capsule.name}",
        "--request",
        str(clean / "request"),
        blocked=True,
        home=tmp_path / "empty-home",
    )
    assert predicted["bundle_id"] == trained["bundle_id"] == exported["bundle_id"]
    actual = _forecast(predicted)
    columns = ["store_nbr", "family", "target_date", "prediction", "bundle_id", "training_run_id"]
    pd.testing.assert_frame_equal(actual[columns], expected[columns])


@pytest.fixture
def capsule(synthetic_config, tmp_path):
    from retail_forecast.capsules import export_inference_capsule
    from retail_forecast.pipeline import train

    trained = train(synthetic_config, model_id="seasonal_naive_7d")
    relative = trained["bundle"].split(":", 1)[1]
    manifest = export_inference_capsule(
        Path(trained["run_dir"]), bundle_relative_path=relative, destination_root=tmp_path / "c"
    )
    return tmp_path / "c" / manifest["capsule_id"], trained, relative


@pytest.mark.parametrize(
    "damage",
    ["witness", "asset", "extra_file", "marker", "renamed"],
)
def test_damaged_capsules_fail_before_native_load(capsule, damage, tmp_path):
    from retail_forecast.bundles import BundleError, BundleRef, load_bundle
    from retail_forecast.capsules import verify_inference_capsule

    root, _, _ = capsule
    if damage == "witness":
        path = root / "provenance/source-complete.json"
        payload = json.loads(path.read_text())
        payload["completed_at_utc"] = "2000-01-01T00:00:00+00:00"
        path.write_text(json.dumps(payload))
    elif damage == "asset":
        (root / "bundle/state/seasonal.json").unlink()
    elif damage == "extra_file":
        (root / "bundle/state/extra.json").write_text("{}")
    elif damage == "marker":
        (root / "CAPSULE_COMPLETE.json").unlink()
    else:
        root = root.rename(root.with_name("0" * 64))
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_inference_capsule(root)
    with pytest.raises(BundleError):
        load_bundle(BundleRef("capsule", root, "bundle"), expected_policy="competition_v1")


def test_capsule_identity_is_never_overwritten(capsule, monkeypatch):
    from datetime import datetime, timezone

    import retail_forecast.capsules as capsules

    root, trained, relative = capsule

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(capsules, "datetime", Frozen)
    destination = root.parent.parent / "frozen"
    first = capsules.export_inference_capsule(
        Path(trained["run_dir"]), bundle_relative_path=relative, destination_root=destination
    )
    before = {
        p: p.read_bytes() for p in (destination / first["capsule_id"]).rglob("*") if p.is_file()
    }
    with pytest.raises(ValueError, match="never overwritten"):
        capsules.export_inference_capsule(
            Path(trained["run_dir"]), bundle_relative_path=relative, destination_root=destination
        )
    assert {p: p.read_bytes() for p in before} == before
    assert [p.name for p in destination.iterdir()] == [first["capsule_id"]]
