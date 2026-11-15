"""Real optimization, safe state, interrupted recovery and process-isolated inference."""

import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("safetensors")
import torch
from test_sequences import panel_from

from retail_forecast.artifacts import read_completed_run
from retail_forecast.bundles import load_bundle, parse_bundle_ref
from retail_forecast.capsules import export_inference_capsule
from retail_forecast.config import PathsConfig, load_config
from retail_forecast.data import write_synthetic_dataset
from retail_forecast.models.gru import initialize, inverse_sales
from retail_forecast.models.gru_training import (
    fit_epochs,
    original_log_loss,
    pack_checkpoint,
    unpack_checkpoint,
)
from retail_forecast.pipeline import predict, request, train
from retail_forecast.recurrent import recurrent_experiment
from retail_forecast.sequences import SequencePreprocessor

pytestmark = pytest.mark.neural


def test_original_log_loss_and_inverse():
    # Equal original-log residuals have equal loss, despite unequal saved scales.
    scales = torch.tensor([[1.0], [10.0]])
    predicted = torch.tensor([[2.0] * 16, [0.2] * 16])
    assert (
        original_log_loss(predicted, torch.zeros_like(predicted), torch.zeros(2, 1), scales).item()
        == 4
    )
    values, clipped = inverse_sales(
        np.array([[-1, 0, 1]], dtype=np.float32), np.array([[0]]), np.array([[1]])
    )
    assert clipped == 1 and values[0, 0] == 0
    with pytest.raises(ValueError, match="Nonfinite"):
        inverse_sales(np.array([[10000]], dtype=np.float32), np.array([[0]]), np.array([[1]]))


def test_epoch_recovery_is_exact_and_rejects_identity_changes(sequence_data):
    config, *_ = sequence_data
    panel = panel_from(sequence_data)
    pre = SequencePreprocessor.fit(panel, cutoff=panel.cutoff)
    params = asdict(config.recurrent)
    identity = {"config": "a", "data": "b", "scaler": pre.sha256, "precision": "float32"}
    uninterrupted = initialize(pre, 8, 2026)
    fit_epochs(uninterrupted, panel, pre, params=params, seed=2026, epochs=3, identity=identity)
    interrupted = initialize(pre, 8, 2026)
    saved = []

    def crash(epoch, payload):
        saved.append(payload)
        if epoch == 1:
            raise InterruptedError("synthetic epoch boundary")

    with pytest.raises(InterruptedError):
        fit_epochs(
            interrupted,
            panel,
            pre,
            params=params,
            seed=2026,
            epochs=3,
            identity=identity,
            on_epoch=crash,
        )
    assert unpack_checkpoint(saved[0])["epoch"] == 1
    resumed = initialize(pre, 8, 9999)
    result = fit_epochs(
        resumed, panel, pre, params=params, seed=2026, epochs=3, identity=identity, resume=saved[0]
    )
    assert [r["epoch"] for r in result["history"]] == [1, 2, 3]
    for key, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    for field in identity:
        changed = {**identity, field: "changed"}
        with pytest.raises(ValueError, match="CHECKPOINT_INCOMPATIBLE"):
            fit_epochs(
                initialize(pre, 8, 0),
                panel,
                pre,
                params=params,
                seed=2026,
                epochs=3,
                identity=changed,
                resume=saved[0],
            )


def test_fresh_process_capsule_and_complete_study(tmp_path):
    config = load_config("configs/gru-smoke.yaml")
    config = replace(config, paths=PathsConfig(tmp_path / "data", tmp_path / "runs"))
    write_synthetic_dataset(config)
    fitted = train(config, model_id="gru_global")
    ref = parse_bundle_ref(fitted["bundle"], artifacts_dir=config.paths.artifacts_dir)
    loaded = load_bundle(ref, expected_policy="asof_v1")
    original_checksum = loaded.fitted.preprocessing["preprocessor_sha256"]
    requested = request(config)
    expected = predict(config, bundle=fitted["bundle"], request_dir=Path(requested["request_dir"]))
    capsule = export_inference_capsule(
        ref.root, bundle_relative_path=ref.relative, destination_root=tmp_path / "capsules"
    )
    capsule_root = tmp_path / "capsules" / capsule["capsule_id"]
    import shutil

    shutil.rmtree(config.paths.raw_data_dir)
    shutil.rmtree(config.paths.artifacts_dir / fitted["run_id"])
    program = """
import json,socket,sys
from pathlib import Path
from retail_forecast.bundles import BundleRef,load_bundle
from retail_forecast.inference import predict_bundle
from retail_forecast.models.base import PredictionContext
from retail_forecast.request_store import read_request_dir
from retail_forecast.models.gru import GRUAdapter
GRUAdapter.fit=lambda *a,**k: (_ for _ in ()).throw(AssertionError('fit called'))
socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(AssertionError('network called'))
loaded=load_bundle(BundleRef('capsule',Path(sys.argv[1]),'bundle'),expected_policy='asof_v1')
req=read_request_dir(Path(sys.argv[2]))
frame,diagnostics=predict_bundle(loaded,req,context=PredictionContext('fresh-process',None,'synthetic','smoke'))
frame.to_csv(sys.argv[3],index=False,float_format='%.17g')
print(json.dumps({'checksum':loaded.fitted.preprocessing['preprocessor_sha256'],'diagnostics':diagnostics}))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(capsule_root),
            requested["request_dir"],
            str(tmp_path / "fresh.csv"),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "MLFLOW_TRACKING_URI": "http://127.0.0.1:1"},
        timeout=180,
    )
    assert json.loads(result.stdout)["checksum"] == original_checksum
    import pandas as pd

    a = pd.read_csv(expected["forecast"])
    b = pd.read_csv(tmp_path / "fresh.csv")
    key = ["store_nbr", "family", "target_date"]
    assert a[key].equals(b[key])
    np.testing.assert_allclose(a.prediction, b.prediction, rtol=1e-5, atol=1e-4)
    # This executes the complete multi-fold synthetic producer/consumer lifecycle.
    study = recurrent_experiment(config)
    manifest = read_completed_run(Path(study["run_dir"]))
    assert manifest["kind"] == "recurrent-study"
    summary = json.loads(Path(study["summary"]).read_text())
    assert (
        summary["oof_rows"]
        == summary["expected_rows"]
        == config.data.expected_stores * config.data.expected_families * 48
    )
    assert summary["empirical_results_are_synthetic"] is True
    assert len(summary["folds"]) == 3


def test_linked_resume_preserves_source_and_rejects_changed_recipe(tmp_path, monkeypatch):
    import retail_forecast.recurrent_training as rt

    config = load_config("configs/gru-smoke.yaml")
    config = replace(config, paths=PathsConfig(tmp_path / "data", tmp_path / "runs"))
    original = rt.write_run_artifact

    def interrupted(run, relative, writer, **kwargs):
        result = original(run, relative, writer, **kwargs)
        if str(relative).endswith("selection/epoch_0001.zip"):
            raise InterruptedError("intentional checkpoint boundary")
        return result

    with monkeypatch.context() as context:
        context.setattr(rt, "write_run_artifact", interrupted)
        with pytest.raises(InterruptedError):
            train(config, model_id="gru_global", run_id="interrupted")
    source = config.paths.artifacts_dir / "interrupted" / "run.json"
    before = source.read_bytes()
    changed = replace(config, recurrent=replace(config.recurrent, hidden_size=9))
    with pytest.raises(ValueError, match="incompatible"):
        train(changed, model_id="gru_global", resume_from="interrupted")
    resumed = train(config, model_id="gru_global", resume_from="interrupted")
    manifest = read_completed_run(Path(resumed["run_dir"]))
    assert manifest["resumed_from_run_id"] == "interrupted"
    assert source.read_bytes() == before
    assert resumed["stages"][0]["history"][0]["epoch"] == 1
    assert len(resumed["stages"][0]["history"]) >= 2


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("training_seconds", 6000.0, "TIME_BUDGET"),
        ("host_peak_bytes", 11 * 2**30, "HOST_MEMORY"),
        ("gpu_peak_bytes", 6 * 2**30, "GPU_MEMORY"),
    ],
)
def test_final_epoch_checkpoint_cannot_bypass_caps(sequence_data, field, value, error):
    config, *_ = sequence_data
    panel = panel_from(sequence_data)
    pre = SequencePreprocessor.fit(panel, cutoff=panel.cutoff)
    model = initialize(pre, 8, 2026)
    saved = []
    fit_epochs(
        model,
        panel,
        pre,
        params=asdict(config.recurrent),
        seed=2026,
        epochs=1,
        identity={"test": "cap"},
        on_epoch=lambda epoch, payload: saved.append(payload),
    )
    state = unpack_checkpoint(saved[-1])
    state[field] = value
    with pytest.raises(RuntimeError, match=error):
        fit_epochs(
            initialize(pre, 8, 2026),
            panel,
            pre,
            params=asdict(config.recurrent),
            seed=2026,
            epochs=1,
            identity={"test": "cap"},
            resume=pack_checkpoint(state),
        )
