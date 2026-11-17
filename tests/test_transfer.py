"""Transfer boundaries, temporal selection and paired forecast evidence."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from retail_forecast.transfer import copy_encoder, split_stores, transfer_experiment


def test_encoder_transfer_keeps_target_embeddings_and_head():
    torch = pytest.importorskip("torch")
    from retail_forecast.models.gru import JointGRU

    source = JointGRU(8, [3] * 6)
    target = JointGRU(8, [5] * 6)
    initial = {k: v.clone() for k, v in target.state_dict().items()}
    copy_encoder(source, target)
    for key, value in target.state_dict().items():
        expected = source.state_dict()[key] if key.startswith("gru.") else initial[key]
        assert torch.equal(value, expected)
    assert all(p.requires_grad for p in target.parameters())


def test_store_partition_is_fixed_and_disjoint():
    series = tuple((store, "A") for store in range(1, 55))
    source, target = split_stores(series)
    assert {s for s, _ in source} == set(range(1, 37))
    assert {s for s, _ in target} == set(range(37, 55))
    assert not set(source) & set(target)


def test_failed_training_is_counted_and_never_completed(sequence_data, monkeypatch):
    pytest.importorskip("torch")
    from pathlib import Path

    from retail_forecast.models import gru_training

    def fail(*args, **kwargs):
        error = RuntimeError("GRU_TIME_BUDGET_EXCEEDED")
        error.training_resources = {"attempt_training_seconds": 1801.5}
        raise error

    monkeypatch.setattr(gru_training, "fit_epochs", fail)
    config, _, _, _ = sequence_data
    with pytest.raises(RuntimeError, match="TIME_BUDGET"):
        transfer_experiment(config)
    manifest_path = next(Path(config.paths.artifacts_dir).glob("transfer/*/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "incomplete"
    assert manifest["training_seconds"] == 1801.5
    assert not (manifest_path.parent / "results.csv").exists()


@pytest.mark.parametrize("minutes", [2.0, None])
def test_transfer_smoke_has_complete_paired_evidence(sequence_data, monkeypatch, minutes):
    pytest.importorskip("torch")
    from retail_forecast import transfer

    config, _, _, _ = sequence_data
    config = replace(config, recurrent=replace(config.recurrent, training_minutes=minutes))
    observed = []
    original = transfer._Study.train

    def track(self, panel, pre, **kwargs):
        observed.append((kwargs["name"], panel.cutoff, pre.state["fit_cutoff"]))
        assert all(panel.dates[o + 16] <= panel.cutoff for _, o in panel.keys)
        return original(self, panel, pre, **kwargs)

    monkeypatch.setattr(transfer._Study, "train", track)
    study = transfer_experiment(config)
    root = __import__("pathlib").Path(study["run_dir"])
    assert study["manifest"]["status"] == "completed"
    assert study["manifest"]["data_source"] == "synthetic"
    assert not json.loads((root / "feasibility.json").read_text())["outer_scores_computed"]
    for fold, origin in enumerate(config.validation.origins, 1):
        for name, cutoff, fit_cutoff in observed:
            if name.startswith(f"fold_{fold}/"):
                expected = pd.Timestamp(origin)
                if "/selection/" in name:
                    expected -= pd.Timedelta(days=16)
                assert cutoff == expected == pd.Timestamp(fit_cutoff)
    feasibility = json.loads((root / "feasibility.json").read_text())
    assert feasibility["budget_seconds"] == (None if minutes is None else minutes * 60)
    assert feasibility["passed"]
    predictions = pd.read_csv(root / "predictions.csv")
    assert np.isfinite(predictions.prediction).all()
    assert (predictions.prediction >= 0).all()
    results = pd.read_csv(root / "results.csv")
    assert len(results[results.scope.eq("global")]) == 2
    broken = predictions.drop(predictions.index[0])
    with pytest.raises(ValueError, match="share every evaluation key"):
        transfer.summarize_predictions(broken)
