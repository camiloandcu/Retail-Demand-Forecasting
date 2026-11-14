"""CLI selection and explicit tracking controls."""

import json

import pytest

from retail_forecast import cli


def test_predict_selection_is_required_before_computation() -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["predict", "--config", "configs/smoke.yaml"])
    assert error.value.code == 2


def test_baseline_passes_explicit_options(monkeypatch, capsys) -> None:
    seen = {}

    def fake_baseline(config, **kwargs):
        seen.update(kwargs)
        return {"run_id": "example"}

    monkeypatch.setattr(cli, "baseline_experiment", fake_baseline)
    assert cli.main(["baseline", "--config", "configs/smoke.yaml", "--no-tracking"]) == 0
    assert seen == {"run_id": None, "resume_from": None, "tracking": False}
    assert json.loads(capsys.readouterr().out) == {"run_id": "example"}
