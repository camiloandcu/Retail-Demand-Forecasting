"""Automatic hardware selection must preserve full-data training recipes."""

from dataclasses import replace

import pytest

from retail_forecast.config import load_config
from retail_forecast.neural_runtime import resolve_device, training_time_limit
from retail_forecast.recurrent_training import _recipe
from retail_forecast.transfer import transfer_recipe


@pytest.mark.parametrize("available, expected", [(False, "cpu"), (True, "cuda")])
def test_full_recipes_select_available_device(monkeypatch, available, expected):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    config = load_config("configs/gru.yaml")
    assert config.mode == "full" and config.data.source == "kaggle"
    assert config.recurrent.device == "auto"
    assert config.recurrent.training_minutes is None
    primary, transfer = _recipe(config), transfer_recipe(config)
    assert primary["device"] == transfer["device"] == expected
    assert primary["hidden_size"] == 64 and primary["max_epochs"] == 40
    assert transfer["hidden_size"] == 32 and transfer["max_epochs"] == 8
    assert transfer["max_training_origins"] == 8
    assert training_time_limit(primary) is training_time_limit(transfer) is None
    explicit = replace(config, recurrent=replace(config.recurrent, device="cpu"))
    assert _recipe(explicit)["device"] == transfer_recipe(explicit)["device"] == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_time_limit_remains_available_for_explicit_bounded_runs():
    assert training_time_limit({"training_minutes": 2}) == 120
    with pytest.raises(ValueError, match="device"):
        resolve_device("tpu")
