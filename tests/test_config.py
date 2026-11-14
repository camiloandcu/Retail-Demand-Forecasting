"""Configuration schema tests."""

from pathlib import Path

import pytest
import yaml

from retail_forecast.config import PROJECT_HORIZON, ConfigError, load_config


def test_smoke_and_full_share_the_same_schema() -> None:
    smoke = yaml.safe_load(Path("configs/smoke.yaml").read_text())
    full = yaml.safe_load(Path("configs/full.yaml").read_text())
    assert smoke.keys() == full.keys()
    for section in smoke:
        assert smoke[section].keys() == full[section].keys()
    assert load_config("configs/smoke.yaml").forecast.horizon == PROJECT_HORIZON
    assert load_config("configs/full.yaml").forecast.horizon == PROJECT_HORIZON


def test_unknown_configuration_key_is_rejected(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/smoke.yaml").read_text())
    raw["forecast"]["unknown"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="Invalid keys"):
        load_config(path)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("seasonal_average_weeks", 3, "four weeks"),
        ("training_origin_stride_days", 15, "same stride"),
        ("tree_l2", -1.0, "non-negative"),
    ],
)
def test_preregistered_baseline_contract_is_enforced(
    tmp_path: Path, key: str, value: object, message: str
) -> None:
    raw = yaml.safe_load(Path("configs/smoke.yaml").read_text())
    raw["baseline"][key] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match=message):
        load_config(path)


@pytest.mark.parametrize(
    ("name", "policy"),
    [
        ("smoke", "competition_v1"),
        ("full", "competition_v1"),
        ("asof", "asof_v1"),
        ("smoke-asof", "asof_v1"),
    ],
)
def test_every_config_declares_its_availability_policy(name: str, policy: str) -> None:
    assert load_config(f"configs/{name}.yaml").data.availability_policy == policy


@pytest.mark.parametrize("value", [None, "", "realtime_v9", "ASOF_V1"])
def test_missing_or_unknown_policy_is_rejected(tmp_path: Path, value: object) -> None:
    raw = yaml.safe_load(Path("configs/smoke-asof.yaml").read_text())
    if value is None:
        del raw["data"]["availability_policy"]
        message = "Invalid keys"
    else:
        raw["data"]["availability_policy"] = value
        message = "availability_policy"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match=message):
        load_config(path)
