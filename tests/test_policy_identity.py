"""Run/snapshot policy identity and refusal to pool availability regimes."""

from dataclasses import replace

import pandas as pd
import pytest

from retail_forecast.artifact_schema import validate_manifest
from retail_forecast.artifacts import allocate_run, capture_inputs
from retail_forecast.availability import PolicyMismatchError, require_same_policy
from retail_forecast.config import ProjectConfig
from retail_forecast.experiments import segmented_rmsle


def test_runs_and_snapshots_record_the_configured_policy(
    synthetic_config: ProjectConfig,
) -> None:
    snapshots = {}
    for policy in ("competition_v1", "asof_v1"):
        config = replace(
            synthetic_config, data=replace(synthetic_config.data, availability_policy=policy)
        )
        run = allocate_run(config, kind="baseline")
        snapshot = capture_inputs(run)
        assert run.manifest["availability_policy"] == policy
        assert snapshot["availability_policy"] == policy
        snapshots[policy] = snapshot
    assert snapshots["asof_v1"]["files"] == snapshots["competition_v1"]["files"]
    assert snapshots["asof_v1"]["snapshot_id"] != snapshots["competition_v1"]["snapshot_id"]


def test_schema_rejects_unregistered_policy(synthetic_config: ProjectConfig) -> None:
    run = allocate_run(synthetic_config, kind="baseline")
    validate_manifest(run.manifest)
    with pytest.raises(ValueError, match="availability policy"):
        validate_manifest({**run.manifest, "availability_policy": "asof_v2"})


def test_mixed_policy_oof_cannot_be_pooled() -> None:
    row = {
        "model_id": "zero",
        "fold": "fold_1",
        "horizon": 1,
        "store_nbr": 1,
        "family": "A",
        "promotion_regime": "not_promoted",
        "temporal_regime": "regular_weekday",
        "y_true": 1.0,
        "y_pred": 1.0,
    }
    mixed = pd.DataFrame(
        [
            {**row, "availability_policy": "competition_v1"},
            {**row, "availability_policy": "asof_v1"},
        ]
    )
    with pytest.raises(PolicyMismatchError, match="POLICY_MISMATCH"):
        segmented_rmsle(mixed)
    with pytest.raises(PolicyMismatchError):
        require_same_policy(["competition_v1", "asof_v1"], context="comparison")
    assert require_same_policy(["asof_v1"], context="comparison") == "asof_v1"
