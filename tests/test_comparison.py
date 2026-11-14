"""Complete-lineage comparator re-emission through verified bundles (AC-9)."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from retail_forecast.artifacts import read_completed_run
from retail_forecast.config import ProjectConfig, load_config
from retail_forecast.data_contracts import LineageIneligibleError, require_complete_lineage

pytestmark = pytest.mark.models


@pytest.fixture(params=["configs/smoke.yaml", "configs/smoke-asof.yaml"])
def compared(request, tmp_path: Path, monkeypatch):
    pytest.importorskip("lightgbm")
    pytest.importorskip("pyarrow")
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    from retail_forecast.config import PathsConfig
    from retail_forecast.pipeline import baseline_experiment, compare

    config: ProjectConfig = replace(
        load_config(request.param),
        paths=PathsConfig(raw_data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts"),
    )
    return compare(config), baseline_experiment(config, tracking=False)


def test_comparison_reproduces_the_research_baseline_scores(compared):
    result, baseline = compared
    new = pd.read_csv(result["results"])
    old = pd.read_csv(baseline["results"])
    keys = ["availability_policy", "model_id", "scope", "segment"]
    joined = new.merge(old, on=keys, suffixes=("_new", "_old"), how="outer", indicator=True)
    assert joined["_merge"].eq("both").all()
    assert (joined["rows_new"] == joined["rows_old"]).all()
    assert (joined["rmsle_new"] - joined["rmsle_old"]).abs().max() < 1e-12


def test_target_keys_match_the_research_oof(compared):
    result, baseline = compared
    columns = ["model_id", "fold", "store_nbr", "family", "target_date"]
    new = pd.read_parquet(result["oof"])[columns]
    old = pd.read_parquet(baseline["oof"])[columns]
    order = list(columns)
    pd.testing.assert_frame_equal(
        new.sort_values(order).reset_index(drop=True),
        old.sort_values(order).reset_index(drop=True),
        check_dtype=False,
    )


def test_serving_models_carry_complete_lineage_and_zero_stays_research_only(compared):
    result, _ = compared
    oof = pd.read_parquet(result["oof"])
    complete = oof.loc[oof["model_id"] != "zero"]
    require_complete_lineage(complete)
    assert set(complete["model_id"]) == {
        "seasonal_naive_7d",
        "seasonal_average_4w",
        "lightgbm_global",
    }
    assert complete.groupby(["model_id", "fold"])["bundle_id"].nunique().eq(1).all()
    with pytest.raises(LineageIneligibleError):
        require_complete_lineage(oof.loc[oof["model_id"] == "zero"])
    _, baseline = compared
    with pytest.raises(LineageIneligibleError):
        require_complete_lineage(pd.read_parquet(baseline["oof"]))


def test_fold_bundles_complete_before_their_separate_prediction_run(compared):
    result, _ = compared
    artifacts = Path(result["run_dir"]).parent
    training_id, prediction_id = result["child_run_ids"]
    training = read_completed_run(artifacts / training_id)
    prediction = read_completed_run(artifacts / prediction_id)
    comparison = read_completed_run(Path(result["run_dir"]))
    assert training["kind"] == "training" and prediction["kind"] == "prediction"
    assert prediction["parent_run_ids"] == [training_id]
    assert comparison["parent_run_ids"] == [training_id, prediction_id]
    assert datetime.fromisoformat(training["finished_at_utc"]) <= datetime.fromisoformat(
        prediction["started_at_utc"]
    )
    bundles = [e for e in training["artifacts"] if e["kind"] == "bundle_manifest"]
    assert len(bundles) == 2 * 3  # folds x serving models; none overwritten
    for entry in bundles:
        manifest = json.loads((artifacts / training_id / entry["path"]).read_text())
        assert manifest["evaluation_ref"] is None  # unpromoted
    oof = pd.read_parquet(result["oof"])
    assert oof["evaluation_run_id"].eq(result["run_id"]).all()
    assert oof.loc[oof["model_id"] != "zero", "prediction_run_id"].eq(prediction_id).all()
