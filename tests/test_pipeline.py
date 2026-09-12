"""End-to-end smoke workflow tests."""

import json

import pandas as pd

from retail_forecast.config import ProjectConfig
from retail_forecast.pipeline import smoke


def test_smoke_runs_batch_inference_without_external_data(
    synthetic_config: ProjectConfig,
) -> None:
    result = smoke(synthetic_config)
    submission = pd.read_csv(result["submission"])
    evaluation = json.loads(
        synthetic_config.paths.artifacts_dir.joinpath("evaluation.json").read_text()
    )

    assert len(submission) == 2 * 2 * 16
    assert list(submission.columns) == ["id", "sales"]
    assert (submission["sales"] >= 0).all()
    assert evaluation["smoke"] is True
    assert len(evaluation["folds"]) == 1
