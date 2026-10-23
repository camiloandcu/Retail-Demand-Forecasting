"""Run-scoped seasonal-naive integration and denied-consumer checks."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from retail_forecast.artifacts import read_completed_run
from retail_forecast.config import ProjectConfig
from retail_forecast.pipeline import evaluate, predict, smoke, train


def test_smoke_runs_batch_inference_without_external_data(
    synthetic_config: ProjectConfig,
) -> None:
    result = smoke(synthetic_config)
    submission = pd.read_csv(result["submission"])
    evaluation = json.loads(Path(result["evaluation"]).read_text())
    manifest = read_completed_run(Path(result["run_dir"]))
    assert len(submission) == 2 * 2 * 16
    assert list(submission.columns) == ["id", "sales"]
    assert (submission["sales"] >= 0).all()
    assert evaluation["smoke"] is True
    assert len(evaluation["folds"]) == 2
    assert manifest["kind"] == "smoke"
    assert manifest["availability_policy"] == "competition_v1"


def test_concurrent_smoke_preserves_shared_fixture_and_legacy_outputs(
    synthetic_config: ProjectConfig,
) -> None:
    root = synthetic_config.paths.artifacts_dir
    root.mkdir(parents=True)
    sentinel = root / "model.json"
    sentinel.write_bytes(b"legacy-full-model")
    source = synthetic_config.paths.raw_data_dir / "train.csv"
    source_bytes = source.read_bytes()
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(lambda _: smoke(synthetic_config), range(2)))
    assert results[0]["run_id"] != results[1]["run_id"]
    assert sentinel.read_bytes() == b"legacy-full-model"
    assert source.read_bytes() == source_bytes
    pd.testing.assert_frame_equal(
        pd.read_csv(results[0]["submission"]), pd.read_csv(results[1]["submission"])
    )
    for result in results:
        manifest = read_completed_run(Path(result["run_dir"]))
        assert Path(result["submission"]).is_relative_to(Path(result["run_dir"]))
        assert manifest["state"] == "SUCCEEDED"


def test_prediction_requires_explicit_verified_parent(synthetic_config: ProjectConfig) -> None:
    with pytest.raises(ValueError, match="TRAINING_RUN_REQUIRED"):
        predict(synthetic_config)
    trained = train(synthetic_config)
    result = predict(synthetic_config, training_run=trained["run_id"])
    manifest = read_completed_run(Path(result["run_dir"]))
    assert manifest["parent_run_ids"] == [trained["run_id"]]
    assert len(pd.read_csv(result["submission"])) == 64


def test_smoke_model_is_an_explicit_supported_parent(synthetic_config: ProjectConfig) -> None:
    original = smoke(synthetic_config)
    result = predict(synthetic_config, training_run=original["run_id"])
    pd.testing.assert_frame_equal(
        pd.read_csv(original["submission"]), pd.read_csv(result["submission"])
    )


def test_prediction_rejects_nontraining_corrupt_and_changed_input_parents(
    synthetic_config: ProjectConfig,
) -> None:
    evaluated = evaluate(synthetic_config)
    with pytest.raises(ValueError, match="TRAINING_RUN_KIND_INVALID"):
        predict(synthetic_config, training_run=evaluated["run_id"])
    trained = train(synthetic_config)
    model_path = Path(trained["model"])
    model_path.write_bytes(model_path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        predict(synthetic_config, training_run=trained["run_id"])
    trained = train(synthetic_config)
    input_path = synthetic_config.paths.raw_data_dir / "train.csv"
    frame = pd.read_csv(input_path)
    frame.loc[0, "sales"] += 1
    frame.to_csv(input_path, index=False)
    with pytest.raises(ValueError, match="TRAINING_RUN_IDENTITY_MISMATCH"):
        predict(synthetic_config, training_run=trained["run_id"])


def test_prediction_rejects_config_changes_and_incomplete_parent(
    synthetic_config: ProjectConfig,
) -> None:
    trained = train(synthetic_config)
    changed = replace(
        synthetic_config,
        runtime=replace(synthetic_config.runtime, seed=synthetic_config.runtime.seed + 1),
    )
    with pytest.raises(ValueError, match="TRAINING_RUN_IDENTITY_MISMATCH"):
        predict(changed, training_run=trained["run_id"])
    Path(trained["run_dir"], "COMPLETE.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        predict(synthetic_config, training_run=trained["run_id"])


def test_cli_result_verifier_rejects_corruption(
    synthetic_config: ProjectConfig, tmp_path: Path
) -> None:
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("verify_run_result", "scripts/verify_run_result.py")
    assert spec is not None and spec.loader is not None
    verifier = module_from_spec(spec)
    spec.loader.exec_module(verifier)
    result = smoke(synthetic_config)
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result))
    assert verifier.verify_result(result_path, mode="smoke", source="synthetic") == Path(
        result["submission"]
    )
    with pytest.raises(ValueError):
        verifier.verify_result(result_path, mode="full", source="kaggle")
    Path(result["submission"]).write_bytes(b"id,sales\n")
    with pytest.raises(ValueError):
        verifier.verify_result(result_path, mode="smoke", source="synthetic")


def test_resume_is_explicit_and_unsupported_native_state_is_not_retrained(
    synthetic_config: ProjectConfig,
) -> None:
    trained = train(synthetic_config)
    before = {p: p.read_bytes() for p in Path(trained["run_dir"]).rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="RESUME_UNSUPPORTED"):
        train(synthetic_config, resume_from=trained["run_id"])
    assert all(p.read_bytes() == content for p, content in before.items())
