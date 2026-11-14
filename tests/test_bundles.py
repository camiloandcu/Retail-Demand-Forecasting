"""Bundle completeness, verification-before-load, prediction contract and export (WI-D2-02)."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retail_forecast.artifacts import read_completed_run
from retail_forecast.bundles import (
    BundleError,
    BundleRef,
    load_bundle,
    validate_bundle_manifest,
)
from retail_forecast.config import ProjectConfig
from retail_forecast.data_contracts import (
    ForecastRequestError,
    LineageIneligibleError,
    require_complete_lineage,
)
from retail_forecast.inference import predict_bundle
from retail_forecast.io_utils import canonical_json
from retail_forecast.kaggle_export import ExportError, export_kaggle_submission
from retail_forecast.models.base import PredictionContext
from retail_forecast.pipeline import predict, request, smoke, train
from retail_forecast.provenance import digest_json
from retail_forecast.request_store import read_request_dir

LIGHTGBM = "lightgbm_global"
SEASONAL = "seasonal_naive_7d"


def _ref(trained: dict) -> BundleRef:
    return BundleRef("run", Path(trained["run_dir"]), trained["bundle"].split(":", 1)[1])


def _load(trained: dict, policy: str = "competition_v1"):
    return load_bundle(_ref(trained), expected_policy=policy)


def _context(mode: str = "smoke", source: str = "synthetic") -> PredictionContext:
    return PredictionContext("20261114T000000000000Z-0000abcd", None, source, mode)


def _request(config: ProjectConfig):
    return read_request_dir(Path(request(config)["request_dir"]))


@pytest.fixture
def lightgbm_bundle(synthetic_config: ProjectConfig) -> dict:
    pytest.importorskip("lightgbm")
    return train(synthetic_config, model_id=LIGHTGBM)


# AC-1 ---------------------------------------------------------------------------------


@pytest.mark.models
def test_lightgbm_bundle_holds_native_state_preprocessing_and_signature(lightgbm_bundle):
    directory = Path(lightgbm_bundle["bundle_dir"])
    manifest = json.loads((directory / "bundle.json").read_text())
    preprocessing = json.loads((directory / "preprocessing.json").read_text())
    signature = json.loads((directory / "signature.json").read_text())
    assert (directory / "state/booster.txt").read_text().startswith("tree")
    assert manifest["state_files"][0]["format"] == "lightgbm_text"
    assert set(preprocessing["numeric_medians"]) == set(preprocessing["numeric_features"])
    assert all(v[0] == "__UNKNOWN__" for v in preprocessing["categorical_vocabulary"].values())
    assert preprocessing["target_transform"] == {
        "fit": "log1p",
        "inverse": "expm1",
        "clip_negative": True,
    }
    assert signature["model_columns"] and len(signature["model_columns"]) == len(
        signature["model_dtypes"]
    )
    assert signature["feature_spec_sha256"] and signature["availability_policy"] == "competition_v1"
    for key in ("training_run_id", "snapshot_id", "config_sha256", "code", "versions", "seed"):
        assert manifest[key] is not None
    assert manifest["evaluation_ref"] is None
    run = read_completed_run(Path(lightgbm_bundle["run_dir"]))
    assert {"bundle_manifest", "bundle_state", "verification"} <= {
        entry["kind"] for entry in run["artifacts"]
    }


def test_missing_learned_state_prevents_completion(synthetic_config, monkeypatch):
    from retail_forecast.models.seasonal import SeasonalNaiveAdapter

    monkeypatch.setattr(SeasonalNaiveAdapter, "state_files", lambda self, fitted: {})
    with pytest.raises(BundleError, match="BUNDLE_MISSING"):
        train(synthetic_config, model_id=SEASONAL)
    (run_dir,) = list(synthetic_config.paths.artifacts_dir.iterdir())
    assert json.loads((run_dir / "run.json").read_text())["state"] == "FAILED"
    assert not (run_dir / "COMPLETE.json").exists()


# AC-3 ---------------------------------------------------------------------------------


def _rewrite_json(path: Path, change) -> None:
    payload = json.loads(path.read_text())
    change(payload)
    path.write_bytes(canonical_json(payload))


@pytest.mark.models
@pytest.mark.parametrize("target", ["median", "vocabulary"])
def test_altered_preprocessing_is_an_integrity_failure(lightgbm_bundle, target):
    path = Path(lightgbm_bundle["bundle_dir"]) / "preprocessing.json"

    def change(payload):
        if target == "median":
            column = next(iter(payload["numeric_medians"]))
            payload["numeric_medians"][column] += 1.0
        else:
            column = next(iter(payload["categorical_vocabulary"]))
            values = payload["categorical_vocabulary"][column]
            values[1], values[-1] = values[-1], values[1]
            if values[1] == values[-1]:
                values.append("extra")

    _rewrite_json(path, change)
    with pytest.raises(BundleError, match="BUNDLE_INVALID"):
        _load(lightgbm_bundle)


@pytest.mark.models
def test_consistently_rewritten_bundle_still_fails_its_container(lightgbm_bundle):
    directory = Path(lightgbm_bundle["bundle_dir"])
    _rewrite_json(
        directory / "preprocessing.json",
        lambda payload: payload["numeric_medians"].update(
            {next(iter(payload["numeric_medians"])): 123.0}
        ),
    )
    data = (directory / "preprocessing.json").read_bytes()

    def rehash(manifest):
        manifest["preprocessing_sha256"] = hashlib.sha256(data).hexdigest()
        manifest.pop("bundle_id")
        manifest["bundle_id"] = digest_json(manifest)

    _rewrite_json(directory / "bundle.json", rehash)
    with pytest.raises(BundleError, match="BUNDLE_INVALID"):
        _load(lightgbm_bundle)


@pytest.mark.models
@pytest.mark.parametrize("damage", ["corrupt", "truncate", "remove"])
def test_damaged_native_state_is_rejected_before_native_load(lightgbm_bundle, damage, monkeypatch):
    import lightgbm

    path = Path(lightgbm_bundle["bundle_dir"]) / "state/booster.txt"
    if damage == "corrupt":
        path.write_bytes(path.read_bytes().replace(b"tree", b"TREE", 1))
    elif damage == "truncate":
        path.write_bytes(path.read_bytes()[:100])
    else:
        path.unlink()
    calls = []
    monkeypatch.setattr(lightgbm, "Booster", lambda *a, **k: calls.append(1))
    with pytest.raises(BundleError, match="BUNDLE_INVALID|BUNDLE_MISSING"):
        _load(lightgbm_bundle)
    assert calls == []


def _manifest(trained: dict) -> dict:
    return json.loads((Path(trained["bundle_dir"]) / "bundle.json").read_text())


def _resigned(manifest: dict, **changes) -> dict:
    body = {**{k: v for k, v in manifest.items() if k != "bundle_id"}, **changes}
    return {**body, "bundle_id": digest_json(body)}


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"schema_version": 2}, "BUNDLE_UNSUPPORTED"),
        ({"adapter_version": "2.0"}, "BUNDLE_UNSUPPORTED"),
        ({"adapter_id": "unregistered"}, "BUNDLE_UNSUPPORTED"),
        (
            {
                "state_files": [
                    {"path": "../outside.txt", "bytes": 1, "sha256": "0" * 64, "format": "x"}
                ]
            },
            "BUNDLE_INVALID",
        ),
        (
            {
                "state_files": [
                    {"path": "/etc/passwd", "bytes": 1, "sha256": "0" * 64, "format": "x"}
                ]
            },
            "BUNDLE_INVALID",
        ),
    ],
)
def test_unsupported_versions_and_unsafe_paths_fail_validation(synthetic_config, changes, code):
    manifest = _manifest(train(synthetic_config, model_id=SEASONAL))
    with pytest.raises(BundleError, match=code):
        validate_bundle_manifest(_resigned(manifest, **changes))
    tampered = {**_manifest(train(synthetic_config, model_id=SEASONAL)), **changes}
    with pytest.raises(BundleError):
        validate_bundle_manifest(tampered)


def test_symlinked_state_escaping_the_bundle_is_not_followed(synthetic_config, tmp_path):
    trained = train(synthetic_config, model_id=SEASONAL)
    state = Path(trained["bundle_dir"]) / "state/seasonal.json"
    outside = tmp_path / "outside.json"
    outside.write_bytes(state.read_bytes())
    state.unlink()
    state.symlink_to(outside)
    before = outside.stat().st_atime_ns
    with pytest.raises(BundleError, match="BUNDLE_INVALID"):
        _load(trained)
    assert outside.read_bytes()  # untouched and still present
    assert outside.stat().st_atime_ns >= before


def test_policy_mismatch_fails_before_native_prediction(synthetic_config, monkeypatch):
    from retail_forecast.models.seasonal import SeasonalNaiveAdapter

    trained = train(synthetic_config, model_id=SEASONAL)
    calls = []
    monkeypatch.setattr(SeasonalNaiveAdapter, "load", lambda self, **k: calls.append(1))
    with pytest.raises(BundleError, match="POLICY_MISMATCH"):
        _load(trained, policy="asof_v1")
    assert calls == []


def test_inference_path_has_no_pickle_loader():
    sources = [
        Path("src/retail_forecast", name)
        for name in ("bundles.py", "inference.py", "capsules.py", "request_store.py")
    ] + list(Path("src/retail_forecast/models").glob("*.py"))
    for source in sources:
        text = source.read_text()
        assert not re.search(r"^\s*(import|from)\s+(pickle|joblib|dill|cloudpickle)\b", text, re.M)
        assert not re.search(r"\b(pickle|joblib)\.load", text), source


# AC-4 ---------------------------------------------------------------------------------


@pytest.fixture
def seasonal_ready(synthetic_config):
    trained = train(synthetic_config, model_id=SEASONAL)
    return _load(trained), _request(synthetic_config)


def test_supported_request_gives_the_complete_attributed_grid(seasonal_ready):
    loaded, forecast_request = seasonal_ready
    frame, diagnostics = predict_bundle(loaded, forecast_request, context=_context())
    assert len(frame) == len(forecast_request.series_keys) * 16
    assert frame["horizon"].tolist() == list(range(1, 17)) * len(forecast_request.series_keys)
    assert np.isfinite(frame["prediction"]).all() and (frame["prediction"] >= 0).all()
    assert frame["bundle_id"].unique().tolist() == [loaded.bundle_id]
    assert frame["evaluation_run_id"].isna().all()
    assert diagnostics["rows"] == len(frame)


def test_shuffled_series_and_rows_keep_keyed_values_and_canonical_order(seasonal_ready):
    loaded, forecast_request = seasonal_ready
    expected, _ = predict_bundle(loaded, forecast_request, context=_context())
    shuffled = replace(
        forecast_request,
        series_keys=tuple(reversed(forecast_request.series_keys)),
        future_plan=forecast_request.future_plan.sample(frac=1, random_state=3),
        sales_history=forecast_request.sales_history.sample(frac=1, random_state=4),
    )
    actual, _ = predict_bundle(loaded, shuffled, context=_context())
    pd.testing.assert_frame_equal(actual, expected)


def test_reordered_request_columns_are_rejected(seasonal_ready):
    loaded, forecast_request = seasonal_ready
    plan = forecast_request.future_plan
    reordered = replace(forecast_request, future_plan=plan[list(reversed(plan.columns))])
    with pytest.raises(ForecastRequestError, match="SCHEMA_INVALID"):
        predict_bundle(loaded, reordered, context=_context())


@pytest.mark.parametrize("case", ["unknown", "context", "cutoff", "plan", "infinity"])
def test_unsupported_requests_fail_safely_without_rows(seasonal_ready, case):
    loaded, forecast_request = seasonal_ready
    if case == "unknown":
        bad = replace(forecast_request, series_keys=((99, "UNKNOWN"),))
        code = "UNSUPPORTED_SERIES"
    elif case == "context":
        bad = replace(forecast_request, sales_history=forecast_request.sales_history.iloc[0:0])
        code = "HISTORY_INSUFFICIENT"
    elif case == "cutoff":
        bad = replace(forecast_request, origin=forecast_request.origin - pd.offsets.Day(1))
        code = "ORIGIN_BEFORE_CUTOFF"
    elif case == "plan":
        bad = replace(forecast_request, future_plan=forecast_request.future_plan.iloc[1:])
        code = "FUTURE_PLAN_INCOMPLETE"
    else:
        history = forecast_request.sales_history.copy()
        history.loc[history.index[0], "sales"] = np.inf
        bad = replace(forecast_request, sales_history=history)
        code = "NONFINITE_VALUE"
    with pytest.raises((ForecastRequestError, BundleError), match=code):
        predict_bundle(loaded, bad, context=_context())


@pytest.mark.models
def test_inverse_transform_overflow_fails_before_publication(
    synthetic_config, lightgbm_bundle, monkeypatch
):
    import retail_forecast.models.lightgbm as adapter_module

    monkeypatch.setattr(adapter_module, "predict_log", lambda booster, x: np.full(len(x), 1e6))
    with pytest.raises(ForecastRequestError, match="NONFINITE_VALUE"):
        predict(synthetic_config, bundle=lightgbm_bundle["bundle"])
    runs = sorted(synthetic_config.paths.artifacts_dir.iterdir())
    failed = [r for r in runs if json.loads((r / "run.json").read_text())["kind"] == "prediction"]
    assert len(failed) == 1 and not (failed[0] / "predictions/forecast.csv").exists()
    assert json.loads((failed[0] / "run.json").read_text())["state"] == "FAILED"


def test_mode_or_source_mismatch_is_refused(seasonal_ready):
    loaded, forecast_request = seasonal_ready
    with pytest.raises(BundleError, match="BUNDLE_INVALID"):
        predict_bundle(loaded, forecast_request, context=_context(mode="full", source="kaggle"))


# AC-5 (request path) ------------------------------------------------------------------


def test_request_directory_needs_no_competition_files_or_labels(synthetic_config, tmp_path):
    trained = train(synthetic_config, model_id=SEASONAL)
    request_dir = Path(request(synthetic_config)["request_dir"])
    files = {p.relative_to(request_dir).as_posix() for p in request_dir.rglob("*") if p.is_file()}
    competition = {"train.csv", "test.csv", "sample_submission.csv", "oil.csv"}
    assert not {Path(name).name for name in files} & competition
    sales = pd.read_csv(request_dir / "tables/sales_history.csv")
    origin = json.loads((request_dir / "request.json").read_text())["origin"]
    assert sales["date"].max() <= origin  # no labels after origin travel with the request
    raw = synthetic_config.paths.raw_data_dir
    for name in list(raw.iterdir()):
        name.rename(tmp_path / f"hidden-{name.name}")
    result = predict(synthetic_config, bundle=trained["bundle"], request_dir=request_dir)
    assert len(pd.read_csv(result["forecast"])) == 64


def test_tampered_request_directory_is_rejected(synthetic_config):
    trained = train(synthetic_config, model_id=SEASONAL)
    request_dir = Path(request(synthetic_config)["request_dir"])
    plan = request_dir / "tables/future_plan.csv"
    plan.write_bytes(plan.read_bytes().replace(b",0\n", b",9\n", 1))
    with pytest.raises(ValueError, match="integrity"):
        predict(synthetic_config, bundle=trained["bundle"], request_dir=request_dir)


# AC-6 ---------------------------------------------------------------------------------


@pytest.fixture
def forecast_and_test(synthetic_config):
    result = smoke(synthetic_config)
    forecast = pd.read_csv(result["forecast"], parse_dates=["target_date"])
    test = pd.read_csv(Path(result["run_dir"]) / "inputs/synthetic/test.csv")
    return forecast, test, pd.read_csv(result["submission"])


def test_export_restores_test_order_from_shuffled_rows(forecast_and_test):
    forecast, test, submission = forecast_and_test
    exported = export_kaggle_submission(forecast.sample(frac=1, random_state=5), test)
    pd.testing.assert_frame_equal(exported, submission)
    assert exported["id"].tolist() == test["id"].tolist()


@pytest.mark.parametrize("case", ["missing", "extra", "duplicate", "duplicate_id"])
def test_export_rejects_key_mismatches(forecast_and_test, case):
    forecast, test, _ = forecast_and_test
    if case == "missing":
        forecast = forecast.iloc[1:]
    elif case == "extra":
        extra = forecast.iloc[[0]].assign(target_date=pd.Timestamp("2099-01-01"))
        forecast = pd.concat([forecast, extra])
    elif case == "duplicate":
        forecast = pd.concat([forecast, forecast.iloc[[0]]])
    else:
        test = test.assign(id=test["id"].where(test.index != 1, test["id"].iloc[0]))
    with pytest.raises(ExportError, match="EXPORT_KEY_INVALID"):
        export_kaggle_submission(forecast, test)


# AC-7 / AC-9 eligibility ------------------------------------------------------------------


def test_core_imports_do_not_load_optional_frameworks():
    import subprocess
    import sys

    code = (
        "import sys, retail_forecast.cli, retail_forecast.pipeline, retail_forecast.bundles;"
        "print(sorted(m for m in ('lightgbm','mlflow','torch','sklearn') if m in sys.modules))"
    )
    output = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    ).stdout
    assert output.strip() == "[]"


def test_research_partial_tables_are_not_complete_records():
    research = pd.DataFrame(
        {
            "model_id": ["lightgbm_global"],
            "run_id": ["x"],
            "snapshot_id": ["s"],
            "availability_policy": ["asof_v1"],
            "lineage_status": ["research_partial"],
        }
    )
    with pytest.raises(LineageIneligibleError) as error:
        require_complete_lineage(research)
    assert {"bundle_id", "training_run_id", "prediction_run_id"} <= set(error.value.missing)
