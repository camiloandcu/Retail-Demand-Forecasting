"""Global LightGBM adapter: native booster text plus JSON learned preprocessing.

`lightgbm` is imported only inside functions so core installs never load it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from retail_forecast.data_contracts import ForecastRequest
from retail_forecast.features import (
    TARGET_COLUMN,
    UNKNOWN_CATEGORY,
    FeatureSpec,
    FittedPreprocessor,
    build_request_feature_batch,
    fit_preprocessor,
    transform_features,
)
from retail_forecast.models.base import (
    RAW_PREDICTION_COLUMNS,
    FitContext,
    FittedModel,
    StateFile,
    TrainingSet,
    base_signature,
)

BOOSTER_FILE = "booster.txt"
BOOSTER_FORMAT = "lightgbm_text"
SUPPORTED_NATIVE_MAJOR = 4


def _lightgbm() -> Any:
    try:
        import lightgbm
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("Install the 'models' extra to use lightgbm_global") from error
    if int(lightgbm.__version__.split(".")[0]) != SUPPORTED_NATIVE_MAJOR:
        raise RuntimeError("Unsupported LightGBM major version")
    return lightgbm


def lightgbm_params(baseline: Any, seed: int) -> dict[str, Any]:
    """Pre-registered tree parameters shared by the baseline and the adapter."""

    return {
        "objective": "regression",
        "n_estimators": baseline.tree_n_estimators,
        "learning_rate": baseline.tree_learning_rate,
        "num_leaves": baseline.tree_num_leaves,
        "min_child_samples": baseline.tree_min_child_samples,
        "colsample_bytree": baseline.tree_feature_fraction,
        "reg_lambda": baseline.tree_l2,
        "random_state": seed,
        "n_jobs": -1,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def model_matrix(transformed: pd.DataFrame) -> pd.DataFrame:
    """Model inputs: drop keys and the label; inputs stay label-free at prediction."""

    return transformed.drop(columns=["forecast_origin", "target_date"]).drop(
        columns=TARGET_COLUMN, errors="ignore"
    )


def fit_booster(x_train: pd.DataFrame, sales: pd.Series, params: Mapping[str, Any]) -> Any:
    """Fit the single pre-registered tree on log1p sales; return the native Booster."""

    lgb = _lightgbm()
    categorical = [column for column in x_train if column.startswith("cat__")]
    tree = lgb.LGBMRegressor(**dict(params))
    tree.fit(x_train, np.log1p(sales.to_numpy()), categorical_feature=categorical)
    return tree.booster_


def predict_log(booster: Any, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(booster.predict(x), dtype=np.float64)


def _preprocessor_from_dict(payload: dict[str, Any]) -> FittedPreprocessor:
    vocabulary = {
        column: tuple(str(value) for value in values)
        for column, values in payload["categorical_vocabulary"].items()
    }
    if any(not values or values[0] != UNKNOWN_CATEGORY for values in vocabulary.values()):
        raise ValueError("Categorical vocabulary must reserve code zero for unknown values")
    if any(len(set(values)) != len(values) for values in vocabulary.values()):
        raise ValueError("Categorical vocabulary contains duplicates")
    medians = {column: float(value) for column, value in payload["numeric_medians"].items()}
    if not all(np.isfinite(value) for value in medians.values()):
        raise ValueError("Medians must be finite")
    return FittedPreprocessor(pd.Timestamp(payload["fitted_through"]), medians, vocabulary)


class LightGBMAdapter:
    adapter_id = "lightgbm_global"
    version = "1.0"
    needs_training_batch = True

    def fit(self, training: TrainingSet, context: FitContext) -> FittedModel:
        if context.adapter_id != self.adapter_id:
            raise ValueError("Fit context names another adapter")
        batch = training.batch
        if batch is None or TARGET_COLUMN not in batch.frame:
            raise ValueError("LightGBM requires a labeled training batch")
        if batch.frame["target_date"].max() > training.cutoff:
            raise ValueError("Training labels cross the training cutoff")
        fitted = fit_preprocessor(batch, training.cutoff)
        transformed = transform_features(batch, fitted)
        x_train = model_matrix(transformed)
        booster = fit_booster(x_train, transformed[TARGET_COLUMN], training.params)
        lgb = _lightgbm()
        spec = training.spec
        signature = base_signature(
            adapter_id=self.adapter_id,
            adapter_version=self.version,
            spec=spec,
            request_context_days=spec.context_days,
            min_context_days=1,
            target_transform="log1p",
        )
        signature.update(
            {
                "feature_spec_sha256": spec.sha256,
                "model_columns": list(x_train.columns),
                "model_dtypes": [str(dtype) for dtype in x_train.dtypes],
                "categorical_columns": [c for c in x_train if c.startswith("cat__")],
                "native_library": "lightgbm",
                "native_version": lgb.__version__,
            }
        )
        preprocessing = {
            "schema_version": 1,
            **fitted.to_dict(),
            "preprocessor_sha256": fitted.sha256,
            "feature_spec": spec.to_dict(),
            "numeric_features": list(batch.numeric_features),
            "categorical_features": list(batch.categorical_features),
            "unknown_category": UNKNOWN_CATEGORY,
            "target_transform": {"fit": "log1p", "inverse": "expm1", "clip_negative": True},
            "training_origins": [origin.date().isoformat() for origin in training.training_origins],
            "params": dict(training.params),
        }
        return FittedModel(
            adapter_id=self.adapter_id,
            adapter_version=self.version,
            availability_policy=spec.availability_policy,
            input_contract_version=spec.policy.input_contract_version,
            training_cutoff=training.cutoff,
            supported_series=training.supported_series,
            min_context_days=1,
            request_context_days=spec.context_days,
            signature=signature,
            preprocessing=preprocessing,
            native={"booster": booster, "preprocessor": fitted, "spec": spec},
        )

    def state_files(self, fitted: FittedModel) -> dict[str, StateFile]:
        text = fitted.native["booster"].model_to_string()
        return {BOOSTER_FILE: StateFile(text.encode("utf-8"), BOOSTER_FORMAT)}

    def load(
        self,
        *,
        signature: dict[str, Any],
        preprocessing: dict[str, Any],
        state: Mapping[str, bytes],
        identity: dict[str, Any],
    ) -> FittedModel:
        if set(state) != {BOOSTER_FILE}:
            raise ValueError("LightGBM state files do not match the adapter")
        spec = FeatureSpec(
            **{
                key: tuple(value) if isinstance(value, list) else value
                for key, value in preprocessing["feature_spec"].items()
            }
        )
        if spec.sha256 != signature["feature_spec_sha256"]:
            raise ValueError("LightGBM feature specification mismatch")
        if spec.availability_policy != signature["availability_policy"]:
            raise ValueError("LightGBM preprocessing policy mismatch")
        fitted = _preprocessor_from_dict(preprocessing)
        if fitted.sha256 != preprocessing["preprocessor_sha256"]:
            raise ValueError("LightGBM preprocessing checksum mismatch")
        # Canonical JSON sorts mapping keys; the explicit feature lists carry the order.
        numeric = preprocessing["numeric_features"]
        categorical = preprocessing["categorical_features"]
        if set(fitted.numeric_medians) != set(numeric) or len(numeric) != len(set(numeric)):
            raise ValueError("LightGBM medians do not match the numeric features")
        if set(fitted.categorical_vocabulary) != set(categorical):
            raise ValueError("LightGBM vocabularies do not match the categorical features")
        fitted = FittedPreprocessor(
            fitted.fitted_through,
            {column: fitted.numeric_medians[column] for column in numeric},
            {column: fitted.categorical_vocabulary[column] for column in categorical},
        )
        lgb = _lightgbm()
        booster = lgb.Booster(model_str=state[BOOSTER_FILE].decode("utf-8"))
        if booster.feature_name() != signature["model_columns"]:
            raise ValueError("LightGBM booster features differ from the signature")
        return FittedModel(
            adapter_id=self.adapter_id,
            adapter_version=signature["adapter_version"],
            availability_policy=signature["availability_policy"],
            input_contract_version=signature["input_contract_version"],
            training_cutoff=pd.Timestamp(preprocessing["fitted_through"]),
            supported_series=identity["supported_series"],
            min_context_days=signature["min_context_days"],
            request_context_days=signature["request_context_days"],
            signature=signature,
            preprocessing=preprocessing,
            native={"booster": booster, "preprocessor": fitted, "spec": spec},
        )

    def predict_raw(self, fitted: FittedModel, request: ForecastRequest) -> pd.DataFrame:
        spec: FeatureSpec = fitted.native["spec"]
        batch = build_request_feature_batch(request, spec, supported_series=fitted.supported_series)
        if (
            list(batch.numeric_features) != fitted.preprocessing["numeric_features"]
            or list(batch.categorical_features) != fitted.preprocessing["categorical_features"]
        ):
            raise ValueError("Request features differ from the fitted preprocessing")
        transformed = transform_features(batch, fitted.native["preprocessor"])
        columns = fitted.signature["model_columns"]
        x = model_matrix(transformed)
        if sorted(x.columns) != sorted(columns):
            raise ValueError("Model matrix columns differ from the signature")
        x = x[columns]  # project by name, never by position
        if [str(dtype) for dtype in x.dtypes] != fitted.signature["model_dtypes"]:
            raise ValueError("Model matrix dtypes differ from the signature")
        with np.errstate(over="ignore"):
            values = np.expm1(predict_log(fitted.native["booster"], x))
        output = batch.frame[["store_nbr", "family", "target_date"]].copy()
        output["prediction"] = values
        return output[list(RAW_PREDICTION_COLUMNS)].reset_index(drop=True)
