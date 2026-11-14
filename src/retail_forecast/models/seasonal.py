"""Seasonal naive and seasonal average adapters with structured JSON state."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd

from retail_forecast.baseline import fit_seasonal_average, fit_seasonal_naive
from retail_forecast.data_contracts import ForecastRequest
from retail_forecast.features import FeatureSpec
from retail_forecast.io_utils import canonical_json, parse_json_bytes
from retail_forecast.models.base import (
    RAW_PREDICTION_COLUMNS,
    FitContext,
    FittedModel,
    StateFile,
    TrainingSet,
    base_signature,
)

STATE_FILE = "seasonal.json"
STATE_FORMAT = "seasonal_table_json_v1"


def _split_key(key: str) -> tuple[int, str, str]:
    # baseline.py joins with "|"; split from both ends so a family cannot shift fields.
    store, rest = key.split("|", 1)
    family, slot = rest.rsplit("|", 1)
    return int(store), family, slot


class _SeasonalAdapter:
    adapter_id: str
    version = "1.0"
    needs_training_batch = False
    weeks: int | None

    def _fit_table(self, history: pd.DataFrame, cutoff: pd.Timestamp, period: int) -> dict:
        raise NotImplementedError

    def fit(self, training: TrainingSet, context: FitContext) -> FittedModel:
        if context.adapter_id != self.adapter_id:
            raise ValueError("Fit context names another adapter")
        period = int(training.params.get("seasonal_period", 7))
        fitted = self._fit_table(training.request.sales_history, training.cutoff, period)
        weekday_rows = sorted(
            [*_split_key(key)[:2], int(_split_key(key)[2]), float(value)]
            for key, value in fitted["weekday_values"].items()
        )
        fallback_rows = sorted(
            [*_split_key(key)[:2], float(value)] for key, value in fitted["fallback_values"].items()
        )
        fallback_keys = {(store, family) for store, family, _ in fallback_rows}
        missing = [key for key in training.supported_series if key not in fallback_keys]
        if missing:
            store, family = missing[0]
            raise ValueError(f"No learned state for series store_nbr={store},family={family}")
        preprocessing = {
            "schema_version": 1,
            "fitted_through": training.cutoff.date().isoformat(),
            "seasonal_period": period,
            "weeks": self.weeks,
            "missing_weekday_rule": "series_last_value_in_window"
            if self.weeks is None
            else "series_mean_of_last_weeks_x_period_rows",
        }
        signature = base_signature(
            adapter_id=self.adapter_id,
            adapter_version=self.version,
            spec=training.spec,
            request_context_days=training.spec.context_days,
            min_context_days=1,
            target_transform="identity",
        )
        signature["feature_spec_sha256"] = training.spec.sha256
        # The table is the history at the cutoff; a later origin would replay stale weeks.
        signature["origin_rule"] = "equals_training_cutoff"
        native = {
            "weekday": {(s, f, w): v for s, f, w, v in weekday_rows},
            "fallback": {(s, f): v for s, f, v in fallback_rows},
            "rows": {"weekday_values": weekday_rows, "fallback_values": fallback_rows},
        }
        return FittedModel(
            adapter_id=self.adapter_id,
            adapter_version=self.version,
            availability_policy=training.spec.availability_policy,
            input_contract_version=training.spec.policy.input_contract_version,
            training_cutoff=training.cutoff,
            supported_series=training.supported_series,
            min_context_days=1,
            request_context_days=training.spec.context_days,
            signature=signature,
            preprocessing=preprocessing,
            native=native,
        )

    def state_files(self, fitted: FittedModel) -> dict[str, StateFile]:
        return {STATE_FILE: StateFile(canonical_json(fitted.native["rows"]), STATE_FORMAT)}

    def load(
        self,
        *,
        signature: dict[str, Any],
        preprocessing: dict[str, Any],
        state: Mapping[str, bytes],
        identity: dict[str, Any],
    ) -> FittedModel:
        if set(state) != {STATE_FILE}:
            raise ValueError("Seasonal state files do not match the adapter")
        rows = parse_json_bytes(state[STATE_FILE])
        if not isinstance(rows, dict) or set(rows) != {"weekday_values", "fallback_values"}:
            raise ValueError("Invalid seasonal state")

        def value_of(raw: object) -> float:
            if type(raw) not in (int, float) or not math.isfinite(raw) or raw < 0:
                raise ValueError("Seasonal values must be finite nonnegative numbers")
            return float(raw)

        weekday = {}
        for row in rows["weekday_values"]:
            store, family, day, value = row
            if type(store) is not int or not isinstance(family, str) or type(day) is not int:
                raise ValueError("Invalid seasonal weekday key")
            if not 0 <= day <= 6:
                raise ValueError("Invalid seasonal weekday state")
            weekday[(store, family, day)] = value_of(value)
        fallback = {}
        for store, family, value in rows["fallback_values"]:
            if type(store) is not int or not isinstance(family, str):
                raise ValueError("Invalid seasonal fallback key")
            fallback[(store, family)] = value_of(value)
        spec = FeatureSpec(signature["availability_policy"], horizon=signature["horizon"])
        if spec.sha256 != signature["feature_spec_sha256"]:
            raise ValueError("Seasonal signature feature specification mismatch")
        supported = identity["supported_series"]
        if set(fallback) != set(supported):
            raise ValueError("Seasonal state does not cover the supported series exactly")
        return FittedModel(
            adapter_id=self.adapter_id,
            adapter_version=signature["adapter_version"],
            availability_policy=signature["availability_policy"],
            input_contract_version=signature["input_contract_version"],
            training_cutoff=pd.Timestamp(preprocessing["fitted_through"]),
            supported_series=supported,
            min_context_days=signature["min_context_days"],
            request_context_days=signature["request_context_days"],
            signature=signature,
            preprocessing=preprocessing,
            native={"weekday": weekday, "fallback": fallback, "rows": rows},
        )

    def predict_raw(self, fitted: FittedModel, request: ForecastRequest) -> pd.DataFrame:
        grid = pd.DataFrame(request.series_keys, columns=["store_nbr", "family"]).merge(
            pd.DataFrame({"target_date": request.target_dates}), how="cross"
        )
        values = []
        for store, family, target in grid.itertuples(index=False):
            key = (int(store), str(family))
            value = fitted.native["weekday"].get((*key, int(target.dayofweek)))
            if value is None:
                value = fitted.native["fallback"][key]
            values.append(max(0.0, float(value)))
        grid["prediction"] = pd.Series(values, index=grid.index, dtype="float64")
        return grid[list(RAW_PREDICTION_COLUMNS)]


class SeasonalNaiveAdapter(_SeasonalAdapter):
    adapter_id = "seasonal_naive_7d"
    weeks = None

    def _fit_table(self, history: pd.DataFrame, cutoff: pd.Timestamp, period: int) -> dict:
        return fit_seasonal_naive(history, cutoff, period)


class SeasonalAverageAdapter(_SeasonalAdapter):
    adapter_id = "seasonal_average_4w"
    weeks = 4

    def _fit_table(self, history: pd.DataFrame, cutoff: pd.Timestamp, period: int) -> dict:
        return fit_seasonal_average(history, cutoff, weeks=self.weeks, seasonal_period=period)
