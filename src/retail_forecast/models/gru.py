"""One global joint 16-output GRU, with safe offline CPU inference state."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors
from torch import nn

from retail_forecast.data_contracts import ForecastRequest
from retail_forecast.models.base import (
    FitContext,
    FittedModel,
    StateFile,
    TrainingSet,
    base_signature,
)
from retail_forecast.models.gru_training import fit_epochs, predict_z
from retail_forecast.sequences import (
    FUTURE_CHANNELS,
    HISTORY_CHANNELS,
    STATIC_CHANNELS,
    SequencePreprocessor,
    request_panel,
)

EMBEDDINGS = (8, 8, 4, 4, 2, 2)


class JointGRU(nn.Module):
    """One unidirectional layer, static embeddings and flattened future covariates."""

    def __init__(self, hidden_size: int, vocabulary_sizes: list[int]) -> None:
        super().__init__()
        if not 1 <= hidden_size <= 512 or len(vocabulary_sizes) != 6:
            raise ValueError("Unsupported GRU architecture")
        if any(type(n) is not int or not 1 <= n <= 10000 for n in vocabulary_sizes):
            raise ValueError("Invalid GRU vocabulary dimensions")
        self.gru = nn.GRU(8, hidden_size, batch_first=True)
        self.embeddings = nn.ModuleList(
            nn.Embedding(n, d) for n, d in zip(vocabulary_sizes, EMBEDDINGS, strict=True)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size + sum(EMBEDDINGS) + 16 * 12, 128), nn.ReLU(), nn.Linear(128, 16)
        )

    def forward(self, history: Any, future: Any, static: Any) -> Any:
        _, h = self.gru(history)
        embeddings = [layer(static[:, i]) for i, layer in enumerate(self.embeddings)]
        return self.head(torch.cat([h[-1], *embeddings, future.flatten(1)], dim=1))


def initialize(pre: SequencePreprocessor, hidden_size: int, seed: int) -> JointGRU:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return JointGRU(hidden_size, [len(pre.state["vocabularies"][c]) for c in STATIC_CHANNELS])


def inverse_sales(z: np.ndarray, center: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, int]:
    # CPU float32 is the declared reference; reject overflow rather than saturating.
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.expm1(center + scale * np.asarray(z, dtype=np.float32))
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite original-scale GRU output")
    return np.maximum(values, 0).astype(np.float64), int((values < 0).sum())


class GRUAdapter:
    adapter_id = "gru_global"
    version = "1.0"
    needs_training_batch = False
    needs_sequences = True

    def fit(self, training: TrainingSet, context: FitContext) -> FittedModel:
        if context.adapter_id != self.adapter_id or training.sequences is None:
            raise ValueError("GRU requires typed sequence training data")
        if context.selected_epochs is None or context.selected_epochs < 1:
            raise ValueError("GRU requires an internally selected fixed epoch count")
        pre = SequencePreprocessor.fit(training.sequences, cutoff=training.cutoff)
        model = initialize(pre, training.params["hidden_size"], context.seed)
        fit_epochs(
            model,
            training.sequences,
            pre,
            params=dict(training.params),
            seed=context.seed,
            epochs=context.selected_epochs,
            identity={"preprocessor": pre.sha256},
        )
        return self.fitted(model, pre, training, context)

    def fitted(
        self, model: Any, pre: SequencePreprocessor, training: TrainingSet, context: FitContext
    ) -> FittedModel:
        model = model.cpu().eval()
        signature = base_signature(
            adapter_id=self.adapter_id,
            adapter_version=self.version,
            spec=training.spec,
            request_context_days=56,
            min_context_days=56,
            target_transform="log1p_per_series_standardization",
        )
        signature.update(
            {
                "history_channels": list(HISTORY_CHANNELS),
                "future_channels": list(FUTURE_CHANNELS),
                "static_channels": list(STATIC_CHANNELS),
                "hidden_size": training.params["hidden_size"],
                "embedding_dimensions": list(EMBEDDINGS),
                "precision": "float32",
                "native_library": "torch",
                "native_version": torch.__version__,
            }
        )
        preprocessing = {
            "schema_version": 1,
            "sequence": pre.state,
            "preprocessor_sha256": pre.sha256,
            "params": dict(training.params),
            "selected_epochs": context.selected_epochs,
            "training_origins": [o.date().isoformat() for o in training.training_origins],
        }
        return FittedModel(
            self.adapter_id,
            self.version,
            context.availability_policy,
            context.input_contract_version,
            training.cutoff,
            training.supported_series,
            56,
            56,
            signature,
            preprocessing,
            model,
        )

    def state_files(self, fitted: FittedModel) -> dict[str, StateFile]:
        state = {k: v.detach().cpu().contiguous() for k, v in fitted.native.state_dict().items()}
        return {"weights.safetensors": StateFile(save_tensors(state), "safetensors")}

    def load(
        self,
        *,
        signature: dict[str, Any],
        preprocessing: dict[str, Any],
        state: Mapping[str, bytes],
        identity: dict[str, Any],
    ) -> FittedModel:
        pre = SequencePreprocessor.load(preprocessing["sequence"])
        if pre.sha256 != preprocessing["preprocessor_sha256"]:
            raise ValueError("Sequence preprocessing identity mismatch")
        if (
            signature["history_channels"] != list(HISTORY_CHANNELS)
            or signature["future_channels"] != list(FUTURE_CHANNELS)
            or signature["static_channels"] != list(STATIC_CHANNELS)
            or signature["embedding_dimensions"] != list(EMBEDDINGS)
            or signature["precision"] != "float32"
            or signature["native_library"] != "torch"
            or signature["native_version"].split(".")[0] != torch.__version__.split(".")[0]
            or set(state) != {"weights.safetensors"}
        ):
            raise ValueError("Incompatible GRU signature/state")
        model = initialize(pre, signature["hidden_size"], 0)
        weights = load_tensors(state["weights.safetensors"])
        if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in weights.values()):
            raise ValueError("Invalid GRU weights")
        model.load_state_dict(weights, strict=True)
        model.eval()
        series = tuple((int(r[0]), str(r[1])) for r in pre.state["target_stats"])
        if series != tuple(tuple(k) for k in identity["supported_series"]):
            raise ValueError("GRU series identity mismatch")
        return FittedModel(
            self.adapter_id,
            self.version,
            signature["availability_policy"],
            signature["input_contract_version"],
            pd.Timestamp(pre.state["fit_cutoff"]),
            series,
            56,
            56,
            signature,
            preprocessing,
            model,
        )

    def predict_raw(self, fitted: FittedModel, request: ForecastRequest) -> pd.DataFrame:
        torch.set_num_threads(fitted.preprocessing["params"]["cpu_threads"])
        pre = SequencePreprocessor.load(fitted.preprocessing["sequence"])
        panel = request_panel(request)
        z = predict_z(fitted.native.cpu(), panel, pre, fitted.preprocessing["params"]["batch_size"])
        b = pre.batch(panel, np.arange(len(panel.keys)))
        values, clipped = inverse_sales(z, b["center"], b["scale"])
        rows = [
            (panel.series[s][0], panel.series[s][1], panel.dates[o + h], float(values[i, h - 1]))
            for i, (s, o) in enumerate(panel.keys)
            for h in range(1, 17)
        ]
        result = pd.DataFrame(rows, columns=["store_nbr", "family", "target_date", "prediction"])
        result.attrs["clipped_negative_count"] = clipped
        result.attrs["sparse_context_count"] = int((b["history"][:, :, 3].sum(axis=1) < 56).sum())
        return result
