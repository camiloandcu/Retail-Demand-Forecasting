"""Bounded, temporal source-store to target-store GRU encoder adaptation."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from retail_forecast.config import ProjectConfig
from retail_forecast.data import load_frames, validate_dataset, write_synthetic_dataset
from retail_forecast.data_contracts import benchmark_labels, project_snapshot
from retail_forecast.io_utils import write_json
from retail_forecast.metrics import rmsle
from retail_forecast.neural_runtime import resolve_device, training_time_limit
from retail_forecast.provenance import digest_json
from retail_forecast.sequences import SequencePreprocessor, build_panel, request_panel
from retail_forecast.training import cutoff_training_origins, prepare_feature_frames


def split_stores(series: tuple, *, synthetic: bool = False) -> tuple[tuple, tuple]:
    """Split by identifier before measuring any outcomes, never by sales volume."""
    stores = sorted({s for s, _ in series})
    boundary = stores[len(stores) // 2 - 1] if synthetic else 36
    source = tuple(k for k in series if k[0] <= boundary)
    target = tuple(k for k in series if k[0] > boundary)
    if not source or not target or set(source) & set(target):
        raise ValueError("Transfer requires two nonempty disjoint groups")
    if not synthetic and stores != list(range(1, 55)):
        raise ValueError("Full transfer protocol requires stores 1–54")
    return source, target


def copy_encoder(source: Any, target: Any) -> None:
    """Transfer only the temporal encoder; target categories/head stay local."""
    target.gru.load_state_dict(source.gru.state_dict(), strict=True)


def transfer_recipe(config: ProjectConfig) -> dict[str, Any]:
    if config.recurrent is None or config.data.availability_policy != "asof_v1":
        raise ValueError("Transfer requires a recurrent config with asof_v1")
    synthetic = config.data.source == "synthetic"
    return {
        **asdict(config.recurrent),
        "hidden_size": 8 if synthetic else 32,
        "max_epochs": 1 if synthetic else 8,
        "patience": 2,
        "max_training_origins": 3 if synthetic else 8,
        "training_minutes": config.recurrent.training_minutes,
        "device": "cpu" if synthetic else resolve_device(config.recurrent.device),
    }


class _Study:
    def __init__(self, config, frames, groups, root, recipe, snapshot):
        self.config, self.frames, self.groups = config, frames, groups
        self.root, self.recipe, self.snapshot = root, recipe, snapshot
        self.training_seconds = 0.0
        self.histories = []

    def panel(self, group, cutoff):
        origins = cutoff_training_origins(
            cutoff,
            16,
            self.frames["train.csv"]["date"],
            stride_days=16,
            max_origins=self.recipe["max_training_origins"],
            name="transfer",
        )
        return build_panel(
            self.frames["train.csv"],
            self.frames["transactions.csv"],
            self.frames["stores.csv"],
            self.frames["holidays_events.csv"],
            series=group,
            origins=origins,
            cutoff=cutoff,
            policy="asof_v1",
        )

    def request(self, group, cutoff):
        return request_panel(
            project_snapshot(
                self.frames,
                snapshot_id=self.snapshot,
                data_source=self.config.data.source,
                origin=cutoff,
                policy="asof_v1",
                context_days=56,
                supported_series=group,
                series_keys=group,
            )
        )

    def labels(self, panel):
        """Keyed join preserves panel order and refuses missing or duplicate labels."""
        keys = pd.DataFrame(
            [
                (panel.series[s][0], panel.series[s][1], panel.dates[o + h])
                for s, o in panel.keys
                for h in range(1, 17)
            ],
            columns=["store_nbr", "family", "target_date"],
        )
        labels = benchmark_labels(self.frames["train.csv"], keys["target_date"].unique())
        joined = keys.merge(labels, how="left", on=list(keys.columns), validate="one_to_one")
        if joined["sales"].isna().any():
            raise ValueError("Transfer labels do not cover every forecast key")
        return joined

    def train(self, panel, pre, *, name, epochs, encoder=None, validation=None):
        from safetensors.torch import save_file

        from retail_forecast.models.gru import initialize
        from retail_forecast.models.gru_training import fit_epochs

        time_limit = training_time_limit(self.recipe)
        remaining = None if time_limit is None else time_limit - self.training_seconds
        if remaining is not None and remaining <= 0:
            raise RuntimeError("TRANSFER_TOTAL_TRAINING_BUDGET_EXCEEDED")
        model = initialize(pre, self.recipe["hidden_size"], self.config.runtime.seed)
        if encoder is not None:
            copy_encoder(encoder, model)
        attempt_started = time.monotonic()
        try:
            result = fit_epochs(
                model,
                panel,
                pre,
                params={
                    **self.recipe,
                    "training_minutes": None if remaining is None else remaining / 60,
                },
                seed=self.config.runtime.seed,
                epochs=epochs,
                validation=validation,
                identity={"phase": name, "preprocessor": pre.sha256},
            )
        except BaseException as error:
            resources = getattr(error, "training_resources", {})
            seconds = resources.get("attempt_training_seconds", time.monotonic() - attempt_started)
            self.training_seconds += seconds
            self.histories.append(
                {
                    "phase": name,
                    "status": "incomplete",
                    "error": str(error),
                    "training_seconds": seconds,
                    "resources": resources,
                    "cutoff": panel.cutoff.date().isoformat(),
                }
            )
            write_json(self.root / "training.json", self.histories)
            raise
        self.training_seconds += result["training_seconds"]
        model.cpu().eval()
        folder = self.root / "models" / name
        folder.mkdir(parents=True)
        save_file(
            {k: v.detach().contiguous() for k, v in model.state_dict().items()},
            folder / "weights.safetensors",
        )
        write_json(folder / "preprocessing.json", pre.state)
        record = {
            "phase": name,
            "cutoff": panel.cutoff.date().isoformat(),
            "training_sample_sha256": panel.identity,
            "preprocessor_sha256": pre.sha256,
            **result,
        }
        self.histories.append(record)
        write_json(self.root / "training.json", self.histories)
        return model, result

    def source(self, cutoff, name, epochs):
        panel = self.panel(self.groups[0], cutoff)
        pre = SequencePreprocessor.fit(panel, cutoff=cutoff)
        return self.train(panel, pre, name=name, epochs=epochs)

    def pair(self, cutoff, name, epochs=None):
        source, _ = self.source(cutoff, f"{name}/source", self.recipe["max_epochs"])
        panel = self.panel(self.groups[1], cutoff)
        pre = SequencePreprocessor.fit(panel, cutoff=cutoff)
        validation = None
        if epochs is None:
            val = self.request(self.groups[1], cutoff)
            actual_log = np.log1p(self.labels(val)["sales"].to_numpy().reshape(-1, 16))
            validation = (val, actual_log)
        outputs = {}
        for variant in ("scratch", "adapted"):
            model, result = self.train(
                panel,
                pre,
                name=f"{name}/{variant}",
                epochs=self.recipe["max_epochs"] if epochs is None else epochs[variant],
                encoder=source if variant == "adapted" else None,
                validation=validation,
            )
            outputs[variant] = (model, pre, result)
        return outputs

    def probe(self, cutoff):
        """Resource-only probe, without internal or outer validation scores."""
        timings = []
        for name, group in zip(("source", "target"), self.groups, strict=True):
            panel = self.panel(group, cutoff)
            pre = SequencePreprocessor.fit(panel, cutoff=cutoff)
            _, result = self.train(panel, pre, name=f"probe/{name}", epochs=1)
            timings.append(result["training_seconds"])
        # Two source fits and four target fits per fold; conservative 2x margin.
        estimate = (
            self.training_seconds
            + (2 * timings[0] + 4 * timings[1])
            * self.recipe["max_epochs"]
            * len(self.config.validation.origins)
            * 2
        )
        time_limit = training_time_limit(self.recipe)
        record = {
            "outer_scores_computed": False,
            "internal_scores_computed": False,
            "estimated_training_seconds_with_2x_margin": estimate,
            "budget_seconds": time_limit,
            "passed": time_limit is None or estimate <= time_limit,
            "limitation": "First-cutoff epoch timing; not a runtime guarantee.",
        }
        write_json(self.root / "feasibility.json", record)
        if not record["passed"]:
            raise RuntimeError("TRANSFER_PROBE_EXCEEDS_BUDGET: register a smaller recipe first")

    def score_fold(self, origin, outputs, fold):
        from retail_forecast.models.gru import inverse_sales
        from retail_forecast.models.gru_training import predict_z

        panel = self.request(self.groups[1], origin)
        predictions = []
        for name, (model, pre, _) in outputs.items():
            z = predict_z(model, panel, pre, self.recipe["batch_size"])
            batch = pre.batch(panel, np.arange(len(panel.keys)))
            values, _ = inverse_sales(z, batch["center"], batch["scale"])
            keys = pd.DataFrame(
                [
                    (panel.series[s][0], panel.series[s][1], panel.dates[o + h], h)
                    for s, o in panel.keys
                    for h in range(1, 17)
                ],
                columns=["store_nbr", "family", "target_date", "horizon"],
            )
            keys["prediction"] = values.flatten()
            keys["model_id"], keys["fold"], keys["origin"] = name, fold, origin
            predictions.append(keys)
        # Models never receive outer labels; attach only after both have predicted.
        labels = self.labels(panel)
        return pd.concat(predictions).merge(
            labels[["store_nbr", "family", "target_date", "sales"]],
            on=["store_nbr", "family", "target_date"],
            validate="many_to_one",
        )


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["fold", "store_nbr", "family", "target_date"]
    left = predictions.loc[predictions.model_id.eq("scratch"), keys + ["sales"]]
    right = predictions.loc[predictions.model_id.eq("adapted"), keys + ["sales"]]
    if left.empty or left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError("Empty or duplicated transfer prediction keys")
    if (
        not left.sort_values(keys)
        .reset_index(drop=True)
        .equals(right.sort_values(keys).reset_index(drop=True))
    ):
        raise ValueError("Transfer models must share every evaluation key and target")
    rows = []
    for name, values in predictions.groupby("model_id"):
        rows.append(
            dict(
                model_id=name,
                scope="global",
                segment="all",
                rows=len(values),
                rmsle=rmsle(values.sales, values.prediction),
            )
        )
        for scope in ("fold", "horizon", "store_nbr", "family"):
            for segment, group in values.groupby(scope):
                rows.append(
                    dict(
                        model_id=name,
                        scope=scope,
                        segment=str(segment),
                        rows=len(group),
                        rmsle=rmsle(group.sales, group.prediction),
                    )
                )
    return pd.DataFrame(rows)


def transfer_experiment(config: ProjectConfig) -> dict[str, Any]:
    """Execute one immutable attempt; incomplete attempts cannot become results."""
    recipe = transfer_recipe(config)
    if config.data.source == "synthetic":
        write_synthetic_dataset(config)
    summary = validate_dataset(config)
    import torch

    if recipe["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE: the explicitly requested CUDA device is unavailable")
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(config.paths.raw_data_dir.glob("*.csv"))
    }
    snapshot = digest_json(files)
    frames = prepare_feature_frames(load_frames(config.paths.raw_data_dir))
    series = tuple(sorted(set(map(tuple, frames["train.csv"][["store_nbr", "family"]].to_numpy()))))
    groups = split_stores(series, synthetic=config.data.source == "synthetic")
    root = (
        config.paths.artifacts_dir
        / "transfer"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    root.mkdir(parents=True)
    manifest = {
        "status": "running",
        "data_source": config.data.source,
        "availability_policy": "asof_v1",
        "seed": config.runtime.seed,
        "origins": [str(o) for o in config.validation.origins],
        "horizon": 16,
        "source_stores": sorted({int(k[0]) for k in groups[0]}),
        "target_stores": sorted({int(k[0]) for k in groups[1]}),
        "snapshot_sha256": snapshot,
        "input_hashes": files,
        "recipe": recipe,
        "dataset": summary.to_dict(),
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name() if recipe["device"] == "cuda" else "cpu",
        "transferred_state": (
            "gru only; target embeddings/head/optimizer/transforms initialized anew"
        ),
    }
    write_json(root / "manifest.json", manifest)
    study = _Study(config, frames, groups, root, recipe, snapshot)
    started = time.monotonic()
    try:
        study.probe(pd.Timestamp(config.validation.origins[0]) - pd.Timedelta(days=16))
        predictions, selections = [], []
        for fold, date in enumerate(config.validation.origins, 1):
            origin = pd.Timestamp(date)
            inner = study.pair(origin - pd.Timedelta(days=16), f"fold_{fold}/selection")
            epochs = {name: item[2]["selected_epochs"] for name, item in inner.items()}
            selections.append({"fold": fold, "selected_epochs": epochs})
            write_json(root / "selection.json", selections)
            del inner
            outer = study.pair(origin, f"fold_{fold}/refit", epochs=epochs)
            predictions.append(study.score_fold(origin, outer, fold))
            del outer
        combined = pd.concat(predictions, ignore_index=True)
        metrics = summarize_predictions(combined)
        combined.to_csv(root / "predictions.csv", index=False)
        metrics.to_csv(root / "results.csv", index=False)
        manifest["status"] = "completed"
        manifest["rows_per_model"] = len(combined) // 2
    except BaseException as error:
        manifest.update(status="incomplete", error=str(error))
        raise
    finally:
        manifest["training_seconds"] = study.training_seconds
        manifest["wall_seconds"] = time.monotonic() - started
        manifest["artifacts_sha256"] = {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "manifest.json"
        }
        write_json(root / "manifest.json", manifest)
    return {"run_dir": str(root), "manifest": manifest}
