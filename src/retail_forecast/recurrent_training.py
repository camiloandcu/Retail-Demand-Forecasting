"""Temporal recurrent fitting and checkpoint recovery.

Each fold completes its own training run. Recovery can reuse verified earlier
folds and continue the interrupted phase in a new linked run. No outer actuals
enter selection, preprocessing or prediction.
"""

from __future__ import annotations

import hashlib
import resource
import time
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retail_forecast.artifact_schema import validate_manifest
from retail_forecast.artifacts import (
    _verify_outputs,
    allocate_run,
    finish_run,
    resolve_run,
    run_result,
    safe_path,
    write_json_artifact,
    write_run_artifact,
)
from retail_forecast.bundles import BundleRef, load_bundle, save_bundle
from retail_forecast.config import ProjectConfig
from retail_forecast.data import validate_dataset
from retail_forecast.data_contracts import benchmark_labels
from retail_forecast.features import FeatureSpec
from retail_forecast.io_utils import read_json
from retail_forecast.models.base import FitContext
from retail_forecast.neural_runtime import resolve_device, training_time_limit
from retail_forecast.provenance import digest_json
from retail_forecast.run_resume import resume_run
from retail_forecast.sequences import SequencePreprocessor, request_panel
from retail_forecast.training import build_training_set


def _recipe(config: ProjectConfig) -> dict[str, Any]:
    if config.recurrent is None or config.data.availability_policy != "asof_v1":
        raise ValueError("GRU requires an explicit recurrent recipe and asof_v1 policy")
    params = asdict(config.recurrent)
    params["device"] = resolve_device(params["device"])
    return params


def _context(run: Any, cutoff: pd.Timestamp, config: ProjectConfig, epochs: int) -> FitContext:
    return FitContext(
        run.run_id,
        "gru_global",
        config.data.availability_policy,
        config.data.source,
        config.mode,
        cutoff,
        epochs,
        config.runtime.seed,
        run.manifest["config_sha256"],
        run.manifest["snapshot_id"],
        1,
    )


def _checkpoint_source(config: ProjectConfig, source: str) -> tuple[str, dict[str, Any]]:
    root = resolve_run(config, source)
    manifest = read_json(safe_path(root, "run.json", regular=True))
    listed = [r["path"] for r in manifest["artifacts"] if r["kind"] == "checkpoint"]
    if not listed:
        raise ValueError("RESUME_UNSUPPORTED: no recurrent checkpoint")
    # Artifact listing is chronological; epoch files are immutable.
    return listed[-1], read_json(safe_path(root, "recurrent/context.json", regular=True))


def _train_one(
    config: ProjectConfig,
    *,
    cutoff: pd.Timestamp,
    slot: str,
    completed: list[dict[str, Any]],
    run_id: str | None = None,
    resume_from: str | None = None,
) -> dict[str, Any]:
    import torch

    from retail_forecast.models.gru import GRUAdapter, initialize
    from retail_forecast.models.gru_training import fit_epochs, pack_checkpoint, unpack_checkpoint
    from retail_forecast.pipeline import (
        _stage_inputs,
        _supported_series,
        _verified_frames,
        compare_raw_predictions,
    )

    params = _recipe(config)
    time_limit = training_time_limit(params)
    attempt_started = time.monotonic()
    hardware = {
        "device": params["device"],
        "name": "cpu",
        "compute_capability": None,
        "torch_cuda_runtime": torch.version.cuda,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "host_peak_method": "process_lifetime_ru_maxrss_linux",
    }
    if params["device"] == "cuda":
        hardware["name"] = "cuda_unavailable"
        if torch.cuda.is_available():
            hardware["name"] = torch.cuda.get_device_name()
            hardware["compute_capability"] = list(torch.cuda.get_device_capability())
    checkpoint, source_info, restored = None, None, None
    if resume_from:
        relative, source_info = _checkpoint_source(config, resume_from)
        if source_info["slot"] != slot or source_info["cutoff"] != cutoff.date().isoformat():
            raise ValueError("CHECKPOINT_INCOMPATIBLE: training phase")
        source_root = resolve_run(config, resume_from)
        source_inputs = (
            source_root / "inputs/synthetic" if config.data.source == "synthetic" else None
        )
        # A resumed synthetic predecessor can itself reference an older immutable input tree.
        if source_inputs is not None and not source_inputs.is_dir():
            ancestor = source_root
            while not source_inputs.is_dir():
                predecessor = read_json(safe_path(ancestor, "run.json", regular=True))[
                    "resumed_from_run_id"
                ]
                if predecessor is None:
                    raise ValueError("Resume source inputs are unavailable")
                ancestor = resolve_run(config, predecessor)
                source_inputs = ancestor / "inputs/synthetic"
        run = resume_run(
            config,
            source_run_id=resume_from,
            kind="training",
            validate_checkpoint=lambda _root, _manifest: relative,
            run_id=run_id,
            input_dir=source_inputs,
        )
        checkpoint = safe_path(source_root, relative, regular=True).read_bytes()
        restored = unpack_checkpoint(checkpoint)
    else:
        run = allocate_run(
            config,
            kind="training",
            run_id=run_id,
            parent_run_ids=tuple(x["run_id"] for x in completed),
        )
    try:
        active = (
            replace(config, paths=replace(config.paths, raw_data_dir=run.input_dir))
            if resume_from
            else _stage_inputs(run, config)
        )
        frames = _verified_frames(run)
        supported = _supported_series(run)
        context_record = {
            "slot": slot,
            "cutoff": cutoff.date().isoformat(),
            "completed_folds": completed,
        }
        write_json_artifact(run, "recurrent/context.json", context_record, kind="training_context")
        spec = FeatureSpec("asof_v1")

        def inputs(at: pd.Timestamp):
            return build_training_set(
                frames,
                cutoff=at,
                spec=spec,
                config=config,
                snapshot_id=run.manifest["snapshot_id"],
                data_source=config.data.source,
                supported_series=supported,
                needs_batch=False,
                needs_sequences=True,
                params=params,
                name=f"gru/{slot}",
            )

        internal_cutoff = cutoff - pd.Timedelta(days=16)
        stages = []
        prior_seconds = 0.0
        # Checkpoint metadata accounts for completed phases and inherited attempts.
        if restored is not None:
            prior_seconds = restored["identity"]["prior_training_seconds"]
        earlier_seconds = sum(x["training_seconds"] for x in completed)
        predecessor_wall = []
        recovery_overhead = 0.0
        inherited_host_peak = 0
        inherited_gpu_peak = 0
        ancestor = resume_from
        while ancestor:
            record = read_json(safe_path(resolve_run(config, ancestor), "run.json", regular=True))
            validate_manifest(record)
            _verify_outputs(resolve_run(config, ancestor), record)
            failures = [
                r for r in record["artifacts"] if r["path"] == "recurrent/failed-costs.json"
            ]
            if failures:
                failure = read_json(
                    safe_path(
                        resolve_run(config, ancestor), "recurrent/failed-costs.json", regular=True
                    )
                )
                recovery_overhead += failure.get("uncheckpointed_training_seconds") or 0.0
                inherited_host_peak = max(inherited_host_peak, failure["host_peak_bytes"])
                inherited_gpu_peak = max(inherited_gpu_peak, failure["gpu_peak_bytes"] or 0)
            from datetime import datetime

            wall = None
            if record["finished_at_utc"] is not None:
                wall = (
                    datetime.fromisoformat(record["finished_at_utc"])
                    - datetime.fromisoformat(record["started_at_utc"])
                ).total_seconds()
            predecessor_wall.append(
                {
                    "run_id": ancestor,
                    "wall_seconds": wall,
                    "unmeasured_partial_epoch_possible": True,
                }
            )
            ancestor = record["resumed_from_run_id"]
        if restored is not None:
            restored["host_peak_bytes"] = max(restored["host_peak_bytes"], inherited_host_peak)
            restored["gpu_peak_bytes"] = max(restored["gpu_peak_bytes"], inherited_gpu_peak)
            checkpoint = pack_checkpoint(restored)
        prior_attempts = []
        if resume_from:
            prior_attempts = [r["run_id"] for r in predecessor_wall]
        for stage, at in (("selection", internal_cutoff), ("refit", cutoff)):
            current_stage_result = None
            training, grid = inputs(at)
            pre = SequencePreprocessor.fit(training.sequences, cutoff=at)
            validation = None
            if stage == "selection":
                validation_panel = request_panel(training.request)
                target_dates = training.request.target_dates
                labels = benchmark_labels(frames["train.csv"], target_dates)
                index = pd.MultiIndex.from_tuples(
                    [
                        (
                            training.request.series_keys[s][0],
                            training.request.series_keys[s][1],
                            validation_panel.dates[o + h],
                        )
                        for s, o in validation_panel.keys
                        for h in range(1, 17)
                    ],
                    names=["store_nbr", "family", "target_date"],
                )
                actual = labels.set_index(["store_nbr", "family", "target_date"])["sales"].reindex(
                    index
                )
                if actual.isna().any():
                    raise ValueError("Incomplete internal validation labels")
                validation = validation_panel, np.log1p(actual.to_numpy()).reshape(-1, 16)
                epochs = params["max_epochs"]
            else:
                epochs = stages[0]["selected_epochs"]
            identity = {
                "adapter": "gru_global",
                "slot": slot,
                "stage": stage,
                "cutoff": at.date().isoformat(),
                "config": run.manifest["config_sha256"],
                "dataset": run.manifest["snapshot_id"],
                "sample": training.sequences.identity,
                "preprocessor": pre.sha256,
                "code": run.manifest["code"],
                "versions": run.manifest["versions"],
                "precision": "float32",
                "source_sha256": digest_json(
                    {
                        p.relative_to(Path(__file__).parent).as_posix(): hashlib.sha256(
                            p.read_bytes()
                        ).hexdigest()
                        for p in sorted(Path(__file__).parent.rglob("*.py"))
                    }
                ),
                "device": params["device"],
                "hardware": hardware,
                "epochs": epochs,
                "prior_training_seconds": prior_seconds,
            }
            stage_resume = None
            if restored is not None:
                source_stage = restored["identity"]["stage"]
                if stage == "selection" and source_stage == "refit":
                    selection = read_json(
                        safe_path(
                            resolve_run(config, resume_from),
                            f"recurrent/{slot}/selection.json",
                            regular=True,
                        )
                    )
                    stages.append(selection)
                    write_json_artifact(
                        run, f"recurrent/{slot}/selection.json", selection, kind="selection"
                    )
                    # Identity time at the refit checkpoint includes selection already.
                    continue
                if source_stage == stage:
                    stage_resume = checkpoint
                    if identity != restored["identity"]:
                        raise ValueError("CHECKPOINT_INCOMPATIBLE")

            def on_epoch(
                epoch: int,
                payload: bytes,
                *,
                phase: str = stage,
                before_seconds: float = prior_seconds,
            ) -> None:
                state = unpack_checkpoint(payload)
                write_run_artifact(
                    run,
                    f"checkpoints/{slot}/{phase}/epoch_{epoch:04d}.zip",
                    lambda path: path.write_bytes(payload),
                    kind="checkpoint",
                )
                used = (
                    earlier_seconds + before_seconds + recovery_overhead + state["training_seconds"]
                )
                if time_limit is not None and used > time_limit:
                    raise RuntimeError("GRU_STUDY_TIME_BUDGET_EXCEEDED")
                # Linux ru_maxrss is KiB; measured process peak includes panel construction.
                if (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                    > params["host_memory_gib"] * 2**30
                ):
                    raise RuntimeError("GRU_HOST_MEMORY_EXCEEDED")

            model = initialize(pre, params["hidden_size"], config.runtime.seed)
            result = fit_epochs(
                model,
                training.sequences,
                pre,
                params=params,
                seed=config.runtime.seed,
                epochs=epochs,
                identity=identity,
                validation=validation,
                resume=stage_resume,
                on_epoch=on_epoch,
            )
            current_stage_result = result
            result.update(
                {
                    "stage": stage,
                    "grid": grid,
                    "preprocessor_sha256": pre.sha256,
                    "fit_rows_sha256": pre.state["fit_rows_sha256"],
                    "fit_cutoff": at.date().isoformat(),
                }
            )
            write_json_artifact(run, f"recurrent/{slot}/{stage}.json", result, kind="selection")
            stages.append(result)
            prior_seconds += result["training_seconds"]
        if (
            time_limit is not None
            and earlier_seconds + prior_seconds + recovery_overhead > time_limit
        ):
            raise RuntimeError("GRU_STUDY_TIME_BUDGET_EXCEEDED")
        adapter = GRUAdapter()
        ctx = _context(run, cutoff, config, epochs)
        fitted = adapter.fitted(model, pre, training, ctx)
        manifest = save_bundle(fitted, run=run, context=ctx, slot=slot)
        relative = f"models/gru_global/{slot}"
        loaded = load_bundle(
            BundleRef("running", run.root, relative), expected_policy="asof_v1", running=run
        )
        expected = adapter.predict_raw(fitted, training.request)
        actual = adapter.predict_raw(loaded.fitted, training.request)
        parity = compare_raw_predictions(expected, actual, "gru_global")
        if not parity["passed"]:
            raise ValueError("RELOAD_PARITY_FAILED")
        write_json_artifact(run, f"checks/{slot}-parity.json", parity, kind="verification")
        write_run_artifact(
            run,
            f"checks/{slot}-fit-predictions.csv",
            lambda path: expected.to_csv(path, index=False, float_format="%.17g"),
            kind="verification",
        )
        write_json_artifact(
            run, "dataset_summary.json", validate_dataset(active).to_dict(), kind="dataset_summary"
        )
        costs = {
            "training_seconds": prior_seconds + recovery_overhead,
            "completed_epoch_training_seconds": prior_seconds,
            "discarded_recovery_training_seconds": recovery_overhead,
            "stages": stages,
            "host_peak_bytes": max(x["host_peak_bytes"] for x in stages),
            "attempt_wall_seconds": time.monotonic() - attempt_started,
            "device": params["device"],
            "precision": "float32",
            "hardware": hardware,
            "prior_attempts": prior_attempts,
            "predecessor_wall": predecessor_wall,
            "training_seconds_method": "Epochs plus continuation; partial tail unmeasured",
        }
        write_json_artifact(run, "recurrent/costs.json", costs, kind="costs")
        finish_run(run)
        return {
            **run_result(run),
            "slot": slot,
            "cutoff": cutoff.date().isoformat(),
            "bundle": f"{run.run_id}:{relative}",
            "bundle_id": manifest["bundle_id"],
            "bundle_dir": str(run.root / relative),
            "training_seconds": prior_seconds + recovery_overhead,
            "completed_epoch_training_seconds": prior_seconds,
            "discarded_recovery_training_seconds": recovery_overhead,
            "selected_epochs": epochs,
            "stages": stages,
            "costs": costs,
        }
    except BaseException as error:
        if run.manifest["state"] == "RUNNING":
            measured = getattr(error, "training_resources", {})
            if not measured and locals().get("current_stage_result") is not None:
                measured = {"stage_training_seconds": current_stage_result["training_seconds"]}
            listed = [r["path"] for r in run.manifest["artifacts"] if r["kind"] == "checkpoint"]
            checkpointed_seconds = 0.0
            if listed:
                last = unpack_checkpoint(safe_path(run.root, listed[-1], regular=True).read_bytes())
                if last["identity"]["stage"] == locals().get("stage"):
                    checkpointed_seconds = last["training_seconds"]
            elif restored is not None and restored["identity"]["stage"] == locals().get("stage"):
                checkpointed_seconds = restored["training_seconds"]
            uncheckpointed = None
            if "stage_training_seconds" in measured:
                uncheckpointed = max(0.0, measured["stage_training_seconds"] - checkpointed_seconds)
            write_json_artifact(
                run,
                "recurrent/failed-costs.json",
                {
                    "attempt_wall_seconds": time.monotonic() - attempt_started,
                    "host_peak_bytes": max(
                        measured.get("host_peak_bytes", 0),
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    ),
                    "gpu_peak_bytes": max(
                        measured.get("gpu_peak_bytes", 0),
                        __import__("torch").cuda.max_memory_allocated(),
                    )
                    if params["device"] == "cuda"
                    else measured.get("gpu_peak_bytes", 0),
                    "uncheckpointed_training_seconds": uncheckpointed,
                    "partial_epoch_tail_unmeasured": uncheckpointed is None,
                    "resumed_from_run_id": resume_from,
                },
                kind="costs",
            )
            finish_run(run, state="FAILED", error_code="EXECUTION_FAILED")
        raise


def train_recurrent(
    config: ProjectConfig,
    *,
    origin: date | None = None,
    run_id: str | None = None,
    resume_from: str | None = None,
) -> dict[str, Any]:
    """Explicit final-origin train command; does not promote the GRU."""
    _recipe(config)
    if origin is None:
        if resume_from:
            _, info = _checkpoint_source(config, resume_from)
            origin = pd.Timestamp(info["cutoff"]).date()
        elif config.data.source == "synthetic":
            origin = (
                pd.Timestamp(config.data.fixture_start_date)
                + pd.Timedelta(days=config.data.fixture_history_days - 1)
            ).date()
        else:
            origin = pd.read_csv(config.paths.raw_data_dir / "train.csv", usecols=["date"])[
                "date"
            ].max()
    return _train_one(
        config,
        cutoff=pd.Timestamp(origin),
        slot="final",
        completed=[],
        run_id=run_id,
        resume_from=resume_from,
    )
