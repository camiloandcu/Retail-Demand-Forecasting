"""Bounded epoch loops and safe, complete epoch-boundary checkpoint state."""

from __future__ import annotations

import io
import json
import random
import resource
import time
import zipfile
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from retail_forecast.io_utils import canonical_json
from retail_forecast.sequences import SequencePanel, SequencePreprocessor


def original_log_loss(pred_z: Any, target_z: Any, center: Any, scale: Any) -> Any:
    """Undo target standardization before MSE so scales never reweight series."""
    pred_log = center + scale * pred_z
    actual_log = center + scale * target_z
    return torch.mean((pred_log - actual_log) ** 2)


def pack_checkpoint(state: Any) -> bytes:
    """Encode tensors plus a typed JSON tree, never executable Python objects."""
    tensors = {}

    def encode(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            key = f"tensor_{len(tensors)}"
            tensors[key] = value.detach().cpu().contiguous()
            return {"tensor": key}
        if isinstance(value, np.ndarray):
            return {"array": value.tolist(), "dtype": str(value.dtype)}
        if isinstance(value, tuple):
            return {"tuple": [encode(v) for v in value]}
        if isinstance(value, list):
            return [encode(v) for v in value]
        if isinstance(value, dict):
            return {"mapping": [[encode(k), encode(v)] for k, v in value.items()]}
        if value is None or type(value) in (str, int, float, bool):
            return value
        raise ValueError("Unsupported checkpoint value")

    metadata = encode(state)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("metadata.json", canonical_json(metadata))
        archive.writestr("tensors.safetensors", save_tensors(tensors))
    return stream.getvalue()


def unpack_checkpoint(data: bytes) -> Any:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if sorted(archive.namelist()) != ["metadata.json", "tensors.safetensors"]:
            raise ValueError("Invalid checkpoint closure")
        metadata = json.loads(archive.read("metadata.json"))
        tensors = load_tensors(archive.read("tensors.safetensors"))

    def decode(value: Any) -> Any:
        if isinstance(value, list):
            return [decode(v) for v in value]
        if isinstance(value, dict):
            if set(value) == {"tensor"}:
                return tensors[value["tensor"]]
            if set(value) == {"array", "dtype"}:
                if value["dtype"] not in {"uint32", "int64", "float64"}:
                    raise ValueError("Invalid checkpoint array dtype")
                return np.asarray(value["array"], dtype=value["dtype"])
            if set(value) == {"tuple"}:
                return tuple(decode(v) for v in value["tuple"])
            if set(value) == {"mapping"}:
                return {decode(k): decode(v) for k, v in value["mapping"]}
            raise ValueError("Invalid checkpoint tree")
        return value

    return decode(metadata)


def predict_z(
    model: Any,
    panel: SequencePanel,
    pre: SequencePreprocessor,
    batch_size: int,
    device: str = "cpu",
) -> np.ndarray:
    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(panel.keys), batch_size):
            batch = pre.batch(panel, np.arange(start, min(start + batch_size, len(panel.keys))))
            x = [torch.as_tensor(batch[k], device=device) for k in ("history", "future", "static")]
            values.append(model(*x).cpu().numpy())
    output = np.concatenate(values)
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite recurrent output")
    return output


def fit_epochs(
    model: Any,
    panel: SequencePanel,
    pre: SequencePreprocessor,
    *,
    params: dict[str, Any],
    seed: int,
    epochs: int,
    identity: dict[str, Any],
    validation: tuple[SequencePanel, np.ndarray] | None = None,
    resume: bytes | None = None,
    on_epoch: Callable[[int, bytes], None] | None = None,
) -> dict[str, Any]:
    """Restore/check state before optimization; checkpoint only completed epochs."""
    device = params["device"]
    torch.set_num_threads(params["cpu_threads"])
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_UNAVAILABLE: select an available reviewed runtime")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    generator = torch.Generator().manual_seed(seed)
    start, history, best, patience, selected = 0, [], None, 0, 0
    elapsed_prior = 0.0
    inherited_host_peak = 0
    inherited_gpu_peak = 0
    if resume is not None:
        state = unpack_checkpoint(resume)
        if state["identity"] != identity:
            raise ValueError("CHECKPOINT_INCOMPATIBLE")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start, history, best = state["epoch"], state["history"], state["best"]
        patience, selected = state["patience"], state["selected_epoch"]
        elapsed_prior = state["training_seconds"]
        inherited_host_peak = state["host_peak_bytes"]
        inherited_gpu_peak = state["gpu_peak_bytes"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        generator.set_state(state["loader_rng"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if not 0 < start <= epochs:
            raise ValueError("Invalid checkpoint epoch")
    started = time.monotonic()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def usage() -> tuple[float, int, int]:
        elapsed = elapsed_prior + time.monotonic() - started
        host_peak = max(
            inherited_host_peak, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        gpu_peak = max(
            inherited_gpu_peak, torch.cuda.max_memory_allocated() if device == "cuda" else 0
        )
        return elapsed, host_peak, gpu_peak

    def check_caps() -> None:
        elapsed, host_peak, gpu_peak = usage()
        if elapsed > params["training_minutes"] * 60:
            raise RuntimeError("GRU_TIME_BUDGET_EXCEEDED")
        if host_peak > params["host_memory_gib"] * 2**30:
            raise RuntimeError("GRU_HOST_MEMORY_EXCEEDED")
        if gpu_peak > params["gpu_memory_gib"] * 2**30:
            raise RuntimeError("GRU_GPU_MEMORY_EXCEEDED")

    try:
        check_caps()
        for epoch in range(start, epochs):
            if validation is not None and patience >= params["patience"]:
                break
            model.train()
            permutation = torch.randperm(len(panel.keys), generator=generator).numpy()
            total, count = 0.0, 0
            for offset in range(0, len(permutation), params["batch_size"]):
                positions = permutation[offset : offset + params["batch_size"]]
                batch = pre.batch(panel, positions)
                if not batch["target_observed"].all():
                    raise ValueError("Training window has missing labels")
                tensors = {k: torch.as_tensor(v, device=device) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                pred = model(*(tensors[k] for k in ("history", "future", "static")))
                loss = original_log_loss(
                    pred, tensors["target_z"], tensors["center"], tensors["scale"]
                )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), params["gradient_clip"], error_if_nonfinite=True
                )
                optimizer.step()
                total += float(loss.detach()) * len(positions) * 16
                count += len(positions) * 16
            row = {"epoch": epoch + 1, "train_log_mse": total / count}
            if validation is not None:
                val_panel, actual_log = validation
                z = predict_z(model, val_panel, pre, params["batch_size"], device)
                values = pre.batch(val_panel, np.arange(len(val_panel.keys)))
                pred_log = values["center"] + values["scale"] * z
                score = float(np.mean((pred_log - actual_log) ** 2))
                if not np.isfinite(score):
                    raise ValueError("Nonfinite internal validation loss")
                row["validation_log_mse"] = score
                if best is None or score < best:
                    best, selected, patience = score, epoch + 1, 0
                else:
                    patience += 1
            else:
                selected = epoch + 1
            history.append(row)
            elapsed, host_peak, gpu_peak = usage()
            # Preserve valid recovery state even when a registered resource cap fails.
            state = {
                "identity": identity,
                "epoch": epoch + 1,
                "history": history,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best": best,
                "patience": patience,
                "selected_epoch": selected,
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "loader_rng": generator.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                "training_seconds": elapsed,
                "host_peak_bytes": host_peak,
                "gpu_peak_bytes": gpu_peak,
            }
            if on_epoch is not None:
                on_epoch(epoch + 1, pack_checkpoint(state))
            check_caps()
        check_caps()
        elapsed, host_peak, gpu_peak = usage()
        return {
            "history": history,
            "selected_epochs": selected,
            "training_seconds": elapsed_prior + time.monotonic() - started,
            "attempt_seconds": time.monotonic() - started,
            "gpu_peak_bytes": gpu_peak if device == "cuda" else None,
            "host_peak_bytes": host_peak,
        }
    except BaseException as error:
        elapsed, host_peak, gpu_peak = usage()
        error.training_resources = {
            "stage_training_seconds": elapsed,
            "attempt_training_seconds": time.monotonic() - started,
            "host_peak_bytes": host_peak,
            "gpu_peak_bytes": gpu_peak,
        }
        raise
