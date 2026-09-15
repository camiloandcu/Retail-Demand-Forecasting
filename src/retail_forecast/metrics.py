"""Official and supporting forecast metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def rmsle(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute Kaggle's unweighted RMSLE for finite, non-negative arrays."""

    actual = np.asarray(y_true, dtype=np.float64)
    predicted = np.asarray(y_pred, dtype=np.float64)
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: y_true={actual.shape}, y_pred={predicted.shape}")
    if actual.size == 0:
        raise ValueError("RMSLE requires at least one observation")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("RMSLE inputs must be finite")
    if (actual < 0).any() or (predicted < 0).any():
        raise ValueError("RMSLE inputs must be non-negative")
    return float(np.sqrt(np.mean(np.square(np.log1p(predicted) - np.log1p(actual)))))


def clip_nonnegative_predictions(values: ArrayLike) -> np.ndarray:
    """Clip raw model outputs at zero before RMSLE or submission generation."""

    predictions = np.asarray(values, dtype=np.float64)
    if predictions.size == 0:
        raise ValueError("Prediction clipping requires at least one value")
    if not np.isfinite(predictions).all():
        raise ValueError("Predictions must be finite before clipping")
    return np.maximum(predictions, 0.0)
