"""Non-neural experiment protocol tests."""

import pandas as pd

from retail_forecast.experiments import (
    MODEL_IDS,
    segmented_rmsle,
    temporal_regime,
    training_origins,
)
from retail_forecast.splits import RollingOriginFold


def test_experiment_budget_has_one_tree_and_at_most_ten_configurations() -> None:
    assert len(MODEL_IDS) == 4
    assert len(MODEL_IDS) <= 10
    assert [model for model in MODEL_IDS if model == "lightgbm_global"] == ["lightgbm_global"]


def test_training_origins_end_before_validation() -> None:
    fold = RollingOriginFold(
        name="fold_1",
        train_start=pd.Timestamp("2020-01-01"),
        origin=pd.Timestamp("2020-03-01"),
        train_end=pd.Timestamp("2020-03-01"),
        validation_start=pd.Timestamp("2020-03-02"),
        validation_end=pd.Timestamp("2020-03-17"),
        horizon=16,
    )
    origins = training_origins(fold, pd.Timestamp("2020-01-01"), stride_days=16, max_origins=3)

    assert origins == tuple(sorted(origins))
    assert origins[-1] + pd.offsets.Day(16) == fold.origin
    assert len(origins) == 3


def test_temporal_regimes_are_mutually_exclusive() -> None:
    frame = pd.DataFrame(
        {
            "target_date": ["2020-02-15", "2020-02-16", "2020-02-17"],
            "is_holiday_event": [1, 0, 0],
        }
    )
    assert temporal_regime(frame).tolist() == ["holiday_event", "weekend", "regular_weekday"]


def test_segmented_results_include_every_required_scope() -> None:
    oof = pd.DataFrame(
        {
            "model_id": ["zero", "zero"],
            "fold": ["fold_1", "fold_1"],
            "horizon": [1, 2],
            "store_nbr": [1, 1],
            "family": ["A", "A"],
            "promotion_regime": ["not_promoted", "promoted"],
            "temporal_regime": ["regular_weekday", "weekend"],
            "y_true": [1.0, 2.0],
            "y_pred": [0.0, 0.0],
        }
    )
    results = segmented_rmsle(oof)

    assert set(results["scope"]) == {
        "global",
        "fold",
        "horizon",
        "store",
        "family",
        "promotion",
        "temporal_regime",
    }
