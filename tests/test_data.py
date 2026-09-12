"""Synthetic fixture and dataset contract tests."""

import json

from retail_forecast.config import ProjectConfig
from retail_forecast.data import FILE_COLUMNS, load_frames, validate_dataset


def test_synthetic_fixture_is_labeled_and_matches_schema(
    synthetic_config: ProjectConfig,
) -> None:
    metadata = json.loads(
        (synthetic_config.paths.raw_data_dir / "SYNTHETIC_FIXTURE.json").read_text()
    )
    assert metadata["synthetic"] is True
    assert "no Kaggle observations" in metadata["purpose"]
    frames = load_frames(synthetic_config.paths.raw_data_dir)
    assert set(frames) == set(FILE_COLUMNS)
    summary = validate_dataset(synthetic_config)
    assert summary.horizon == 16
    assert summary.series == 4
    assert summary.source == "synthetic"


def test_future_transactions_are_rejected(synthetic_config: ProjectConfig) -> None:
    transactions_path = synthetic_config.paths.raw_data_dir / "transactions.csv"
    transactions = load_frames(synthetic_config.paths.raw_data_dir)["transactions.csv"]
    transactions.loc[len(transactions)] = ["2099-01-01", 1, 999]
    transactions.to_csv(transactions_path, index=False)

    try:
        validate_dataset(synthetic_config)
    except ValueError as error:
        assert "Future transactions" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Future transactions must violate the contract")
