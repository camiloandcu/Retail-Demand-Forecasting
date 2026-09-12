"""Dataset contracts and deterministic schema-only fixture generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from retail_forecast.config import ProjectConfig
from retail_forecast.io_utils import write_json

FILE_COLUMNS: Final[dict[str, list[str]]] = {
    "train.csv": ["id", "date", "store_nbr", "family", "sales", "onpromotion"],
    "test.csv": ["id", "date", "store_nbr", "family", "onpromotion"],
    "sample_submission.csv": ["id", "sales"],
    "stores.csv": ["store_nbr", "city", "state", "type", "cluster"],
    "oil.csv": ["date", "dcoilwtico"],
    "holidays_events.csv": [
        "date",
        "type",
        "locale",
        "locale_name",
        "description",
        "transferred",
    ],
    "transactions.csv": ["date", "store_nbr", "transactions"],
}

EXPECTED_KAGGLE_SHA256: Final[dict[str, str]] = {
    "holidays_events.csv": "81a183d6c4d691b57f84a0fde6bbf734a5b3c36a74b97378bf33a592648e2999",
    "oil.csv": "944b23b857580f9d804399346fd3ed69bffcb7facfd98c55fdb408b8d057cca7",
    "sample_submission.csv": "17505ec561d9bc64a3c10c3eb00474becf162d9053cebf014a58a6c82ed530c4",
    "stores.csv": "af503b2bce11d7906d249f81cc0598f10e2addcc9f6c59aa2d95c9f2652c296b",
    "test.csv": "087ec1ecec76dbba9ea2adb48b8b2fd527c7c59adb82b0dfb9f6f82739400d26",
    "train.csv": "99a1b7f4241821df10fa71f23abdd71c634f71f901e3cab1400c0f0c583f862c",
    "transactions.csv": "e116384a6981af74932832436aa2f6a43121f77ca81accc44e3a5160158ca03c",
}


class DataContractError(ValueError):
    """Raised when source files violate the expected Kaggle data contract."""


@dataclass(frozen=True)
class DatasetSummary:
    """Validated panel dimensions used in logs and artifacts."""

    train_rows: int
    test_rows: int
    stores: int
    families: int
    series: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    horizon: int
    source: str

    def to_dict(self) -> dict[str, int | str]:
        """Return a JSON-safe representation."""

        return asdict(self)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise DataContractError(f"Missing required file: {path}")
    return pd.read_csv(path)


def load_frames(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load every required competition CSV with exact column order checks."""

    frames: dict[str, pd.DataFrame] = {}
    for filename, expected_columns in FILE_COLUMNS.items():
        frame = _read_csv(data_dir / filename)
        if list(frame.columns) != expected_columns:
            raise DataContractError(
                f"{filename} columns must be {expected_columns}; got {list(frame.columns)}"
            )
        frames[filename] = frame
    return frames


def required_files_exist(data_dir: Path) -> bool:
    """Return whether all seven expected CSVs already exist."""

    return all((data_dir / filename).is_file() for filename in FILE_COLUMNS)


def _dates(frame: pd.DataFrame, filename: str) -> pd.Series:
    try:
        return pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    except (TypeError, ValueError) as error:
        raise DataContractError(f"{filename} contains invalid ISO dates") from error


def validate_dataset(config: ProjectConfig) -> DatasetSummary:
    """Validate schema, temporal boundaries, panel keys, and submission identity."""

    frames = load_frames(config.paths.raw_data_dir)
    train = frames["train.csv"].copy()
    test = frames["test.csv"].copy()
    stores = frames["stores.csv"]
    oil = frames["oil.csv"]
    transactions = frames["transactions.csv"]
    sample = frames["sample_submission.csv"]

    train["date"] = _dates(train, "train.csv")
    test["date"] = _dates(test, "test.csv")
    oil_dates = _dates(oil, "oil.csv")
    transaction_dates = _dates(transactions, "transactions.csv")

    key = ["date", "store_nbr", "family"]
    if train.duplicated(key).any() or test.duplicated(key).any():
        raise DataContractError("Store-family-date keys must be unique in train and test")
    if train["id"].duplicated().any() or test["id"].duplicated().any():
        raise DataContractError("IDs must be unique in train and test")
    if train["sales"].isna().any() or (train["sales"] < 0).any():
        raise DataContractError("Training sales must be present and non-negative")
    if train["onpromotion"].isna().any() or test["onpromotion"].isna().any():
        raise DataContractError("Promotion values must be present")
    if (train["onpromotion"] < 0).any() or (test["onpromotion"] < 0).any():
        raise DataContractError("Promotion values must be non-negative")

    train_stores = set(train["store_nbr"])
    test_stores = set(test["store_nbr"])
    train_families = set(train["family"])
    test_families = set(test["family"])
    if train_stores != test_stores or train_families != test_families:
        raise DataContractError("Train and test must contain the same stores and families")
    if len(train_stores) != config.data.expected_stores:
        raise DataContractError("Observed store count does not match configuration")
    if len(train_families) != config.data.expected_families:
        raise DataContractError("Observed family count does not match configuration")
    if set(stores["store_nbr"]) != train_stores or stores["store_nbr"].duplicated().any():
        raise DataContractError("stores.csv must map each panel store exactly once")

    test_dates = pd.DatetimeIndex(sorted(test["date"].unique()))
    expected_dates = pd.date_range(
        train["date"].max() + pd.offsets.Day(1),
        periods=config.forecast.horizon,
        freq="D",
    )
    if not test_dates.equals(expected_dates):
        raise DataContractError("Test dates must be the contiguous 16-day period after train")
    expected_test_rows = len(train_stores) * len(train_families) * config.forecast.horizon
    if len(test) != expected_test_rows:
        raise DataContractError("Test must contain the complete store-family-horizon grid")
    if oil_dates.max() < test["date"].max():
        raise DataContractError("oil.csv must cover the Kaggle test period")
    if transaction_dates.max() > train["date"].max():
        raise DataContractError("Future transactions are not part of the Kaggle contract")

    if list(sample["id"]) != list(test["id"]):
        raise DataContractError("sample_submission IDs and order must match test.csv")
    if sample["sales"].isna().any() or (sample["sales"] < 0).any():
        raise DataContractError("Sample submission sales must be non-negative")

    return DatasetSummary(
        train_rows=len(train),
        test_rows=len(test),
        stores=len(train_stores),
        families=len(train_families),
        series=len(train_stores) * len(train_families),
        train_start=train["date"].min().date().isoformat(),
        train_end=train["date"].max().date().isoformat(),
        test_start=test["date"].min().date().isoformat(),
        test_end=test["date"].max().date().isoformat(),
        horizon=len(test_dates),
        source=config.data.source,
    )


def write_synthetic_dataset(config: ProjectConfig) -> Path:
    """Generate deterministic CSVs that mimic schemas, never real observations."""

    if config.data.source != "synthetic":
        raise DataContractError("Synthetic fixtures can only be generated for source=synthetic")
    output_dir = config.paths.raw_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stores = list(range(1, config.data.expected_stores + 1))
    families = [
        f"SYNTHETIC_FAMILY_{index:02d}" for index in range(1, config.data.expected_families + 1)
    ]
    train_dates = pd.date_range(
        config.data.fixture_start_date, periods=config.data.fixture_history_days, freq="D"
    )
    test_dates = pd.date_range(
        train_dates.max() + pd.offsets.Day(1), periods=config.forecast.horizon, freq="D"
    )

    train_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    row_id = 0
    for day_index, current_date in enumerate(train_dates):
        for store in stores:
            for family_index, family in enumerate(families, start=1):
                weekly = 4.0 if current_date.dayofweek in {4, 5} else 0.0
                sales = float((day_index + store * 3 + family_index * 5) % 23) + weekly + 0.5
                train_rows.append(
                    {
                        "id": row_id,
                        "date": current_date.date().isoformat(),
                        "store_nbr": store,
                        "family": family,
                        "sales": sales,
                        "onpromotion": (day_index + store + family_index) % 4,
                    }
                )
                row_id += 1
    for day_index, current_date in enumerate(test_dates):
        for store in stores:
            for family_index, family in enumerate(families, start=1):
                test_rows.append(
                    {
                        "id": row_id,
                        "date": current_date.date().isoformat(),
                        "store_nbr": store,
                        "family": family,
                        "onpromotion": (day_index + store + family_index) % 4,
                    }
                )
                row_id += 1

    train = pd.DataFrame(train_rows, columns=FILE_COLUMNS["train.csv"])
    test = pd.DataFrame(test_rows, columns=FILE_COLUMNS["test.csv"])
    store_frame = pd.DataFrame(
        [
            {
                "store_nbr": store,
                "city": f"SYNTHETIC_CITY_{store}",
                "state": f"SYNTHETIC_STATE_{(store - 1) % 2 + 1}",
                "type": "SYNTHETIC_TYPE_A" if store % 2 else "SYNTHETIC_TYPE_B",
                "cluster": store,
            }
            for store in stores
        ],
        columns=FILE_COLUMNS["stores.csv"],
    )

    all_dates = pd.date_range(train_dates.min(), test_dates.max(), freq="D")
    oil = pd.DataFrame(
        {
            "date": [value.date().isoformat() for value in all_dates],
            "dcoilwtico": [
                np.nan if value.dayofweek >= 5 else 50.0 + (index % 11)
                for index, value in enumerate(all_dates)
            ],
        },
        columns=FILE_COLUMNS["oil.csv"],
    )
    transactions = pd.DataFrame(
        [
            {
                "date": current_date.date().isoformat(),
                "store_nbr": store,
                "transactions": 100 + day_index * 2 + store,
            }
            for day_index, current_date in enumerate(train_dates)
            for store in stores
        ],
        columns=FILE_COLUMNS["transactions.csv"],
    )
    holiday_date = test_dates[2].date().isoformat()
    holidays = pd.DataFrame(
        [
            {
                "date": holiday_date,
                "type": "Holiday",
                "locale": "National",
                "locale_name": "Synthetic Country",
                "description": "Synthetic national holiday",
                "transferred": False,
            },
            {
                "date": holiday_date,
                "type": "Event",
                "locale": "Local",
                "locale_name": "SYNTHETIC_CITY_1",
                "description": "Synthetic local event",
                "transferred": False,
            },
        ],
        columns=FILE_COLUMNS["holidays_events.csv"],
    )
    sample = pd.DataFrame(
        {"id": test["id"].astype(int), "sales": np.zeros(len(test), dtype=float)},
        columns=FILE_COLUMNS["sample_submission.csv"],
    )

    frames = {
        "train.csv": train,
        "test.csv": test,
        "sample_submission.csv": sample,
        "stores.csv": store_frame,
        "oil.csv": oil,
        "holidays_events.csv": holidays,
        "transactions.csv": transactions,
    }
    for filename, frame in frames.items():
        frame.to_csv(output_dir / filename, index=False)
    write_json(
        output_dir / "SYNTHETIC_FIXTURE.json",
        {
            "synthetic": True,
            "seed": config.runtime.seed,
            "purpose": "schema and pipeline testing only; contains no Kaggle observations",
        },
    )
    return output_dir
