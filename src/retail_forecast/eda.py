"""Reusable manifest, EDA summaries, tables, and artifact export."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from retail_forecast.data import EXPECTED_KAGGLE_SHA256, FILE_COLUMNS, load_frames
from retail_forecast.io_utils import write_json


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it entirely into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_kaggle_hashes(data_dir: Path) -> dict[str, str]:
    """Require every raw Kaggle CSV to match the audited SHA-256 manifest."""

    observed = {filename: sha256_file(data_dir / filename) for filename in FILE_COLUMNS}
    mismatches = {
        filename: {"expected": EXPECTED_KAGGLE_SHA256[filename], "observed": digest}
        for filename, digest in observed.items()
        if digest != EXPECTED_KAGGLE_SHA256[filename]
    }
    if mismatches:
        raise ValueError(f"Kaggle file integrity check failed: {mismatches}")
    return observed


def load_eda_frames(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load validated schemas and parse every date column without changing raw files."""

    frames = load_frames(data_dir)
    for frame in frames.values():
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    return frames


def build_manifest(
    data_dir: Path, frames: dict[str, pd.DataFrame], include_hashes: bool = True
) -> dict[str, Any]:
    """Describe physical files, schemas, time ranges, and missingness."""

    manifest: dict[str, Any] = {}
    for filename in FILE_COLUMNS:
        frame = frames[filename]
        entry: dict[str, Any] = {
            "bytes": (data_dir / filename).stat().st_size,
            "rows": len(frame),
            "columns": list(frame.columns),
            "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
            "missing": {column: int(value) for column, value in frame.isna().sum().items()},
        }
        if "date" in frame.columns:
            entry["date_min"] = frame["date"].min().date().isoformat()
            entry["date_max"] = frame["date"].max().date().isoformat()
            entry["unique_dates"] = int(frame["date"].nunique())
        if include_hashes:
            entry["sha256"] = sha256_file(data_dir / filename)
        manifest[filename] = entry
    return manifest


def build_eda_tables(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Build compact aggregates used by every EDA view."""

    train = frames["train.csv"]
    transactions = frames["transactions.csv"]
    oil = frames["oil.csv"]
    holidays = frames["holidays_events.csv"]

    daily = train.groupby("date", as_index=False)["sales"].sum().sort_values("date")
    daily["rolling_28d"] = daily["sales"].rolling(28, min_periods=7).median()
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["weekday"] = daily["date"].dt.dayofweek
    daily["dayofyear"] = daily["date"].dt.dayofyear

    store = (
        train.groupby("store_nbr", as_index=False)["sales"]
        .sum()
        .sort_values("sales", ascending=False)
    )
    family = train.groupby("family", as_index=False).agg(
        sales=("sales", "sum"), zero_rate=("sales", lambda values: float((values == 0).mean()))
    )
    family = family.sort_values("sales", ascending=False)
    promotion = (
        train.assign(promoted=train["onpromotion"].gt(0))
        .groupby("promoted", as_index=False)
        .agg(
            rows=("sales", "size"),
            mean_sales=("sales", "mean"),
            median_sales=("sales", "median"),
            zero_rate=("sales", lambda values: float((values == 0).mean())),
        )
    )
    daily_transactions = transactions.groupby("date", as_index=False)["transactions"].sum()
    sales_transactions = daily.merge(daily_transactions, on="date", how="inner")
    oil_train = oil.loc[oil["date"] <= train["date"].max()].copy()
    oil_sales = daily.merge(oil_train, on="date", how="left")
    holiday_types = (
        holidays.groupby("type", as_index=False).size().sort_values("size", ascending=False)
    )
    weekday = daily.groupby("weekday", as_index=False)["sales"].mean()
    month = daily.groupby("month", as_index=False)["sales"].mean()
    annual_cycle = daily.groupby("dayofyear", as_index=False)["sales"].mean()
    year = daily.groupby("year", as_index=False).agg(
        mean_daily_sales=("sales", "mean"),
        median_daily_sales=("sales", "median"),
        observed_days=("date", "size"),
    )
    return {
        "daily_sales": daily,
        "store_sales": store,
        "family_sales": family,
        "promotion": promotion,
        "sales_transactions": sales_transactions,
        "oil_sales": oil_sales,
        "holiday_types": holiday_types,
        "weekday": weekday,
        "month": month,
        "annual_cycle": annual_cycle,
        "year": year,
    }


def build_eda_summary(
    frames: dict[str, pd.DataFrame], tables: dict[str, pd.DataFrame], source: str
) -> dict[str, Any]:
    """Create JSON-ready observations, inferences, hypotheses, and design effects."""

    train = frames["train.csv"]
    test = frames["test.csv"]
    transactions = frames["transactions.csv"]
    oil = frames["oil.csv"]
    holidays = frames["holidays_events.csv"]
    calendar = pd.date_range(train["date"].min(), train["date"].max(), freq="D")
    missing_dates = calendar.difference(pd.DatetimeIndex(train["date"].unique()))
    quantiles = train["sales"].quantile([0.5, 0.9, 0.95, 0.99, 0.999, 1.0])
    q1, q3 = train["sales"].quantile([0.25, 0.75])
    iqr_threshold = float(q3 + 1.5 * (q3 - q1))
    transaction_pairs = transactions[["date", "store_nbr"]].drop_duplicates()
    possible_pairs = int(train["date"].nunique() * train["store_nbr"].nunique())
    promoted = tables["promotion"].set_index("promoted")
    promoted_mean = float(promoted.loc[True, "mean_sales"]) if True in promoted.index else None
    regular_mean = float(promoted.loc[False, "mean_sales"]) if False in promoted.index else None
    daily = tables["daily_sales"]
    first_window = float(daily.head(min(90, len(daily)))["sales"].mean())
    last_window = float(daily.tail(min(90, len(daily)))["sales"].mean())

    return {
        "source": source,
        "synthetic": source == "synthetic",
        "observations": {
            "train_rows": len(train),
            "test_rows": len(test),
            "stores": int(train["store_nbr"].nunique()),
            "families": int(train["family"].nunique()),
            "series": int(train[["store_nbr", "family"]].drop_duplicates().shape[0]),
            "train_date_range": [
                train["date"].min().date().isoformat(),
                train["date"].max().date().isoformat(),
            ],
            "test_date_range": [
                test["date"].min().date().isoformat(),
                test["date"].max().date().isoformat(),
            ],
            "test_horizon_days": int(test["date"].nunique()),
            "zero_sales_rate": float(train["sales"].eq(0).mean()),
            "fractional_sales_rows": int(train["sales"].mod(1).ne(0).sum()),
            "sales_quantiles": {str(level): float(value) for level, value in quantiles.items()},
            "iqr_outlier_threshold": iqr_threshold,
            "iqr_outlier_candidates": int(train["sales"].gt(iqr_threshold).sum()),
            "missing_calendar_dates": [value.date().isoformat() for value in missing_dates],
            "promotion_mean_sales": promoted_mean,
            "nonpromotion_mean_sales": regular_mean,
            "transaction_store_date_coverage": float(len(transaction_pairs) / possible_pairs),
            "oil_missing_values": int(oil["dcoilwtico"].isna().sum()),
            "holiday_rows": len(holidays),
            "holiday_unique_dates": int(holidays["date"].nunique()),
            "holiday_duplicate_date_rows": int(holidays.duplicated("date", keep=False).sum()),
            "first_90d_mean_total_sales": first_window,
            "last_90d_mean_total_sales": last_window,
            "last_to_first_90d_ratio": float(last_window / first_window) if first_window else None,
        },
        "inferences": [
            (
                "Use log1p-compatible objectives and robust diagnostics because the sales "
                "distribution is right-skewed."
            ),
            (
                "Keep zero sales distinct from missing dates; absence of a row is not evidence "
                "of zero demand."
            ),
            (
                "Preserve weekly calendar signals and evaluate every one of the 16 horizons "
                "separately."
            ),
            (
                "Fit imputers, scalers, encoders, and aggregates inside each rolling-origin "
                "training fold."
            ),
            (
                "Resolve holiday locale, transfer semantics, and duplicate dates before joining "
                "to the sales panel."
            ),
        ],
        "hypotheses": [
            (
                "Promotion is associated with higher sales, but this EDA cannot identify a "
                "causal lift."
            ),
            (
                "Store and family scale differences may benefit from categorical "
                "representations and segmented diagnostics."
            ),
            (
                "Long-run level changes may require recent-window features or fold-specific "
                "normalization."
            ),
            (
                "Oil and transaction associations may be indirect and must not be interpreted "
                "causally."
            ),
        ],
        "model_design_effects": [
            "Use a 16-step direct forecast matching the test grid.",
            (
                "Retain a recursive weekly seasonal-naive baseline that never reads validation "
                "targets."
            ),
            "Report RMSLE globally, by horizon, store, and family.",
            (
                "Use only future covariates explicitly supplied by Kaggle; future sales and "
                "transactions remain unavailable."
            ),
        ],
    }


def export_eda_artifacts(
    summary: dict[str, Any], manifest: dict[str, Any], metrics_dir: Path
) -> tuple[Path, Path]:
    """Export the requested EDA summary and a standalone file manifest."""

    summary_path = metrics_dir / "eda_summary.json"
    manifest_path = metrics_dir / "dataset_manifest.json"
    write_json(summary_path, {**summary, "manifest": manifest})
    write_json(manifest_path, manifest)
    return summary_path, manifest_path


def save_figure(figure: Any, figures_dir: Path, filename: str) -> Path:
    """Save one selected EDA figure with consistent publication settings."""

    if Path(filename).suffix.lower() != ".png":
        raise ValueError("EDA figures must use the PNG extension")
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / filename
    figure.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    return path
