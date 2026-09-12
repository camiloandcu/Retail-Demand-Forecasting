"""Reusable acquisition and EDA artifact tests."""

import json
from pathlib import Path

from retail_forecast.acquisition import download_competition_data
from retail_forecast.config import ProjectConfig
from retail_forecast.eda import (
    build_eda_summary,
    build_eda_tables,
    build_manifest,
    export_eda_artifacts,
    load_eda_frames,
)


def test_existing_dataset_skips_network(synthetic_config: ProjectConfig) -> None:
    """Complete local inputs must make acquisition idempotent without Kaggle auth."""

    result = download_competition_data(synthetic_config.paths.raw_data_dir)
    assert result == synthetic_config.paths.raw_data_dir


def test_eda_summary_and_manifest_export(synthetic_config: ProjectConfig) -> None:
    """Smoke EDA exports machine-readable facts without changing fixture files."""

    data_dir = synthetic_config.paths.raw_data_dir
    frames = load_eda_frames(data_dir)
    before = build_manifest(data_dir, frames)
    tables = build_eda_tables(frames)
    summary = build_eda_summary(frames, tables, source="synthetic")
    summary_path, manifest_path = export_eda_artifacts(
        summary,
        before,
        synthetic_config.paths.artifacts_dir / "metrics",
    )
    after = build_manifest(data_dir, frames)

    exported = json.loads(summary_path.read_text(encoding="utf-8"))
    assert exported["synthetic"] is True
    assert exported["observations"]["test_horizon_days"] == 16
    assert manifest_path.is_file()
    assert before == after
    assert set(tables) == {
        "daily_sales",
        "store_sales",
        "family_sales",
        "promotion",
        "sales_transactions",
        "oil_sales",
        "holiday_types",
        "weekday",
        "month",
        "annual_cycle",
        "year",
    }


def test_figure_export_rejects_non_png(tmp_path: Path) -> None:
    """Keep the notebook's selected figure format explicit."""

    from retail_forecast.eda import save_figure

    class FigureStub:
        def savefig(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("savefig should not run for an invalid extension")

    try:
        save_figure(FigureStub(), tmp_path, "plot.jpg")
    except ValueError as error:
        assert "PNG" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("A non-PNG EDA figure must be rejected")
