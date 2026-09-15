"""Reusable acquisition and EDA artifact tests."""

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

import retail_forecast.acquisition as acquisition
from retail_forecast.acquisition import (
    COMPETITION_ARCHIVE,
    acquire_competition_data,
    download_competition_data,
    extract_competition_archive,
    find_competition_archive,
)
from retail_forecast.config import ProjectConfig
from retail_forecast.data import FILE_COLUMNS
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


def test_archive_acquisition_extracts_only_expected_csvs(
    synthetic_config: ProjectConfig, tmp_path: Path
) -> None:
    source_dir = synthetic_config.paths.raw_data_dir
    archive_path = tmp_path / "competition.zip"
    output_dir = tmp_path / "raw"
    with ZipFile(archive_path, "w") as archive:
        for filename in FILE_COLUMNS:
            archive.write(source_dir / filename, arcname=f"nested/{filename}")
        archive.writestr("ignored.txt", "not competition data")

    result = extract_competition_archive(archive_path, output_dir)

    assert result == output_dir
    assert {path.name for path in output_dir.iterdir()} == set(FILE_COLUMNS)


def test_archive_acquisition_rejects_incomplete_zip(tmp_path: Path) -> None:
    archive_path = tmp_path / "incomplete.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("stores.csv", "store_nbr\n1\n")

    with pytest.raises(RuntimeError, match="exactly one copy"):
        download_competition_data(tmp_path / "raw", archive_path)


def test_archive_lookup_uses_the_first_matching_directory(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (second / COMPETITION_ARCHIVE).write_bytes(b"zip placeholder")

    assert find_competition_archive(first, second) == second / COMPETITION_ARCHIVE
    assert find_competition_archive(first) is None


def test_acquisition_surfaces_missing_sources_after_empty_prompt(tmp_path: Path) -> None:
    prompts: list[str] = []

    def empty_token(prompt: str) -> str:
        prompts.append(prompt)
        return ""

    with pytest.raises(FileNotFoundError, match="No Kaggle token"):
        acquire_competition_data(
            tmp_path / "raw",
            (tmp_path / "data", tmp_path),
            token="",
            token_reader=empty_token,
        )

    assert prompts == ["Kaggle API token (press Enter to continue without one): "]


def test_prompted_token_is_not_left_in_environment(monkeypatch, tmp_path: Path) -> None:
    observed_tokens: list[str | None] = []
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)

    def fake_download(data_dir: Path, archive_path: Path | None = None) -> Path:
        assert archive_path is None
        observed_tokens.append(acquisition.os.getenv("KAGGLE_API_TOKEN"))
        return data_dir

    monkeypatch.setattr(acquisition, "download_competition_data", fake_download)
    result = acquire_competition_data(
        tmp_path / "raw",
        (tmp_path,),
        token_reader=lambda _prompt: "temporary-secret",
    )

    assert result == tmp_path / "raw"
    assert observed_tokens == ["temporary-secret"]
    assert acquisition.os.getenv("KAGGLE_API_TOKEN") is None


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
