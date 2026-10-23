"""Denied acquisition paths use small explicit trusted fixture manifests."""

import hashlib
import io
import logging
import stat
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

import retail_forecast.acquisition as acquisition
from retail_forecast.acquisition_security import AcquisitionManifest, copy_verified
from retail_forecast.data import FILE_COLUMNS


@pytest.fixture
def manifest():
    payload = b"safe\n"
    return AcquisitionManifest(
        dict.fromkeys(FILE_COLUMNS, len(payload)),
        dict.fromkeys(FILE_COLUMNS, hashlib.sha256(payload).hexdigest()),
    )


def make_archive(path, *, extra=None, bad_content=False):
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name in FILE_COLUMNS:
            archive.writestr(name, b"evil\n" if bad_content else b"safe\n")
        if extra:
            archive.writestr(*extra)
    return path


@pytest.mark.parametrize("name", ["../escape", "/escape", "..\\escape"])
def test_traversal_is_rejected(tmp_path, manifest, name):
    archive = make_archive(tmp_path / "data.zip", extra=(name, b""))
    with pytest.raises(RuntimeError, match="unsafe member"):
        acquisition.extract_competition_archive(archive, tmp_path / "raw", manifest=manifest)
    assert not (tmp_path / "raw").exists()
    assert not list(tmp_path.glob(".acquire-*"))


@pytest.mark.parametrize("kind", [stat.S_IFLNK, stat.S_IFIFO])
def test_unsupported_archive_member_type(tmp_path, manifest, kind):
    member = ZipInfo("unsafe")
    member.create_system = 3
    member.external_attr = (kind | 0o777) << 16
    archive = make_archive(tmp_path / "data.zip", extra=(member, b""))
    with pytest.raises(RuntimeError, match="unsafe member"):
        acquisition.extract_competition_archive(archive, tmp_path / "raw", manifest=manifest)
    assert not (tmp_path / "raw").exists()


@pytest.mark.parametrize("target", ["root", "parent", "file"])
def test_destination_symlinks_are_never_followed(tmp_path, manifest, target):
    outside = tmp_path / "outside"
    outside.mkdir()
    data_dir = tmp_path / "raw"
    if target == "root":
        data_dir.symlink_to(outside, target_is_directory=True)
    elif target == "parent":
        (tmp_path / "alias").symlink_to(outside, target_is_directory=True)
        data_dir = tmp_path / "alias" / "raw"
    else:
        data_dir.mkdir()
        for name in FILE_COLUMNS:
            (data_dir / name).symlink_to(outside / name)
    archive = make_archive(tmp_path / "data.zip")
    with pytest.raises(RuntimeError):
        acquisition.extract_competition_archive(archive, data_dir, manifest=manifest)
    assert list(outside.iterdir()) == []


def test_duplicate_and_oversized_members_rejected(tmp_path, manifest):
    for extra in [("nested/train.csv", b""), ("unexpected", b"overflow")]:
        archive = make_archive(tmp_path / "data.zip", extra=extra)
        with pytest.raises(RuntimeError, match="exactly one copy|declared byte limit"):
            acquisition.extract_competition_archive(archive, tmp_path / "raw", manifest=manifest)
        assert not (tmp_path / "raw").exists()


def test_actual_stream_limit_is_enforced(manifest):
    target = io.BytesIO()
    with pytest.raises(RuntimeError, match="byte limit"):
        copy_verified(io.BytesIO(b"safe\noverflow"), target, "train.csv", manifest)
    assert target.getvalue() == b""


def test_hash_failure_and_interruption_publish_nothing(tmp_path, manifest, monkeypatch):
    archive = make_archive(tmp_path / "data.zip", bad_content=True)
    with pytest.raises(RuntimeError, match="trusted manifest"):
        acquisition.extract_competition_archive(archive, tmp_path / "raw", manifest=manifest)
    assert not (tmp_path / "raw").exists()
    archive = make_archive(tmp_path / "data.zip")

    def interrupted(*args):
        raise OSError("interrupted fixture")

    monkeypatch.setattr(acquisition, "copy_verified", interrupted)
    with pytest.raises(RuntimeError, match="destination unchanged"):
        acquisition.extract_competition_archive(archive, tmp_path / "raw", manifest=manifest)
    assert not (tmp_path / "raw").exists()
    assert not list(tmp_path.glob(".acquire-*"))


def test_partial_destination_is_preserved(tmp_path, manifest):
    raw = tmp_path / "raw"
    raw.mkdir()
    existing = raw / "train.csv"
    existing.write_bytes(b"existing user data")
    archive = make_archive(tmp_path / "data.zip")
    with pytest.raises(RuntimeError, match="new empty directory"):
        acquisition.extract_competition_archive(archive, raw, manifest=manifest)
    assert existing.read_bytes() == b"existing user data"


def test_valid_dataset_is_not_replaced(tmp_path, manifest):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in FILE_COLUMNS:
        (raw / name).write_bytes(b"safe\n")
    archive = make_archive(tmp_path / "data.zip", bad_content=True)
    assert acquisition.extract_competition_archive(archive, raw, manifest=manifest) == raw
    assert all((raw / name).read_bytes() == b"safe\n" for name in FILE_COLUMNS)


@pytest.mark.parametrize("failure", [False, True])
def test_client_download_is_staged_and_verified(tmp_path, manifest, monkeypatch, failure):
    def download(slug, output_dir):
        stage = Path(output_dir)
        assert stage != tmp_path / "raw"
        for name in FILE_COLUMNS:
            (stage / name).write_bytes(b"evil\n" if failure else b"safe\n")

    monkeypatch.setitem(sys.modules, "kagglehub", SimpleNamespace(competition_download=download))
    if failure:
        with pytest.raises(RuntimeError, match="download failed"):
            acquisition.download_competition_data(tmp_path / "raw", manifest=manifest)
        assert not (tmp_path / "raw").exists()
    else:
        acquisition.download_competition_data(tmp_path / "raw", manifest=manifest)
        assert all((tmp_path / "raw" / name).read_bytes() == b"safe\n" for name in FILE_COLUMNS)


@pytest.mark.parametrize("previous", [None, "previous-token"])
@pytest.mark.parametrize("failure", [False, True])
def test_token_restored_and_secret_absent_from_failure_output(
    tmp_path, monkeypatch, capsys, caplog, previous, failure
):
    sentinel = "fixture-secret-sentinel"
    if previous is None:
        monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    else:
        monkeypatch.setenv("KAGGLE_API_TOKEN", previous)

    def fake_download(data_dir, archive_path=None):
        assert acquisition.os.environ["KAGGLE_API_TOKEN"] == sentinel
        if failure:
            raise RuntimeError(sentinel)
        return data_dir

    monkeypatch.setattr(acquisition, "download_competition_data", fake_download)
    if failure:
        with pytest.raises(RuntimeError) as error:
            acquisition.acquire_competition_data(tmp_path / "raw", (tmp_path,), token=sentinel)
        assert sentinel not in "".join(traceback.format_exception(error.value))
    else:
        acquisition.acquire_competition_data(tmp_path / "raw", (tmp_path,), token=sentinel)
    assert acquisition.os.getenv("KAGGLE_API_TOKEN") == previous
    captured = capsys.readouterr()
    assert sentinel not in captured.out + captured.err + caplog.text


def test_client_failure_does_not_expose_third_party_output(
    tmp_path, manifest, monkeypatch, capsys, caplog
):
    sentinel = "fixture-secret-sentinel"

    def download(*args, **kwargs):
        print(sentinel)
        print(sentinel, file=sys.stderr)
        logging.warning(sentinel)
        raise RuntimeError(sentinel)

    monkeypatch.setitem(sys.modules, "kagglehub", SimpleNamespace(competition_download=download))
    with pytest.raises(RuntimeError) as error:
        acquisition.download_competition_data(tmp_path / "raw", manifest=manifest)
    captured = capsys.readouterr()
    assert sentinel not in captured.out + captured.err + caplog.text
    assert sentinel not in "".join(traceback.format_exception(error.value))
    assert not (tmp_path / "raw").exists()
