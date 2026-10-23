"""Behavioral verification of immutable setup using disposable Git repositories."""

import json
import subprocess
from pathlib import Path

import pytest

from retail_forecast import notebook_runtime as runtime


def command(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "checkout"
    root.mkdir()
    command(root, "init", "-q")
    command(root, "config", "user.email", "test@example.invalid")
    command(root, "config", "user.name", "Test")
    command(root, "remote", "add", "origin", "https://example.invalid/repo.git")
    for name in ("pyproject.toml", "uv.lock", "src/retail_forecast/notebook_runtime.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    command(root, "add", ".")
    command(root, "commit", "-qm", "fixture")
    return root, command(root, "rev-parse", "HEAD"), "https://example.invalid/repo.git"


def test_clean_checkout_is_idempotent(checkout):
    assert runtime.validate_checkout(*checkout) == checkout[0]
    assert runtime.validate_checkout(*checkout) == checkout[0]


@pytest.mark.parametrize("change", ["tracked", "untracked", "origin", "revision", "partial"])
def test_refusal_preserves_user_checkout(checkout, change):
    root, revision, origin = checkout
    if change == "tracked":
        (root / "uv.lock").write_text("user edits")
    elif change == "untracked":
        (root / "notes.txt").write_text("user notes")
    elif change == "origin":
        command(root, "remote", "set-url", "origin", "wrong")
    elif change == "revision":
        revision = "0" * 40
    else:
        (root / "pyproject.toml").unlink()
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(RuntimeError):
        runtime.validate_checkout(root, revision, origin)
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_nested_checkout_refused(checkout):
    root, revision, origin = checkout
    with pytest.raises(RuntimeError, match="root"):
        runtime.validate_checkout(root / "src", revision, origin)


@pytest.mark.parametrize("version", [(3, 9), (3, 14), (2, 7)])
def test_unsupported_runtime(version):
    with pytest.raises(RuntimeError, match="3.10"):
        runtime.check_python(version)


def test_foreign_import_requires_restart(tmp_path):
    with pytest.raises(RuntimeError, match="Restart"):
        runtime.verify_loaded_source(tmp_path)


def test_install_failure_stops_before_import(tmp_path, monkeypatch):
    root = Path.cwd()
    calls = []

    def fail(*args, **kwargs):
        calls.append(args[0])
        return subprocess.CompletedProcess(args[0], 1, "", "private sentinel")

    monkeypatch.setattr(runtime.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="installation failed") as error:
        runtime.install(root)
    assert len(calls) == 1
    assert "sentinel" not in str(error.value)


def test_lock_profile_mismatch_stops_before_install(tmp_path):
    (tmp_path / "constraints").mkdir()
    (tmp_path / "uv.lock").write_text("different lock")
    (tmp_path / "constraints/01-notebook.txt").write_text("old constraints")
    with pytest.raises(RuntimeError, match="Constraints"):
        runtime.install(tmp_path)


def test_pinned_revision_survives_origin_advance(checkout, tmp_path):
    root, revision, _ = checkout
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(root), str(clone)], check=True)
    command(clone, "checkout", "--detach", revision)
    (root / "new.txt").write_text("new default branch content")
    command(root, "add", ".")
    command(root, "commit", "-qm", "advance")
    assert runtime.validate_checkout(clone, revision, str(root)) == clone


def test_smoke_acquisition_cell_never_requests_credentials(tmp_path, monkeypatch):
    from retail_forecast.config import load_config
    from retail_forecast.data import required_files_exist, write_synthetic_dataset

    monkeypatch.setenv("RETAIL_FORECAST_MODE", "smoke")
    for path in sorted(Path.cwd().glob("0*.ipynb")):
        notebook = json.loads(path.read_text())
        acquisition = next(
            "".join(c["source"])
            for c in notebook["cells"]
            if c["cell_type"] == "code" and "elif not required_files_exist" in "".join(c["source"])
        )
        # Execute only the acquisition branch, excluding downstream empirical work.
        acquisition = acquisition.split('if MODE == "full":')[0]
        acquisition = acquisition.split("\nhashes = ")[0]
        acquisition = acquisition.split("\ndataset_contract = ")[0]
        config = load_config("configs/smoke.yaml")
        from dataclasses import replace

        config = replace(config, paths=replace(config.paths, raw_data_dir=tmp_path / path.stem))

        def denied(*args, **kwargs):
            pytest.fail("Synthetic acquisition requested credentials or competition data")

        exec(
            acquisition,
            {
                "config": config,
                "MODE": "smoke",
                "required_files_exist": required_files_exist,
                "write_synthetic_dataset": write_synthetic_dataset,
                "acquire_competition_data": denied,
                "getpass": denied,
                "REPO_ROOT": tmp_path,
            },
        )
        assert required_files_exist(config.paths.raw_data_dir)


@pytest.mark.parametrize(
    "module_name, distribution_name", [("yaml", "PyYAML"), ("ipykernel", "ipykernel")]
)
def test_loaded_dependency_outside_original_allowlist_requires_restart(
    checkout, monkeypatch, module_name, distribution_name
):
    import types

    root, revision, origin = checkout
    monkeypatch.setattr(runtime, "verify_loaded_source", lambda root: None)
    monkeypatch.setattr(
        runtime.importlib.metadata,
        "packages_distributions",
        lambda: {module_name: [distribution_name]},
    )
    monkeypatch.setitem(runtime.sys.modules, module_name, types.ModuleType(module_name))
    installed = {"version": "6.0.1"}
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda name: installed["version"])
    monkeypatch.setattr(
        runtime, "install", lambda *args, **kwargs: installed.update(version="6.0.3")
    )
    with pytest.raises(RuntimeError, match=f"{distribution_name}.*Restart"):
        runtime.bootstrap(root, revision, origin)


@pytest.mark.parametrize("resident,metadata", [("2026.07.22", "2026.7.22"), ("1.2", "1.2.0")])
def test_equivalent_release_versions_do_not_require_restart(
    checkout, monkeypatch, resident, metadata
):
    import types

    root, revision, origin = checkout
    monkeypatch.setattr(runtime, "verify_loaded_source", lambda root: None)
    monkeypatch.setattr(
        runtime.importlib.metadata, "packages_distributions", lambda: {"certifi": ["certifi"]}
    )
    module = types.ModuleType("certifi")
    module.__version__ = resident
    monkeypatch.setitem(runtime.sys.modules, "certifi", module)
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda name: metadata)

    # Deliberately stop at installation to prove preflight accepted equivalence.
    def reached_install(*args, **kwargs):
        raise LookupError("reached installation")

    monkeypatch.setattr(runtime, "install", reached_install)
    with pytest.raises(LookupError, match="reached installation"):
        runtime.bootstrap(root, revision, origin)
