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
    for name in (
        "pyproject.toml",
        "uv.lock",
        "src/retail_forecast/notebook_runtime.py",
        "constraints/01-notebook.txt",
        "constraints/03-colab.txt",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# uv.lock SHA256: fixture\nexample==1.0\n"
            "PyYAML==6.0.1\nipykernel==6.0.1\ntraitlets==5.16.1\n"
            "setuptools==84.2.1\n"
            if name in {"constraints/01-notebook.txt", "constraints/03-colab.txt"}
            else "fixture"
        )
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


def test_stale_module_version_clears_loaded_package_bytecode(tmp_path, monkeypatch):
    import types

    package = tmp_path / "traitlets"
    package.mkdir()
    stale_cache = package / "__pycache__" / "_version.cpython-313.pyc"
    stale_cache.parent.mkdir()
    stale_cache.write_bytes(b"stale")
    module = types.ModuleType("traitlets._version")
    module.__cached__ = str(stale_cache)
    monkeypatch.setitem(runtime.sys.modules, "traitlets._version", module)

    runtime.clear_loaded_bytecode("traitlets")

    assert not stale_cache.exists()


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


def test_install_reuses_environment_only_while_lock_and_freeze_match(tmp_path, monkeypatch):
    import hashlib

    root = tmp_path / "checkout"
    (root / "constraints").mkdir(parents=True)
    lock = root / "uv.lock"
    lock.write_text("fixture lock")
    lock_hash = hashlib.sha256(lock.read_bytes()).hexdigest()
    (root / "constraints/01-notebook.txt").write_text(
        f"# uv.lock SHA256: {lock_hash}\nexample==1.0\n"
    )
    marker_root = tmp_path / "tmp"
    marker_root.mkdir()
    monkeypatch.setattr(runtime.tempfile, "gettempdir", lambda: str(marker_root))
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda name: True)
    monkeypatch.setattr(runtime.site, "getsitepackages", lambda: [])
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        output = "example==1.0\n" if command[-1] == "freeze" else ""
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)

    runtime.install(root)
    runtime.install(root)

    assert sum(command[-1] != "freeze" for command in calls) == 2
    assert sum(command[-1] == "freeze" for command in calls) == 2


def test_lock_profile_mismatch_stops_before_install(tmp_path):
    (tmp_path / "constraints").mkdir()
    (tmp_path / "uv.lock").write_text("different lock")
    (tmp_path / "constraints/01-notebook.txt").write_text("old constraints")
    with pytest.raises(RuntimeError, match="Constraints"):
        runtime.install(tmp_path)


def test_colab_profiles_leave_managed_notebook_packages_to_runtime():
    root = Path.cwd()
    for models, expected in (
        (False, {"numpy", "pandas", "pyyaml", "kagglehub", "matplotlib"}),
        (
            True,
            {
                "numpy",
                "pandas",
                "pyyaml",
                "kagglehub",
                "matplotlib",
                "lightgbm",
                "mlflow",
                "scikit-learn",
            },
        ),
    ):
        profile = runtime.selected_profile(root, models=models, colab=True)
        packages = runtime.profile_distributions(profile)
        assert expected <= packages
        assert not packages.intersection(
            {"ipykernel", "jupyter", "traitlets", "setuptools", "wheel", "pytest", "ruff"}
        )


def test_colab_report_includes_installed_managed_packages_without_requiring_all(
    checkout, monkeypatch
):
    import types

    root, revision, origin = checkout
    distribution_map = {"yaml": ["PyYAML"], "traitlets": ["traitlets"]}
    monkeypatch.setattr(runtime, "verify_loaded_source", lambda root: None)
    monkeypatch.setattr(
        runtime.importlib.metadata,
        "packages_distributions",
        lambda: distribution_map,
    )
    monkeypatch.setattr(
        runtime.importlib.metadata,
        "version",
        lambda name: {"PyYAML": "6.0.1", "traitlets": "5.16.1", "pip": "26.0"}.get(name, "1.0"),
    )
    monkeypatch.setattr(runtime, "install", lambda *args, **kwargs: None)
    monkeypatch.setitem(runtime.sys.modules, "yaml", types.ModuleType("yaml"))
    loaded_traitlets = types.ModuleType("traitlets")
    loaded_traitlets.__version__ = "5.16.1"
    monkeypatch.setitem(runtime.sys.modules, "traitlets", loaded_traitlets)

    class Package:
        __file__ = str(root / "src/retail_forecast/__init__.py")

    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: Package)
    monkeypatch.setattr(
        runtime.importlib.metadata,
        "distribution",
        lambda name: types.SimpleNamespace(
            read_text=lambda filename: json.dumps({"url": root.as_uri()})
        ),
    )
    report = runtime.bootstrap(root, revision, origin, colab=True)
    assert report["dependency_profile"] == "colab"
    assert set(report["colab_platform_packages"]) == {"traitlets"}
    assert report["colab_platform_packages"]["traitlets"]["installed"] == "5.16.1"


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
        acquisition = acquisition.split("\ncapture_inputs(")[0]
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
    cleared = []
    monkeypatch.setattr(runtime, "clear_loaded_bytecode", cleared.append)
    with pytest.raises(RuntimeError, match="Locked dependencies.*Restart"):
        runtime.bootstrap(root, revision, origin)
    assert cleared == [module_name]


def test_all_loaded_package_caches_are_cleared_in_one_restart(checkout, monkeypatch):
    import types

    root, revision, origin = checkout
    monkeypatch.setattr(runtime, "verify_loaded_source", lambda root: None)
    monkeypatch.setattr(
        runtime.importlib.metadata,
        "packages_distributions",
        lambda: {"traitlets": ["traitlets"], "setuptools": ["setuptools"]},
    )
    for name, version in (("traitlets", "5.7.1"), ("setuptools", "80.9.0")):
        module = types.ModuleType(name)
        module.__version__ = version
        monkeypatch.setitem(runtime.sys.modules, name, module)
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda name: "5.16.1")
    cleared = []
    monkeypatch.setattr(runtime, "clear_loaded_bytecode", cleared.append)

    with pytest.raises(RuntimeError, match="Loaded dependency versions.*Restart"):
        runtime.bootstrap(root, revision, origin)

    assert sorted(cleared) == ["setuptools", "traitlets"]


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
