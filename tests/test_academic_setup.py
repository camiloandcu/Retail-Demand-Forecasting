"""An unchanged environment must still install the requested source checkout."""

import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from retail_forecast import notebook_runtime as runtime


def test_install_cache_does_not_reuse_a_different_checkout(tmp_path, monkeypatch):
    roots = [tmp_path / name for name in ("portfolio", "academic")]
    for root in roots:
        (root / "constraints").mkdir(parents=True)
        (root / "uv.lock").write_text("same lock")
        digest = hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()
        (root / "constraints/01-notebook.txt").write_text(
            f"# uv.lock SHA256: {digest}\nexample==1.0\n"
        )
    markers = tmp_path / "markers"
    markers.mkdir()
    monkeypatch.setattr(runtime.tempfile, "gettempdir", lambda: str(markers))
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda name: True)
    monkeypatch.setattr(runtime.site, "getsitepackages", lambda: [])
    installs = []

    def run(command, **kwargs):
        if command[-1] != "freeze":
            installs.append(command)
        return subprocess.CompletedProcess(command, 0, "example==1.0\n", "")

    monkeypatch.setattr(runtime.subprocess, "run", run)
    runtime.install(roots[0])
    runtime.install(roots[1])
    runtime.install(roots[1])
    assert len(installs) == 4
    assert installs[1][-1] == str(roots[0])
    assert installs[3][-1] == str(roots[1])


def mock_colab_install(monkeypatch, *, planned=(), status=0, snapshots=None):
    """Simulate pip resolution, including an attempted resident replacement."""
    versions = snapshots or [{"numpy": "2.1.3", "requests": "2.32.4"}] * 2
    snapshots_iter = iter(versions)
    monkeypatch.setattr(runtime, "installed_versions", lambda: next(snapshots_iter))
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda name: True)
    monkeypatch.setattr(runtime.site, "getsitepackages", lambda: [])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        constraints = runtime.Path(command[command.index("--constraint") + 1]).read_text()
        assert "numpy==2.1.3\n" in constraints
        assert "requests==2.32.4\n" in constraints
        if "--dry-run" in command:
            runtime.Path(command[command.index("--report") + 1]).write_text(
                json.dumps({"install": [{"metadata": {"name": n}} for n in planned]})
            )
            return subprocess.CompletedProcess(command, status, "", "resolution failed")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runtime.subprocess, "run", run)
    return calls


def test_colab_preserves_native_versions_and_records_new_packages(tmp_path, monkeypatch):
    calls = mock_colab_install(
        monkeypatch,
        planned=["retail-forecast", "lightgbm"],
        snapshots=[
            {"numpy": "2.1.3", "requests": "2.32.4"},
            {"numpy": "2.1.3", "requests": "2.32.4", "lightgbm": "4.6.0"},
        ],
    )
    report = runtime.install_colab(tmp_path, models=True)
    assert len(calls) == 2
    assert calls[0][-1].endswith("[colab,colab-models]")
    assert "--dry-run" in calls[0] and "--dry-run" not in calls[1]
    assert report["added_distributions"] == ["lightgbm"]
    assert report["packages"]["numpy"] == "2.1.3"


def test_colab_conflict_stops_before_installation(tmp_path, monkeypatch):
    calls = mock_colab_install(monkeypatch, status=1)
    with pytest.raises(RuntimeError, match="No packages were replaced"):
        runtime.install_colab(tmp_path)
    assert len(calls) == 1


def test_colab_refuses_even_same_version_resident_reinstallation(tmp_path, monkeypatch):
    calls = mock_colab_install(monkeypatch, planned=["NumPy"])
    with pytest.raises(RuntimeError, match="refused to replace resident"):
        runtime.install_colab(tmp_path)
    assert len(calls) == 1


def test_colab_detects_unexpected_installer_mutation(tmp_path, monkeypatch):
    mock_colab_install(
        monkeypatch,
        snapshots=[
            {"numpy": "2.1.3", "requests": "2.32.4"},
            {"numpy": "2.5.3", "requests": "2.32.4"},
        ],
    )
    with pytest.raises(RuntimeError, match="unexpectedly changed: .*numpy"):
        runtime.install_colab(tmp_path, neural=True)


def test_colab_bootstrap_routes_to_native_installer(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "validate_checkout", lambda *args: tmp_path)
    monkeypatch.setattr(runtime, "verify_loaded_source", lambda *args: None)
    monkeypatch.setattr(runtime, "profile_distributions", lambda *args: set())
    monkeypatch.setattr(runtime.importlib.metadata, "packages_distributions", lambda: {})
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda name: "1.0")
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda name: True)
    monkeypatch.setattr(runtime, "git", lambda *args: "a" * 40)
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        lambda name: SimpleNamespace(__file__=str(tmp_path / "src/retail_forecast/__init__.py")),
    )
    monkeypatch.setattr(
        runtime.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(read_text=lambda f: json.dumps({"url": tmp_path.as_uri()})),
    )
    (tmp_path / "uv.lock").write_text("lock")
    calls = []

    def install_colab(root, **kwargs):
        calls.append(kwargs)
        return {"policy": "colab-compatible-preserve-installed"}

    def forbidden(*args, **kwargs):
        raise AssertionError("Colab must not install the universal locked profile")

    monkeypatch.setattr(runtime, "install_colab", install_colab)
    monkeypatch.setattr(runtime, "install", forbidden)
    report = runtime.bootstrap(tmp_path, "a" * 40, "origin", colab=True, neural=True)
    assert calls == [{"models": False, "neural": True}]
    assert report["installation"]["policy"] == "colab-compatible-preserve-installed"


def test_pip_diagnostic_hides_urls_and_credentials():
    result = subprocess.CompletedProcess(
        [], 1, "", "ERROR https://user:secret@example.test/simple token=private password:secret"
    )
    detail = runtime.pip_diagnostic(result)
    assert "secret" not in detail and "private" not in detail and "example.test" not in detail
    assert "ERROR" in detail
