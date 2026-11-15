"""An unchanged environment must still install the requested source checkout."""

import hashlib
import subprocess

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
