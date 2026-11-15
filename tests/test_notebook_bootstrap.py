"""Execute notebook setup cells against real local Git fixtures without network."""

import json
import subprocess
from pathlib import Path

import pytest
from test_notebook_runtime import checkout, command  # noqa: F401


def setup_cell(path):
    notebook = json.loads(path.read_text())
    return next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "CODE_REVISION =" in "".join(cell["source"])
    )


@pytest.mark.parametrize(
    "name",
    [
        "01_data_audit_eda.ipynb",
        "02_feature_pipeline.ipynb",
        "03_non_neural_baselines.ipynb",
        "04_recurrent_forecast.ipynb",
    ],
)
def test_initial_cell_reentry_and_dirty_refusal(name, checkout, monkeypatch):  # noqa: F811
    root, revision, origin = checkout
    cell = setup_cell(Path(name).resolve())
    import re
    import runpy

    cell = re.sub(r'CODE_REVISION = "[a-f0-9]{40}"', f'CODE_REVISION = "{revision}"', cell)
    cell = cell.replace("https://github.com/camiloandcu/Retail-Demand-Forecasting.git", origin)
    monkeypatch.setenv("RETAIL_FORECAST_CHECKOUT", str(root))
    actual = []
    monkeypatch.setattr(
        runpy,
        "run_path",
        lambda path: {
            "bootstrap": lambda *args, **kwargs: actual.append(command(root, "rev-parse", "HEAD"))
        },
    )
    monkeypatch.chdir(root)
    exec(cell, {})
    exec(cell, {})
    assert actual == [revision, revision]
    (root / "user.txt").write_text("preserve")
    before = command(root, "rev-parse", "HEAD")
    with pytest.raises(RuntimeError, match="Dirty"):
        exec(cell, {})
    assert actual == [revision, revision]
    assert command(root, "rev-parse", "HEAD") == before
    assert (root / "user.txt").read_text() == "preserve"


def test_notebook_pin_contains_all_runtime_and_acquisition_changes():
    import re

    for path in Path.cwd().glob("0*.ipynb"):
        revision = re.search(r'CODE_REVISION = "([a-f0-9]{40})"', setup_cell(path))[1]
        available = subprocess.run(["git", "cat-file", "-e", f"{revision}^{{commit}}"])
        assert available.returncode == 0, (
            f"{path.name} pins {revision}, which is not in the local object store. "
            "Run `python scripts/fetch_notebook_revisions.py`; if that fails, the pin "
            "is unpublished and a fresh Colab runtime cannot obtain it either."
        )
        for name in [
            "src/retail_forecast/notebook_runtime.py",
            "src/retail_forecast/acquisition_security.py",
            "constraints/02-notebook-models.txt",
            "constraints/03-colab.txt",
            "constraints/04-colab-models.txt",
        ]:
            subprocess.run(["git", "cat-file", "-e", f"{revision}:{name}"], check=True)


@pytest.mark.parametrize(
    "name",
    [
        "01_data_audit_eda.ipynb",
        "02_feature_pipeline.ipynb",
        "03_non_neural_baselines.ipynb",
        "04_recurrent_forecast.ipynb",
    ],
)
def test_colab_clone_obtains_pin_reachable_only_from_pull_request_ref(name, tmp_path, monkeypatch):
    """After a squash merge the pinned commit is on no branch; a plain clone lacks it."""
    import importlib.util
    import re
    import runpy

    work = tmp_path / "work"
    work.mkdir()
    for args in (
        ("init", "-q", "-b", "main"),
        ("config", "user.email", "test@example.invalid"),
        ("config", "user.name", "Test"),
    ):
        command(work, *args)
    (work / "package.txt").write_text("reviewed\n")
    command(work, "add", "package.txt")
    command(work, "commit", "-q", "-m", "base")
    command(work, "checkout", "-q", "-b", "feature")
    (work / "package.txt").write_text("reviewed feature\n")
    command(work, "commit", "-q", "-am", "pinned feature commit")
    pinned = command(work, "rev-parse", "HEAD")
    command(work, "checkout", "-q", "main")
    command(work, "merge", "-q", "--squash", "feature")
    command(work, "commit", "-q", "-m", "squash merge")
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    command(origin, "config", "uploadpack.allowReachableSHA1InWant", "true")
    command(work, "push", "-q", str(origin), "main", f"{pinned}:refs/pull/1/head")
    origin_url = origin.as_uri()  # file:// uses the real transport, not a local object copy
    plain = tmp_path / "plain"
    subprocess.run(["git", "clone", "-q", "--no-checkout", origin_url, str(plain)], check=True)
    missing = subprocess.run(["git", "-C", str(plain), "cat-file", "-e", f"{pinned}^{{commit}}"])
    assert missing.returncode != 0, "fixture must reproduce the post-squash clone gap"

    root = tmp_path / "colab-checkout"
    cell = setup_cell(Path(name).resolve())
    cell = re.sub(r'CODE_REVISION = "[a-f0-9]{40}"', f'CODE_REVISION = "{pinned}"', cell)
    cell = cell.replace("https://github.com/camiloandcu/Retail-Demand-Forecasting.git", origin_url)
    monkeypatch.setenv("RETAIL_FORECAST_CHECKOUT", str(root))
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda module, *args: (
            object() if module in {"google", "google.colab"} else real_find_spec(module, *args)
        ),
    )
    actual = []
    monkeypatch.setattr(
        runpy,
        "run_path",
        lambda path: {
            "bootstrap": lambda *args, **kwargs: actual.append(
                (command(root, "rev-parse", "HEAD"), kwargs["colab"])
            )
        },
    )
    monkeypatch.chdir(tmp_path)
    exec(cell, {})
    assert actual == [(pinned, True)]
    assert (root / "package.txt").read_text() == "reviewed feature\n"
