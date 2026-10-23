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
    ["01_data_audit_eda.ipynb", "02_feature_pipeline.ipynb", "03_non_neural_baselines.ipynb"],
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
        for name in [
            "src/retail_forecast/notebook_runtime.py",
            "src/retail_forecast/acquisition_security.py",
            "constraints/02-notebook-models.txt",
        ]:
            subprocess.run(["git", "cat-file", "-e", f"{revision}:{name}"], check=True)
