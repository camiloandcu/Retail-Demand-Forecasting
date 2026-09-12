"""Structural checks for the first academic notebook."""

import json
from pathlib import Path


def test_eda_notebook_is_colab_ready_and_keeps_logic_in_package() -> None:
    notebook = json.loads(Path("01_data_audit_eda.ipynb").read_text())
    sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    full_text = "\n".join(sources)

    assert notebook["nbformat"] == 4
    assert "Open In Colab" in full_text
    assert "KAGGLE_API_TOKEN" in full_text
    assert 'pip", "install", "--quiet", "-e", ".[notebook]' in full_text
    assert full_text.count("save_figure(") == 9
    assert "def rmsle" not in full_text
    assert "train_test_split" not in full_text
    assert "eda_summary.json" in full_text
