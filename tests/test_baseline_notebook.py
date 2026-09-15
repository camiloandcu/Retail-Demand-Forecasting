"""Structural checks for the non-neural baseline notebook."""

import json
from pathlib import Path


def test_baseline_notebook_keeps_the_experiment_contract() -> None:
    notebook = json.loads(Path("03_non_neural_baselines.ipynb").read_text())
    full_text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "Open In Colab" in full_text
    assert 'BRANCH = "feat-leakage-safe-features"' in full_text
    assert '".[notebook,dev,models]"' in full_text
    assert "run_baseline_experiments" in full_text
    assert "artifacts/metrics/baseline_results.csv" in full_text
    assert "LightGBM is the only tree model" in full_text
    assert "at least 2% below LightGBM" in full_text
    assert "LSTM" not in full_text
    assert "GRU" not in full_text
    assert "train_test_split" not in full_text
