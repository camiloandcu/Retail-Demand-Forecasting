"""Structural checks for the leakage-safe feature notebook."""

import json
from pathlib import Path


def test_feature_notebook_enforces_the_real_temporal_contract() -> None:
    notebook = json.loads(Path("02_feature_pipeline.ipynb").read_text())
    full_text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert notebook["nbformat"] == 4
    assert "Open In Colab" in full_text
    assert 'BRANCH = "feat-leakage-safe-features"' in full_text
    assert '"--branch"' in full_text
    assert '"--single-branch"' in full_text
    assert '"fetch", "--depth", "1", "origin", BRANCH' in full_text
    assert '"checkout", "--detach", "FETCH_HEAD"' in full_text
    assert "Kaggle API token" in full_text
    assert "token_reader=getpass" in full_text
    assert "acquire_competition_data" in full_text
    assert "USE_KAGGLE_SECRET" not in full_text
    assert "userdata" not in full_text
    assert "files.upload" not in full_text
    assert "The 16 horizons correspond to the 16 dates present in `test.csv`" in full_text
    assert "build_fold_feature_batches" in full_text
    assert "tests/test_splits.py" in full_text
    assert "tests/test_features.py" in full_text
    assert "feature_manifest.json" in full_text
    assert "split_manifest.json" in full_text
    assert "processed_dataset_exported" in full_text
    assert "train_test_split" not in full_text
    assert "interpolate(" not in full_text
    assert "G3 FAILED" in full_text
