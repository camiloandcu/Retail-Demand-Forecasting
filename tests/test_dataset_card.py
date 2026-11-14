"""Dataset card traceability, stale-identity rejection and exposure checks."""

import copy
import json

import pytest

from retail_forecast.artifacts import allocate_run, capture_inputs
from retail_forecast.config import ProjectConfig
from retail_forecast.dataset_card import build_data_card, validate_data_card
from retail_forecast.eda import build_eda_summary, build_eda_tables, load_eda_frames


@pytest.fixture
def card_inputs(synthetic_config: ProjectConfig):
    run = allocate_run(synthetic_config, kind="baseline")
    snapshot = capture_inputs(run)
    frames = load_eda_frames(synthetic_config.paths.raw_data_dir)
    summary = build_eda_summary(frames, build_eda_tables(frames), "synthetic")
    return snapshot, json.loads(json.dumps(summary))


def test_card_values_resolve_from_manifests(card_inputs) -> None:
    snapshot, summary = card_inputs
    card = build_data_card(snapshot, summary, policy="competition_v1")
    validate_data_card(card, snapshot, summary)
    for entry in card["summary"]:
        section, name = entry["field"].split(".")
        assert summary[section][name] == entry["value"]
    assert card["dataset_card_schema_version"] == 1
    assert card["rights"]["status"] == "synthetic_fixture"


def test_stale_or_tampered_card_is_rejected(card_inputs) -> None:
    snapshot, summary = card_inputs
    card = build_data_card(snapshot, summary, policy="competition_v1")
    changed = copy.deepcopy(summary)
    changed["observations"]["train_rows"] += 1
    with pytest.raises(ValueError, match="stale"):
        validate_data_card(card, snapshot, changed)
    tampered = copy.deepcopy(card)
    tampered["summary"][0]["value"] = 1
    with pytest.raises(ValueError, match="differs"):
        validate_data_card(tampered, snapshot, summary)
    with pytest.raises(ValueError, match="POLICY_MISMATCH"):
        build_data_card(snapshot, summary, policy="asof_v1")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda card: card.update(note="/home/someone/data/raw/train.csv"),
        lambda card: card.update(scores={"rmsle": 0.4}),
        lambda card: card["rights"].update(status="granted"),
    ],
)
def test_card_exposure_and_claims_are_rejected(card_inputs, mutation) -> None:
    snapshot, summary = card_inputs
    card = build_data_card(snapshot, summary, policy="competition_v1")
    mutation(card)
    with pytest.raises(ValueError):
        validate_data_card(card, snapshot, summary)
