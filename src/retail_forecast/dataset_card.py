"""Dataset card resolved from validated snapshot and EDA metadata (no model claims)."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Final

from retail_forecast.availability import PolicyMismatchError, resolve_policy
from retail_forecast.io_utils import canonical_json

DATASET_CARD_SCHEMA_VERSION: Final = 1
RULES_URL: Final = "https://www.kaggle.com/competitions/store-sales-time-series-forecasting/rules"
RIGHTS_CHECKED: Final = "2026-11-14"
SNAPSHOT_REF: Final = "inputs/dataset-manifest.json"
EDA_REF: Final = "inputs/eda-summary.json"

# Every numerical card value is a reference into one of the two manifests above.
SUMMARY_FIELDS: Final = (
    (EDA_REF, "observations.train_rows"),
    (EDA_REF, "observations.stores"),
    (EDA_REF, "observations.families"),
    (EDA_REF, "observations.series"),
    (EDA_REF, "observations.train_date_range"),
    (EDA_REF, "observations.test_date_range"),
    (EDA_REF, "observations.test_horizon_days"),
    (EDA_REF, "observations.zero_sales_rate"),
    (EDA_REF, "observations.fractional_sales_rows"),
)
MISSINGNESS_FIELDS: Final = (
    (EDA_REF, "observations.missing_calendar_dates"),
    (EDA_REF, "observations.transaction_store_date_coverage"),
    (EDA_REF, "observations.oil_missing_values"),
)
LOCAL_PATH: Final = re.compile(
    r"(?<![\w.])/(home|tmp|mnt|content|Users|root|var|opt)/|(?<![A-Za-z])[A-Za-z]:[\\/]|\\\\Users"
)
FORBIDDEN_KEYS: Final = frozenset({"rmsle", "mae", "wape", "bias", "y_pred", "prediction"})


def _resolve(document: dict[str, Any], field: str) -> Any:
    value: Any = document
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"Card reference does not resolve: {field}")
        value = value[part]
    return value


def build_data_card(
    snapshot: dict[str, Any], eda_summary: dict[str, Any], *, policy: str
) -> dict[str, Any]:
    """Build a compact card whose values are copied only from referenced manifests."""

    resolved = resolve_policy(policy)
    if snapshot["availability_policy"] != resolved.name:
        raise PolicyMismatchError("POLICY_MISMATCH: card policy differs from snapshot")
    if eda_summary["source"] != snapshot["data_source"]:
        raise ValueError("Card EDA summary source differs from snapshot")
    documents = {SNAPSHOT_REF: snapshot, EDA_REF: eda_summary}

    def entries(fields: tuple[tuple[str, str], ...]) -> list[dict[str, Any]]:
        return [
            {"ref": ref, "field": field, "value": _resolve(documents[ref], field)}
            for ref, field in fields
        ]

    synthetic = snapshot["data_source"] == "synthetic"
    return {
        "dataset_card_schema_version": DATASET_CARD_SCHEMA_VERSION,
        "snapshot_id": snapshot["snapshot_id"],
        "raw_manifest_ref": SNAPSHOT_REF,
        "eda_summary_ref": EDA_REF,
        "eda_summary_sha256": _digest(eda_summary),
        "data_source": snapshot["data_source"],
        "extraction_code": snapshot["extraction_code"],
        "schema_ref": "retail_forecast.data.FILE_COLUMNS; ForecastRequest v1",
        "summary": entries(SUMMARY_FIELDS),
        "sales_unit_note": (
            "Daily observed sales per store and product family; fractional values are valid "
            "and units differ across families, so cross-family totals are not one quantity "
            "or currency. Sales are not unconstrained demand."
        ),
        "missingness": entries(MISSINGNESS_FIELDS)
        + [
            {
                "note": "Absent rows and missing transaction counts stay missing; they are "
                "never filled with zero sales or counts."
            }
        ],
        "availability_policy": resolved.name,
        "input_contract_version": resolved.input_contract_version,
        "assumptions": list(resolved.assumptions),
        "intended_use": (
            "Academic and portfolio evaluation of 16-day store-family sales forecasts."
        ),
        "limitations": [
            "No stock, price, cost or lead-time data; no inventory or savings claims.",
            "Associations in this data are not causal effects.",
            "Synthetic fixtures are schema checks, not empirical observations."
            if synthetic
            else "Kaggle test labels are unavailable; evaluation uses historical folds.",
        ],
        "rights": _rights(synthetic),
    }


def _rights(synthetic: bool) -> dict[str, Any]:
    if synthetic:
        return {
            "status": "synthetic_fixture",
            "primary_source_url": None,
            "checked_date": RIGHTS_CHECKED,
            "terms": [],
            "unknowns": [],
        }
    return {
        "status": "restricted_noncommercial",
        "primary_source_url": RULES_URL,
        "checked_date": RIGHTS_CHECKED,
        "source_note": "Rules sections 7-8 text supplied by the project owner from the "
        "competition rules page on the checked date; the page did not render for automated "
        "retrieval.",
        "terms": [
            "Rule 7A: non-commercial use only, including the competition, Kaggle forums, "
            "academic research and education.",
            "Rule 7B: no transmission, duplication, publication or redistribution of the "
            "Competition Data to non-participants.",
            "Rule 8B: public code sharing is permitted; code shared must also be shared on "
            "the competition's Kaggle forum or notebooks under a commercially unrestricted "
            "OSI-approved license.",
        ],
        "unknowns": [
            "Competition-specific rules outside sections 7-8 were not reviewed.",
            "Whether a public non-Kaggle repository alone satisfies rule 8B is a project "
            "owner decision, not established here.",
            "Whether derived aggregate tables or figures count as publishing the data is "
            "not stated; raw rows stay local and are never displayed.",
        ],
    }


def _digest(document: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(document)).hexdigest()


def validate_data_card(
    card: dict[str, Any], snapshot: dict[str, Any], eda_summary: dict[str, Any]
) -> None:
    """Reject stale identities, unresolved values and exposure of restricted content."""

    if card["snapshot_id"] != snapshot["snapshot_id"]:
        raise ValueError("Card snapshot identity is stale")
    if card["eda_summary_sha256"] != _digest(eda_summary):
        raise ValueError("Card EDA summary identity is stale")
    documents = {SNAPSHOT_REF: snapshot, EDA_REF: eda_summary}
    for entry in [*card["summary"], *card["missingness"]]:
        if "ref" in entry and _resolve(documents[entry["ref"]], entry["field"]) != entry["value"]:
            raise ValueError(f"Card value differs from its manifest: {entry['field']}")
    text = canonical_json(card).decode()
    if LOCAL_PATH.search(text):
        raise ValueError("Card contains a local path")

    def keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value)) if value else set()
        return set()

    if keys(card) & FORBIDDEN_KEYS:
        raise ValueError("Card contains model-performance fields")
    if card["rights"]["status"] not in {"synthetic_fixture", "restricted_noncommercial"}:
        raise ValueError("Card cannot assert granted rights without reviewed evidence")
