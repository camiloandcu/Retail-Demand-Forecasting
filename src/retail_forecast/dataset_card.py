"""Dataset card resolved from validated snapshot and EDA metadata (no model claims)."""

from __future__ import annotations

import hashlib
from typing import Any, Final

from retail_forecast.availability import resolve_policy
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
FORBIDDEN_KEYS: Final = frozenset({"rmsle", "mae", "wape", "bias", "y_pred", "prediction"})

POLICY_ASSUMPTIONS: Final = {
    "asof_v1": [
        "Sales and transactions are used only up to the forecast origin.",
        "Oil is excluded entirely; no oil-derived field is produced.",
        "Calendar rows of type Event are excluded, including scheduled ones.",
        "Other calendar rows are treated as scheduled; calendar vintages are unavailable.",
        "Historical onpromotion values proxy an unavailable promotion-plan vintage.",
        "Store metadata is assumed unchanged; metadata vintages are unavailable.",
    ],
    "competition_v1": [
        "Kaggle-supplied target-date oil and the full holiday/event table are used.",
        "This is a competition interface, not an operational availability regime.",
    ],
}


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
        raise ValueError("POLICY_MISMATCH: card policy differs from snapshot")
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
        "assumptions": POLICY_ASSUMPTIONS[resolved.name],
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
        "rights": {
            "status": "synthetic_fixture" if synthetic else "unverified",
            "primary_source_url": None if synthetic else RULES_URL,
            "checked_date": RIGHTS_CHECKED,
            "unknowns": []
            if synthetic
            else [
                "Competition rules text could not be retrieved for review on the checked date.",
                "Redistribution or public display of raw rows or derived tables is not "
                "established; raw data stays local.",
            ],
        },
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
    if "/home/" in text or "\\Users\\" in text or "/content/" in text:
        raise ValueError("Card contains a local path")

    def keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value)) if value else set()
        return set()

    if keys(card) & FORBIDDEN_KEYS:
        raise ValueError("Card contains model-performance fields")
    if card["rights"]["status"] == "granted":
        raise ValueError("Card cannot assert granted rights without reviewed evidence")
