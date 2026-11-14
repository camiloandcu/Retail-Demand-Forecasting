"""Normalized ForecastRequest directories: typed CSV tables bound by a hashed manifest.

A request directory lets prediction run without the seven competition files or any
future labels. CSV keeps core installs free of optional parquet dependencies.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Final

import pandas as pd

from retail_forecast.artifacts import _trusted_path, read_regular_bytes, tree_matches
from retail_forecast.data_contracts import (
    TABLE_COLUMNS,
    ForecastRequest,
    validate_forecast_request,
)
from retail_forecast.io_utils import canonical_json, parse_json_bytes

MANIFEST_NAME: Final = "request.json"
MANIFEST_KEYS: Final = frozenset(
    {
        "schema_version",
        "kind",
        "input_contract_version",
        "snapshot_id",
        "data_source",
        "availability_policy",
        "origin",
        "horizon",
        "series_keys",
        "tables",
    }
)
_DTYPES: Final = {
    "store_nbr": "int64",
    "onpromotion": "int64",
    "transactions": "int64",
    "cluster": "int64",
    "sales": "float64",
    "dcoilwtico": "float64",
}


def _table_bytes(frame: pd.DataFrame) -> bytes:
    output = frame.copy()
    if "date" in output:
        output["date"] = output["date"].dt.strftime("%Y-%m-%d")
    buffer = io.StringIO()
    output.to_csv(buffer, index=False, float_format="%.17g", lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def _parse_table(name: str, data: bytes) -> pd.DataFrame:
    columns = TABLE_COLUMNS[name]
    dtypes = {column: _DTYPES.get(column, str) for column in columns if column != "date"}
    if "transferred" in columns:
        dtypes.pop("transferred")
    frame = pd.read_csv(
        io.BytesIO(data),
        dtype=dtypes,
        keep_default_na=False,
        na_values={"dcoilwtico": [""]} if name == "competition_oil" else {},
    )
    if tuple(frame.columns) != columns:
        raise ValueError(f"Request table {name} has unexpected columns")
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d").astype("datetime64[ns]")
    if "transferred" in frame:
        text = frame["transferred"].astype(str)
        if not text.isin(["True", "False"]).all():
            raise ValueError("Request calendar.transferred must be True/False")
        frame["transferred"] = text.eq("True")
    return frame


def serialize_request(request: ForecastRequest) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Return table files and the manifest binding their hashes and request identity."""

    tables = {
        "sales_history": request.sales_history,
        "transactions_history": request.transactions_history,
        "future_plan": request.future_plan,
        "stores": request.stores,
        "calendar": request.calendar,
    }
    if request.competition_oil is not None:
        tables["competition_oil"] = request.competition_oil
    files = {f"tables/{name}.csv": _table_bytes(frame) for name, frame in tables.items()}
    manifest = {
        "schema_version": 1,
        "kind": "forecast_request",
        "input_contract_version": request.input_contract_version,
        "snapshot_id": request.snapshot_id,
        "data_source": request.data_source,
        "availability_policy": request.availability_policy,
        "origin": request.origin.date().isoformat(),
        "horizon": request.horizon,
        "series_keys": [[int(store), str(family)] for store, family in request.series_keys],
        "tables": {
            name: {
                "path": f"tables/{name}.csv",
                "bytes": len(files[f"tables/{name}.csv"]),
                "sha256": hashlib.sha256(files[f"tables/{name}.csv"]).hexdigest(),
            }
            for name in tables
        },
    }
    return files, manifest


def write_request_dir(request: ForecastRequest, directory: Path) -> Path:
    """Write a request directory outside a run (tests, external producers)."""

    directory = _trusted_path(directory)
    if directory.exists():
        raise ValueError("Request directory already exists")
    files, manifest = serialize_request(request)
    for name, data in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (directory / MANIFEST_NAME).write_bytes(canonical_json(manifest))
    return directory


def request_files(directory: Path) -> dict[str, bytes]:
    """Read the manifest and every declared table once, verifying hashes and closure."""

    directory = _trusted_path(directory)
    manifest_bytes = read_regular_bytes(directory, MANIFEST_NAME)
    manifest = parse_json_bytes(manifest_bytes)
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_KEYS:
        raise ValueError("Invalid request manifest schema")
    if manifest["schema_version"] != 1 or manifest["kind"] != "forecast_request":
        raise ValueError("Unsupported request manifest")
    tables = manifest["tables"]
    required = {"sales_history", "transactions_history", "future_plan", "stores", "calendar"}
    if not isinstance(tables, dict) or not required <= set(tables) <= required | {
        "competition_oil"
    }:
        raise ValueError("Request manifest tables are incomplete or unknown")
    contents = {MANIFEST_NAME: manifest_bytes}
    for name, entry in tables.items():
        if entry.get("path") != f"tables/{name}.csv":
            raise ValueError("Unexpected request table path")
        data = read_regular_bytes(directory, entry["path"])
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError("Request table failed integrity verification")
        contents[entry["path"]] = data
    if not tree_matches(directory, set(contents)):
        raise ValueError("Request directory contains undeclared, special or missing entries")
    return contents


def request_from_files(contents: dict[str, bytes]) -> ForecastRequest:
    """Build and validate a ForecastRequest from verified directory contents."""

    manifest = parse_json_bytes(contents[MANIFEST_NAME])
    frames = {
        name: _parse_table(name, contents[entry["path"]])
        for name, entry in manifest["tables"].items()
    }
    keys = tuple((int(store), str(family)) for store, family in manifest["series_keys"])
    request = ForecastRequest(
        input_contract_version=manifest["input_contract_version"],
        snapshot_id=manifest["snapshot_id"],
        data_source=manifest["data_source"],
        availability_policy=manifest["availability_policy"],
        origin=pd.Timestamp(manifest["origin"]),
        horizon=manifest["horizon"],
        series_keys=keys,
        sales_history=frames["sales_history"],
        transactions_history=frames["transactions_history"],
        future_plan=frames["future_plan"],
        stores=frames["stores"],
        calendar=frames["calendar"],
        competition_oil=frames.get("competition_oil"),
    )
    validate_forecast_request(request, supported_series=keys)
    return request


def read_request_dir(directory: Path) -> ForecastRequest:
    return request_from_files(request_files(directory))
