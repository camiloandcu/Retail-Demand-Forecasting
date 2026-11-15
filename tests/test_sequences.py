"""Causality, sample identity and train-only sequence transformations."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from retail_forecast.data_contracts import project_snapshot
from retail_forecast.sequences import SequencePreprocessor, build_panel, request_panel
from retail_forecast.training import cutoff_training_origins


def panel_from(data, *, cutoff=None):
    config, f, series, outer = data
    cutoff = cutoff or outer
    origins = cutoff_training_origins(
        cutoff, 16, f["train.csv"]["date"], stride_days=16, max_origins=3, name="test"
    )
    return build_panel(
        f["train.csv"],
        f["transactions.csv"],
        f["stores.csv"],
        f["holidays_events.csv"],
        series=series,
        origins=origins,
        cutoff=cutoff,
        policy="asof_v1",
    )


def test_shapes_masks_and_frozen_new_samples(sequence_data):
    _, frames, series, cutoff = sequence_data
    panel = panel_from(sequence_data)
    pre = SequencePreprocessor.fit(panel, cutoff=cutoff)
    original_hash = pre.sha256
    b = pre.batch(panel, np.array([0, 1]))
    assert b["history"].shape == (2, 56, 8)
    assert b["future"].shape == (2, 16, 12)
    assert b["static"].shape == (2, 6)
    assert b["static"].dtype == np.int64
    assert b["target_observed"].all()
    request = project_snapshot(
        frames,
        snapshot_id="a" * 64,
        data_source="synthetic",
        origin=cutoff,
        policy="asof_v1",
        context_days=56,
        supported_series=series,
    )
    values = pre.batch(request_panel(request), np.arange(len(series)))
    assert not values["target_observed"].any()
    assert np.isfinite(values["history"]).all()
    assert np.isfinite(values["future"]).all()
    assert pre.sha256 == original_hash
    restored = SequencePreprocessor.load(pre.state)
    assert restored.sha256 == original_hash


def test_future_labels_and_transactions_do_not_change_inputs(sequence_data):
    _, f, _, cutoff = sequence_data
    before = panel_from(sequence_data)
    pre = SequencePreprocessor.fit(before, cutoff=cutoff)
    for name, col in [("train.csv", "sales"), ("transactions.csv", "transactions")]:
        f[name].loc[f[name]["date"] > cutoff, col] = 999999
    after = panel_from(sequence_data)
    changed = SequencePreprocessor.fit(after, cutoff=cutoff)
    assert changed.sha256 == pre.sha256
    for k, v in pre.batch(before, np.arange(len(before.keys))).items():
        np.testing.assert_equal(v, changed.batch(after, np.arange(len(after.keys)))[k])


def test_internal_cutoff_and_deduplication(sequence_data):
    *_, outer = sequence_data
    panel = panel_from(sequence_data, cutoff=outer - pd.Timedelta(days=16))
    pre = SequencePreprocessor.fit(panel, cutoff=panel.cutoff)
    with pytest.raises(ValueError, match="cutoff"):
        SequencePreprocessor.fit(panel, cutoff=outer)
    used = np.zeros_like(panel.sales, dtype=bool)
    for s, o in panel.keys:
        used[s, o - 55 : o + 17] = True
    raw = panel.sales[0, used[0]]
    assert pre.state["target_stats"][0][2] == pytest.approx(np.log1p(raw).mean())
    # Independently deduplicate transactions by store/date, not by family.
    selected = []
    for store in sorted({s for s, _ in panel.series}):
        rows = [i for i, k in enumerate(panel.series) if k[0] == store]
        selected.extend(panel.transactions[rows[0], used[rows].any(axis=0)])
    expected = np.log1p(np.array(selected))
    assert pre.state["transaction_stats"] == pytest.approx([expected.mean(), expected.std()])


def test_constant_missing_and_unknown_series(sequence_data):
    _, f, series, cutoff = sequence_data
    key = series[0]
    mask = f["train.csv"]["store_nbr"].eq(key[0]) & f["train.csv"]["family"].eq(key[1])
    f["train.csv"].loc[mask, "sales"] = 0
    panel = panel_from(sequence_data)
    pre = SequencePreprocessor.fit(panel, cutoff=cutoff)
    assert pre.state["target_stats"][0][2:] == [0.0, 1.0]
    req = project_snapshot(
        f,
        snapshot_id="a" * 64,
        data_source="synthetic",
        origin=cutoff,
        policy="asof_v1",
        context_days=56,
        supported_series=series,
    )
    keep = ~(
        req.sales_history["store_nbr"].eq(key[0])
        & req.sales_history["family"].eq(key[1])
        & req.sales_history["date"].eq(cutoff)
    )
    sparse = replace(req, sales_history=req.sales_history.loc[keep].copy())
    b = pre.batch(request_panel(sparse), np.array([0]))
    assert b["history"][0, -1, 3] == 0
    assert b["history"][0, -2, 3] == 1
    assert b["history"][0, -1, 0] == 0
    none = replace(
        req,
        sales_history=req.sales_history.loc[
            ~(req.sales_history["store_nbr"].eq(key[0]) & req.sales_history["family"].eq(key[1]))
        ].copy(),
    )
    with pytest.raises(ValueError, match="HISTORY_INSUFFICIENT"):
        request_panel(none)


def test_declared_calendar_coverage_allows_missing_leading_day(sequence_data):
    _, frames, series, cutoff = sequence_data
    from retail_forecast.data_contracts import validate_forecast_request
    from retail_forecast.io_utils import canonical_json
    from retail_forecast.request_store import request_from_files, serialize_request

    panel = panel_from(sequence_data)
    pre = SequencePreprocessor.fit(panel, cutoff=cutoff)
    req = project_snapshot(
        frames,
        snapshot_id="a" * 64,
        data_source="synthetic",
        origin=cutoff,
        policy="asof_v1",
        context_days=56,
        series_keys=(series[0],),
        supported_series=series,
    )
    sparse = replace(
        req,
        sales_history=req.sales_history.loc[req.sales_history["date"] > req.history_start].copy(),
    )
    validate_forecast_request(sparse, supported_series=series, min_context_days=56)
    b = pre.batch(request_panel(sparse), np.array([0]))
    assert b["history"][0, 0, 3] == 0
    assert b["history"][0, 1:, 3].all()
    files, metadata = serialize_request(sparse)
    restored = request_from_files({**files, "request.json": canonical_json(metadata)})
    assert restored.history_start == sparse.history_start
    with pytest.raises(ValueError, match="HISTORY_INSUFFICIENT"):
        validate_forecast_request(
            replace(sparse, history_start=None), supported_series=series, min_context_days=56
        )
