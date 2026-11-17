"""Indexed causal windows and frozen train-only transforms for the joint GRU.

Only minibatches are expanded to [B, 56, 8] and [B, 16, 12]. Panel arrays retain
missing values until masks are made; observed zero is never confused with absence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from retail_forecast.availability import resolve_policy
from retail_forecast.data_contracts import ForecastRequest, ForecastRequestError, SeriesKey
from retail_forecast.features import build_store_holiday_calendar
from retail_forecast.provenance import digest_json

HISTORY_CHANNELS = (
    "sales_log_z",
    "promotion_log_z",
    "transactions_log_z",
    "sales_observed",
    "promotion_observed",
    "transactions_observed",
    "day_of_week_sin",
    "day_of_week_cos",
)
FUTURE_CHANNELS = (
    "promotion_log_z",
    "is_promoted",
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_month_scaled",
    "month_sin",
    "month_cos",
    "is_payday",
    "is_holiday_event",
    "is_bridge",
    "is_transfer",
    "is_work_day",
)
STATIC_CHANNELS = ("store_nbr", "family", "city", "state", "type", "cluster")
CONTEXT = 56
HORIZON = 16


def _stats(values: np.ndarray) -> tuple[float, float]:
    observed = values[np.isfinite(values)]
    if not len(observed) or (observed < 0).any():
        raise ValueError("Sequence fitting requires observed nonnegative values")
    values = np.log1p(observed.astype(np.float64))
    center, scale = float(values.mean()), float(values.std(ddof=0))
    return center, scale if scale >= 1e-6 else 1.0


@dataclass(frozen=True)
class SequencePanel:
    """Compact daily arrays; keys index (series position, origin-date position)."""

    series: tuple[SeriesKey, ...]
    dates: pd.DatetimeIndex
    sales: np.ndarray
    promotion: np.ndarray
    transactions: np.ndarray
    calendar: np.ndarray
    static: tuple[tuple[str, ...], ...]
    keys: tuple[tuple[int, int], ...]
    cutoff: pd.Timestamp

    @property
    def identity(self) -> str:
        return digest_json(
            {
                "samples": [
                    [*self.series[s], self.dates[o].date().isoformat()] for s, o in self.keys
                ],
                "cutoff": self.cutoff.date().isoformat(),
            }
        )


def build_panel(
    sales: pd.DataFrame,
    transactions: pd.DataFrame,
    stores: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    series: tuple[SeriesKey, ...],
    origins: tuple[pd.Timestamp, ...],
    cutoff: pd.Timestamp,
    policy: str,
    future_plan: pd.DataFrame | None = None,
) -> SequencePanel:
    """Consume validated tables, project by cutoff, and index eligible windows.

    Training labels stop at cutoff. An inference future plan extends only the
    promotion array; its target sales remain absent. No adapter reads source files.
    """
    if not origins or not series or len(set(series)) != len(series):
        raise ValueError("Sequence origins and unique series are required")
    cutoff = pd.Timestamp(cutoff)
    start = min(origins) - pd.Timedelta(days=CONTEXT - 1)
    if future_plan is None and sales["date"].min() > start:
        raise ForecastRequestError("HISTORY_INSUFFICIENT", "snapshot.calendar_coverage")
    end = cutoff if future_plan is None else max(origins) + pd.Timedelta(days=HORIZON)
    dates = pd.date_range(start, end)
    n, t = len(series), len(dates)
    index = pd.MultiIndex.from_product([range(n), dates], names=["series", "date"])
    lookup = pd.DataFrame(series, columns=["store_nbr", "family"]).reset_index(names="series")
    observed = sales.loc[sales["date"].between(start, cutoff)].merge(
        lookup, on=["store_nbr", "family"], validate="many_to_one"
    )
    if observed.duplicated(["series", "date"]).any():
        raise ValueError("Duplicate sequence observation")
    aligned = observed.set_index(["series", "date"]).reindex(index)
    y = aligned["sales"].to_numpy(dtype=np.float64).reshape(n, t)
    promo = aligned["onpromotion"].to_numpy(dtype=np.float64).reshape(n, t)
    if future_plan is not None:
        planned = future_plan.merge(lookup, on=["store_nbr", "family"], validate="many_to_one")
        if planned.duplicated(["series", "date"]).any():
            raise ValueError("Duplicate sequence plan")
        p = planned.set_index(["series", "date"])["onpromotion"].reindex(index)
        p = p.to_numpy(dtype=np.float64).reshape(n, t)
        promo = np.where(np.isfinite(p), p, promo)
    tx = transactions.loc[transactions["date"].between(start, cutoff)]
    tx = lookup[["series", "store_nbr"]].merge(tx, on="store_nbr", validate="many_to_many")
    trans = tx.set_index(["series", "date"])["transactions"].reindex(index)
    trans = trans.to_numpy(dtype=np.float64).reshape(n, t)
    store_ids = sorted({s for s, _ in series})
    store_rows = stores.loc[stores["store_nbr"].isin(store_ids)]
    cal = build_store_holiday_calendar(store_rows, calendar, dates, policy=resolve_policy(policy))
    cal = lookup[["series", "store_nbr"]].merge(cal, on="store_nbr", validate="many_to_many")
    columns = ["is_holiday_event", "is_bridge", "is_transfer", "is_work_day"]
    flags = cal.set_index(["series", "date"])[columns].reindex(index)
    flags = flags.to_numpy(dtype=np.float32).reshape(n, t, 4)
    metadata = stores.set_index("store_nbr")
    static = tuple(
        (
            str(store),
            family,
            *(str(metadata.loc[store, c]) for c in ("city", "state", "type", "cluster")),
        )
        for store, family in series
    )
    keys = []
    for origin in origins:
        o = int(dates.get_indexer([origin])[0])
        if o < CONTEXT - 1 or o + HORIZON >= t:
            raise ValueError("Sequence interval is not covered")
        for s in range(n):
            if not np.isfinite(y[s, o - CONTEXT + 1 : o + 1]).any():
                if future_plan is not None:
                    raise ForecastRequestError("HISTORY_INSUFFICIENT", "sales_history")
                continue
            if future_plan is None and not np.isfinite(y[s, o + 1 : o + 17]).all():
                continue
            if not np.isfinite(promo[s, o + 1 : o + 17]).all():
                raise ForecastRequestError("FUTURE_PLAN_INCOMPLETE", "future_plan")
            keys.append((s, o))
    if not keys:
        raise ValueError("No eligible sequence samples")
    return SequencePanel(series, dates, y, promo, trans, flags, static, tuple(keys), cutoff)


def request_panel(request: ForecastRequest) -> SequencePanel:
    """Build an inference panel without accessing forecast labels."""
    start = (
        request.history_start
        if request.history_start is not None
        else request.sales_history["date"].min()
    )
    if start > request.origin - pd.Timedelta(days=55):
        raise ForecastRequestError("HISTORY_INSUFFICIENT", "sales_history")
    return build_panel(
        request.sales_history,
        request.transactions_history,
        request.stores,
        request.calendar,
        series=request.series_keys,
        origins=(request.origin,),
        cutoff=request.origin,
        policy=request.availability_policy,
        future_plan=request.future_plan,
    )


@dataclass(frozen=True)
class SequencePreprocessor:
    """JSON-native learned state. The checksum covers every transform and identity."""

    state: dict[str, Any]

    @property
    def sha256(self) -> str:
        return digest_json(self.state)

    @classmethod
    def fit(cls, panel: SequencePanel, *, cutoff: pd.Timestamp) -> SequencePreprocessor:
        if cutoff != panel.cutoff:
            raise ValueError("Preprocessor cutoff differs from eligible training data")
        used = np.zeros_like(panel.sales, dtype=bool)
        for s, o in panel.keys:
            if panel.dates[o + 16] > cutoff:
                raise ValueError("Fitting targets cross cutoff")
            used[s, o - 55 : o + 17] = True
        centers = []
        for s, key in enumerate(panel.series):
            center, scale = _stats(panel.sales[s, used[s]])
            centers.append([*key, center, scale])
        promo = _stats(panel.promotion[used])
        # One store/date observation, independent of the number of its families.
        tx_values = []
        for store in sorted({s for s, _ in panel.series}):
            positions = [i for i, key in enumerate(panel.series) if key[0] == store]
            mask = used[positions].any(axis=0)
            tx_values.append(panel.transactions[positions[0], mask])
        tx = _stats(np.concatenate(tx_values))
        vocab = {
            column: ["__UNKNOWN__", *sorted({row[i] for row in panel.static})]
            for i, column in enumerate(STATIC_CHANNELS)
        }
        row_hash = hashlib.sha256()
        row_hash.update(
            digest_json(
                {"series": panel.series, "dates": [d.date().isoformat() for d in panel.dates]}
            ).encode()
        )
        row_hash.update(np.packbits(used & np.isfinite(panel.sales)).tobytes())
        return cls(
            {
                "schema_version": 1,
                "fit_cutoff": cutoff.date().isoformat(),
                "eligible_origins_sha256": panel.identity,
                "fit_rows_sha256": row_hash.hexdigest(),
                "target_stats": centers,
                "promotion_stats": list(promo),
                "transaction_stats": list(tx),
                "vocabularies": vocab,
                "history_channels": list(HISTORY_CHANNELS),
                "future_channels": list(FUTURE_CHANNELS),
                "static_channels": list(STATIC_CHANNELS),
                "missing_fill_z": 0.0,
                "constant_scale": 1.0,
            }
        )

    @classmethod
    def load(cls, state: dict[str, Any]) -> SequencePreprocessor:
        if state.get("schema_version") != 1 or (
            state.get("history_channels") != list(HISTORY_CHANNELS)
            or state.get("future_channels") != list(FUTURE_CHANNELS)
            or state.get("static_channels") != list(STATIC_CHANNELS)
        ):
            raise ValueError("Incompatible sequence preprocessing")
        stats = [row[2:] for row in state["target_stats"]]
        stats += [state["promotion_stats"], state["transaction_stats"]]
        a = np.asarray(stats, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all() or (a[:, 1] <= 0).any():
            raise ValueError("Invalid sequence statistics")
        keys = [(r[0], r[1]) for r in state["target_stats"]]
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("Invalid saved series")
        for c in STATIC_CHANNELS:
            values = state["vocabularies"][c]
            if not values or values[0] != "__UNKNOWN__" or len(set(values)) != len(values):
                raise ValueError("Invalid saved vocabulary")
        return cls(state)

    def batch(self, panel: SequencePanel, positions: np.ndarray) -> dict[str, np.ndarray]:
        selected = [panel.keys[int(i)] for i in positions]
        s = np.array([k[0] for k in selected])
        o = np.array([k[1] for k in selected])
        hi = o[:, None] + np.arange(-55, 1)
        fi = o[:, None] + np.arange(1, 17)
        target_stats = {(r[0], r[1]): r[2:] for r in self.state["target_stats"]}
        try:
            stats = np.array([target_stats[panel.series[i]] for i in s], dtype=np.float32)
        except KeyError as error:
            raise ForecastRequestError("UNSUPPORTED_SERIES", "series_keys") from error
        center, scale = stats[:, :1], stats[:, 1:]
        raw = [
            panel.sales[s[:, None], hi],
            panel.promotion[s[:, None], hi],
            panel.transactions[s[:, None], hi],
        ]
        transforms = [
            (center, scale),
            self.state["promotion_stats"],
            self.state["transaction_stats"],
        ]
        z = [
            np.where(np.isfinite(x), (np.log1p(x) - c) / v, 0).astype(np.float32)
            for x, (c, v) in zip(raw, transforms, strict=True)
        ]
        dow = panel.dates.dayofweek.to_numpy()
        history = np.stack(
            [
                *z,
                *(np.isfinite(x).astype(np.float32) for x in raw),
                np.sin(2 * np.pi * dow[hi] / 7),
                np.cos(2 * np.pi * dow[hi] / 7),
            ],
            axis=-1,
        ).astype(np.float32)
        p = panel.promotion[s[:, None], fi]
        pc, ps = self.state["promotion_stats"]
        month = panel.dates.month.to_numpy()[fi]
        day = panel.dates.day.to_numpy()[fi]
        flags = panel.calendar[s[:, None], fi]
        future = np.stack(
            [
                (np.log1p(p) - pc) / ps,
                (p > 0).astype(float),
                np.sin(2 * np.pi * dow[fi] / 7),
                np.cos(2 * np.pi * dow[fi] / 7),
                day / 31,
                np.sin(2 * np.pi * month / 12),
                np.cos(2 * np.pi * month / 12),
                ((day == 15) | np.asarray(panel.dates.is_month_end)[fi]).astype(float),
                *(flags[:, :, i] for i in range(4)),
            ],
            axis=-1,
        ).astype(np.float32)
        codes = []
        for i, c in enumerate(STATIC_CHANNELS):
            vocabulary = {v: k for k, v in enumerate(self.state["vocabularies"][c])}
            codes.append([vocabulary.get(panel.static[j][i], 0) for j in s])
        actual = panel.sales[s[:, None], fi]
        target = (np.log1p(actual) - center) / scale
        return {
            "history": history,
            "future": future,
            "static": np.asarray(codes, dtype=np.int64).T,
            "target_z": target.astype(np.float32),
            "target_observed": np.isfinite(actual),
            "center": center,
            "scale": scale,
        }
