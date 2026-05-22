"""Real-spread accounting on the HistData bid+ask panel.

The data layer (``core.data.histdata_loader``, ``core.data.aggregator``)
already computes per-bar ``spread_close = close_ask - close_bid`` and tags
each bar with ``bid_ask_data_quality ∈ {ok, zero_or_negative_spread,
nan_bid_or_ask}``. This module exposes the thin "trade through this bar?"
predicate and a summary helper that walks an arbitrary
DatetimeIndex'd OHLC frame.

L_PROTOCOL §1 non-negotiable: zero or NaN bid/ask is a data-quality flag,
not silently backfilled. No fallback mechanism — strict drop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from core.data.histdata_loader import (
    DQ_NAN_BID_OR_ASK,
    DQ_OK,
    DQ_ZERO_OR_NEG_SPREAD,
)


def per_bar_spread(df: pd.DataFrame) -> pd.Series:
    """Return the per-bar spread (in price units) for ``df``.

    Equivalent to ``df['spread_close']`` for frames produced by the v3 data
    layer; defined as a function for forward compatibility with alternate
    spread aggregations (e.g. minute-mean) that callers may want in the
    future.
    """
    return df["spread_close"]


def is_tradable_bar(df: pd.DataFrame) -> pd.Series:
    """Boolean mask: True iff the bar is safe to simulate a trade through.

    A bar is tradable iff its ``bid_ask_data_quality`` is ``DQ_OK``. Bars
    flagged ``DQ_ZERO_OR_NEG_SPREAD`` or ``DQ_NAN_BID_OR_ASK`` are dropped
    from trade simulation by every caller (no fallback).
    """
    return df["bid_ask_data_quality"] == DQ_OK


@dataclass(frozen=True)
class DataQualitySummary:
    """Per-frame breakdown of ``bid_ask_data_quality`` counts."""

    pair: str | None
    tf: str | None
    total_bars: int
    ok: int
    zero_or_neg_spread: int
    nan_bid_or_ask: int

    @property
    def bad_total(self) -> int:
        return self.zero_or_neg_spread + self.nan_bid_or_ask

    @property
    def bad_pct(self) -> float:
        return self.bad_total / self.total_bars if self.total_bars else 0.0

    def as_dict(self) -> dict:
        return {
            "pair": self.pair,
            "tf": self.tf,
            "total_bars": self.total_bars,
            "ok": self.ok,
            "zero_or_neg_spread": self.zero_or_neg_spread,
            "nan_bid_or_ask": self.nan_bid_or_ask,
            "bad_total": self.bad_total,
            "bad_pct": self.bad_pct,
        }


def data_quality_summary(
    df: pd.DataFrame, pair: str | None = None, tf: str | None = None
) -> DataQualitySummary:
    """Count rows per ``bid_ask_data_quality`` bucket on ``df``.

    ``pair`` and ``tf`` are pass-through tags so the caller can collate
    summaries from many pairs/TFs into a single report.
    """
    counts = df["bid_ask_data_quality"].value_counts()
    return DataQualitySummary(
        pair=pair,
        tf=tf,
        total_bars=int(len(df)),
        ok=int(counts.get(DQ_OK, 0)),
        zero_or_neg_spread=int(counts.get(DQ_ZERO_OR_NEG_SPREAD, 0)),
        nan_bid_or_ask=int(counts.get(DQ_NAN_BID_OR_ASK, 0)),
    )


def collate_summaries(summaries: Iterable[DataQualitySummary]) -> dict:
    """Aggregate many per-pair/TF summaries into totals.

    Returns a dict with keys ``total_bars, ok, zero_or_neg_spread,
    nan_bid_or_ask, bad_total, bad_pct, n_summaries``.
    """
    summaries = list(summaries)
    total = sum(s.total_bars for s in summaries)
    ok = sum(s.ok for s in summaries)
    zns = sum(s.zero_or_neg_spread for s in summaries)
    nan = sum(s.nan_bid_or_ask for s in summaries)
    bad = zns + nan
    return {
        "n_summaries": len(summaries),
        "total_bars": total,
        "ok": ok,
        "zero_or_neg_spread": zns,
        "nan_bid_or_ask": nan,
        "bad_total": bad,
        "bad_pct": (bad / total) if total else 0.0,
    }
