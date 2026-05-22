"""``kijun_d1`` exit predicate for KH-24.

Closes a long position when the bar-close mid-price drops below the
prior D1 Kijun(26). Uses the L_PROTOCOL §1 one-day-lag alignment so
the D1 Kijun referenced is from calendar day T-1 (never same-day).

Implemented as a closure factory that captures a per-pair lookup table
of lag-1 D1 Kijun values indexed by H4 timestamp. The closure
satisfies the ``ExitPredicate`` signature from ``core.sim.exit_hooks``.

Causal lineage: clean. D1 Kijun(26) is computed on D1 bars closed
strictly before the H4 bar's calendar day; perturbing the same-day D1
has zero impact on the predicate output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.features._helpers import kijun, mid_high, mid_low
from core.sim.account import Direction, Position
from core.sim.exit_hooks import ExitDecision, ExitPredicate
from core.sim.fill import long_exit_market_price


def _build_d1_kijun_lag1(
    h4_index: pd.DatetimeIndex, df_d1: pd.DataFrame, kijun_period: int = 26
) -> pd.Series:
    """Return D1 Kijun(period) aligned to ``h4_index`` with one-day lag.

    ``shift the H4 calendar date back one day, then merge_asof backward
    against the D1 series'' — same pattern as the signal evaluator's
    `_build_d1_lag1_arrays`.
    """
    d1 = pd.DataFrame(index=df_d1.index.copy())
    d1["d1_kijun"] = kijun(mid_high(df_d1), mid_low(df_d1), period=kijun_period).values
    d1["_date"] = d1.index.normalize()
    d1 = d1.drop_duplicates(subset=["_date"], keep="last").reset_index(drop=True)

    shifted = pd.DataFrame(
        {
            "_date": h4_index.normalize() - pd.Timedelta(days=1),
            "_idx": np.arange(len(h4_index), dtype=np.int64),
        }
    ).sort_values("_date")
    merged = pd.merge_asof(shifted, d1[["_date", "d1_kijun"]], on="_date", direction="backward")
    merged = merged.sort_values("_idx").reset_index(drop=True)
    return pd.Series(merged["d1_kijun"].values, index=h4_index, name="d1_kijun_lag1")


def make_kijun_d1_exit_predicate(
    pair: str, df_h4: pd.DataFrame, df_d1: pd.DataFrame, kijun_period: int = 26
) -> ExitPredicate:
    """Return an ``ExitPredicate`` that fires when H4 mid-close < D1 Kijun lag-1.

    The predicate only applies to positions on ``pair`` (other pairs
    fall through to subsequent predicates). Fill price is the long
    market-exit (close_bid).
    """
    d1_kijun_lag1 = _build_d1_kijun_lag1(df_h4.index, df_d1, kijun_period=kijun_period)

    def predicate(
        position: Position, snapshot: dict[str, pd.Series | None], t: pd.Timestamp
    ) -> ExitDecision | None:
        if position.pair != pair:
            return None
        if position.direction is not Direction.LONG:
            return None
        bar = snapshot.get(pair)
        if bar is None:
            return None
        cb = bar.get("close_bid")
        ca = bar.get("close_ask")
        if pd.isna(cb) or pd.isna(ca):
            return None
        mid_close_val = float((cb + ca) / 2.0)
        try:
            k = float(d1_kijun_lag1.loc[t])
        except KeyError:
            return None
        if not np.isfinite(k):
            return None
        if mid_close_val < k:
            return ExitDecision(fill_price=long_exit_market_price(bar), exit_reason="kijun_d1")
        return None

    return predicate
