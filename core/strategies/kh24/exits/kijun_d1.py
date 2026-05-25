"""``kijun_d1`` exit predicate for KH-24.

Matches the deployed MT5 EA ``KH24_EA.mq5`` (see
``reference/kh24_ea/KH24_EA.mq5`` lines 456-469). The exit condition is:

    prev_d1_close < prev_d1_kijun

— **both** measured at the just-closed D1 bar (shift=1, lag-1 rule per
L_PROTOCOL §1). The EA checks the **D1 close**, NOT the current H4
bar's close. This is a D1-resolution exit: it can only change when a
new D1 bar closes, then stays in force across all H4 bars of the
following day until the next D1 close moves back above the Kijun.

Both quantities are computed on **bid-side single OHLC** because
that's what MT5's ``CopyRates`` returns by broker convention — the EA
reads single-side ticks, so the v3 port must mirror that to reproduce
EA decisions. See ``docs/dispatches/kh24_fix_diff.md`` Section A.

The predicate fires on every H4 bar's exit-check step in the v3
driver; while the underlying D1 condition only flips at D1 close
boundaries, evaluating per-H4-bar is consistent with the EA's
ProcessExits-per-4H-bar pattern (the EA also evaluates it 6 times per
day; same answer until a new D1 closes).

Causal lineage: clean. Both inputs are from the strictly-prior D1
bar; perturbing the same-day D1 has zero impact on the predicate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.signals.htf_alignment import get_htf_row_at
from core.sim.account import Direction, Position
from core.sim.exit_hooks import ExitDecision, ExitPredicate
from core.sim.fill import long_exit_market_price


def _kijun_bid(df_d1: pd.DataFrame, period: int) -> pd.Series:
    """Ichimoku Kijun-sen on bid-side OHLC: (max(high_bid) + min(low_bid)) / 2."""
    hi = df_d1["high_bid"].rolling(window=period, min_periods=period).max()
    lo = df_d1["low_bid"].rolling(window=period, min_periods=period).min()
    return (hi + lo) / 2.0


def _build_d1_lag1_close_and_kijun(
    h4_index: pd.DatetimeIndex, df_d1: pd.DataFrame, kijun_period: int = 26
) -> tuple[pd.Series, pd.Series]:
    """Return ``(d1_close_lag1, d1_kijun_lag1)`` aligned to ``h4_index``.

    Both are computed on bid-side single OHLC (matches EA's ``CopyRates``
    convention) and aligned to the most-recently fully-closed D1 bar at
    each H4 timestamp (under whatever timezone convention the panels
    share — L_PROTOCOL §1 lag-1 rule). Byte-identical to the legacy
    normalize-and-shift idiom under UTC convention; fixes the silent
    same-EET-day lookahead under 5ers EET convention.
    """
    d1_panel = pd.DataFrame(
        {
            "d1_close": df_d1["close_bid"].values.astype(float),
            "d1_kijun": _kijun_bid(df_d1, period=kijun_period).values.astype(float),
        },
        index=df_d1.index,
    )
    rows = get_htf_row_at(h4_index, d1_panel, require_fully_closed=True)
    d1_close = pd.Series(rows["d1_close"].to_numpy(dtype=float), index=h4_index, name="d1_close_lag1")
    d1_kijun = pd.Series(rows["d1_kijun"].to_numpy(dtype=float), index=h4_index, name="d1_kijun_lag1")
    return d1_close, d1_kijun


def make_kijun_d1_exit_predicate(
    pair: str, df_h4: pd.DataFrame, df_d1: pd.DataFrame, kijun_period: int = 26
) -> ExitPredicate:
    """Return an ``ExitPredicate`` that fires when prev D1 close < prev D1 Kijun.

    Matches ``KH24_EA.mq5`` ProcessExits (lines 456-469). Fill price for
    the long market exit is the bid (``close_bid`` via PR-B's
    ``long_exit_market_price``).
    """
    d1_close_lag1, d1_kijun_lag1 = _build_d1_lag1_close_and_kijun(
        df_h4.index, df_d1, kijun_period=kijun_period
    )

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
        try:
            d1c = float(d1_close_lag1.loc[t])
            d1k = float(d1_kijun_lag1.loc[t])
        except KeyError:
            return None
        if not (np.isfinite(d1c) and np.isfinite(d1k)):
            return None
        if d1c < d1k:
            return ExitDecision(
                fill_price=long_exit_market_price(bar), exit_reason="kijun_d1"
            )
        return None

    return predicate
