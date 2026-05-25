"""Liquidity sweep + reclaim long signal — 4H structural reversal.

Spec (locked v0.1, source: ``docs/archive/arc_results/ARC_7_RESULT.md`` Trigger):

    swing_low_N    = min(low_bid[t-N..t-1]),  N = 20
    low_bid[t]    < swing_low_N                                # sweep
    close_bid[t]  > swing_low_N                                # reclaim
    swing_low_N − low_bid[t] ≥ 0.25 × ATR(14)[t]               # magnitude
    close_bid[t] > open_bid[t]                                 # bullish bar
    (close_bid[t] − swing_low_N) /
        (swing_low_N − low_bid[t]) ≥ 0.5                       # reclaim strength
    gap_since_last_signal_on_pair ≥ 20 bars                    # refractory

Bid-side OHLC throughout — matches the KH-24 v3 convention. Long entry fills at
next bar's ``open_ask`` via :mod:`core.sim.fill` primitives (handled in
``core.arc.arc_pool_builder.build_arc_pool``). PR #189's mid-price refactor
applies to Step-1 FEATURES, not to signal evaluation per
``docs/PROTOCOL_RUNTIME.md`` §15.1.

ATR uses :func:`core.features._helpers.wilder_atr` on (high_bid, low_bid,
close_bid) — same period 14 used by KH-24.

Causal lineage = ``clean``. Producer-level trace (Arc 9 lesson):

  - ``swing_low[t]`` = ``low_bid.rolling(N).min().shift(1).iloc[t]`` →
    pandas semantics: rolling(N).min() at index t returns min over [t-N+1, t];
    .shift(1) shifts the entire series forward by one index, so the value
    AT index t becomes the value previously at index t-1. Net effect:
    ``swing_low.iloc[t] = min(low_bid.iloc[t-N], ..., low_bid.iloc[t-1])``.
    ``low_bid.iloc[t]`` is NEVER in the window for ``swing_low.iloc[t]``.
    No centred-window pattern; no right-edge lookahead.
  - ``ATR(14)[t]`` — Wilder EWM on shifted TR. TR at bar t uses high[t],
    low[t], close[t-1] — all known by bar t's close.
  - ``low_bid[t]``, ``close_bid[t]``, ``open_bid[t]`` — current bar, known
    at bar t's close.
  - ``gap_since_last_signal`` — stateful walk over prior True indices on
    the same pair only.

All inputs at signal bar t are known by t's close. Verified by
``tests/test_liquidity_sweep_reclaim_long_signal.py``.

Step 6 §6.1 lookahead audit exercises this producer via byte-compare from
raw OHLC on a random trade sample per L_PROTOCOL Amendment 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from core.arc.signal_protocol import (
    PerPairSignalState,
    SignalEvaluation,
    SignalModule,
)
from core.features._helpers import wilder_atr
from core.sim.panel import Panel


@dataclass(frozen=True)
class LSRSignalParams:
    """Locked parameters for the liquidity_sweep_reclaim_long signal."""

    swing_window: int = 20          # N in swing_low_N
    atr_period: int = 14            # Wilder ATR period
    magnitude_atr_mult: float = 0.25  # sweep depth threshold (× ATR)
    reclaim_strength_min: float = 0.5  # (close − swing_low) / (swing_low − low)
    refractory_bars: int = 20       # gap between signals on the same pair


def _evaluate_pair(
    pair_df: pd.DataFrame, params: LSRSignalParams
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the LSR trigger to one pair's H4 frame.

    Returns ``(signal_mask, atr_h4)`` aligned to ``pair_df.index``. The
    signal_mask is True at the signal-bar close (bar N); the arc-pool
    builder fills the entry at bar N+1's open.
    """
    n = len(pair_df)
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float)

    high_bid = pair_df["high_bid"]
    low_bid = pair_df["low_bid"]
    close_bid = pair_df["close_bid"]
    open_bid = pair_df["open_bid"]

    atr = wilder_atr(high_bid, low_bid, close_bid, period=params.atr_period)
    atr_arr = atr.values.astype(float)

    low_arr = low_bid.values
    close_arr = close_bid.values
    open_arr = open_bid.values

    # swing_low_N = min(low_bid[t-N..t-1]) — rolling over prior N bars (exclude
    # current bar). pd.Series.rolling(window=N).min().shift(1) gives this.
    # Producer trace in module docstring.
    swing_low = (
        low_bid.rolling(window=params.swing_window, min_periods=params.swing_window).min().shift(1)
    )
    swing_arr = swing_low.values

    warm = max(params.swing_window, params.atr_period) + 1
    sig = np.zeros(n, dtype=bool)

    last_sig_idx = -10_000  # large negative so first signal is allowed
    for i in range(warm, n):
        a = atr_arr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        s = swing_arr[i]
        if not np.isfinite(s):
            continue
        lo = low_arr[i]
        c = close_arr[i]
        o = open_arr[i]
        # C1: sweep — low pierces swing low
        if not (lo < s):
            continue
        # C2: reclaim — close above swing low
        if not (c > s):
            continue
        # C3: magnitude — sweep depth >= 0.25 × ATR
        depth = s - lo
        if not (depth >= params.magnitude_atr_mult * a):
            continue
        # C4: bullish reclaim bar
        if not (c > o):
            continue
        # C5: reclaim strength ratio
        ratio = (c - s) / depth if depth > 0 else 0.0
        if not (ratio >= params.reclaim_strength_min):
            continue
        # C6: refractory — at least `refractory_bars` since last signal on this pair
        if i - last_sig_idx < params.refractory_bars:
            continue
        sig[i] = True
        last_sig_idx = i

    return sig, atr_arr


@dataclass(frozen=True)
class LiquiditySweepReclaimLongSignal:
    """L_PROTOCOL v3.0 SignalModule for Arc 7 — liquidity sweep + reclaim long.

    Implements the :class:`core.arc.signal_protocol.SignalModule` Protocol.
    """

    signal_name: str = "liquidity_sweep_reclaim_long"
    primary_tf: str = "H4"
    auxiliary_tfs: tuple[str, ...] = ()
    causal_lineage: str = "clean"
    params: LSRSignalParams = LSRSignalParams()

    def required_aux_data(self) -> list[str]:
        """Backward-compat with SignalAdapter callers (KH-24 legacy);
        SignalModule callers ignore this."""
        return list(self.auxiliary_tfs)

    def evaluate(self, panels: Mapping[str, Panel]) -> SignalEvaluation:
        """Apply the signal to every pair in panels[primary_tf]."""
        primary = panels[self.primary_tf]
        per_pair: dict[str, PerPairSignalState] = {}
        for pair in sorted(primary.pairs):
            df = primary.pair_dfs[pair]
            sig_arr, atr_arr = _evaluate_pair(df, self.params)
            per_pair[pair] = PerPairSignalState(
                signal_mask=pd.Series(sig_arr, index=df.index, name="signal_mask"),
                atr=pd.Series(atr_arr, index=df.index, name="atr"),
                additional_gates={},  # no signal-class filters for LSR
                exit_predicate=None,  # no signal-class exits for LSR
                path_feature_anchor=None,
            )
        return SignalEvaluation(
            primary_tf=self.primary_tf,
            per_pair=per_pair,
            signal_name=self.signal_name,
            causal_lineage=self.causal_lineage,
        )


# Protocol conformance check (runtime).
assert isinstance(LiquiditySweepReclaimLongSignal(), SignalModule)


__all__ = ("LiquiditySweepReclaimLongSignal", "LSRSignalParams")
