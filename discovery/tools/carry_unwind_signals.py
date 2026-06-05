"""Carry-unwind cascade SHORT SignalModule (EXPERIMENT tool).

Built by arc 1017 (chat 1000s). EXPERIMENT tool per `discovery/TOOL_REGISTRY.md`: defines an entry
MASK + ATR (price geometry) ONLY; never realizes P&L. Scoring routes through the canonical apparatus
(`build_arc_pool`, `ArcFoldRunner` -> `MultiPairBacktester`). Conforms to
`core.arc.signal_protocol.SignalModule`.

Mechanism (the *because*): leveraged carry trades (long a high-yield risk cross / short JPY) build up
during calm risk-on as the cross grinds UP. When a risk-off shock hits, those positions are
force-unwound — JPY is bought back — and because the unwind is a forced-deleveraging cascade it is
PERSISTENT and one-directional ("up the stairs, down the elevator"). The IGNITION is a vol-expansion
big-RED bar fired from a built-up carry position (the cross in an uptrend). Betting on forward
continuation DOWN means entering at the *start* of the unwind (forward-confirming, arc-1013 lesson),
not after an established downtrend (arc 3010, which reverts).

Whether the carry-UPTREND context is load-bearing (vs a generic big-red bar, which mean-reverts UP)
is the structure control arc 1017 runs. Intended universe = JPY carry crosses (AUD/NZD/EUR/GBP/CAD
JPY); intended TF = H4.

Ex-ante (no-lookahead): SMA(sma_period) on MID close and its 20-bar slope read bar i's own close
(known at bar i close). vol-ignition true-range and the signed body read bar i's own OHLC. ATR is
Wilder(`atr_period`) on MID, shift(1) (strictly prior bars). The signal fires at bar i close; the
pool/engine enters at the next bar's open (sell at open_bid). No future bar is read.

arc 1017 finding: see DISCOVERY_LOG / arc doc.
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
from core.sim.account import Direction
from core.sim.panel import Panel
from discovery.tools.trend_entry_signals import _atr_shift1_mid


@dataclass(frozen=True)
class CarryUnwindCascadeShortSignal:
    """Short the ignition bar of a carry-unwind cascade on a JPY carry cross.

    ``sma_period``: trend/carry-build-up filter window (close_mid > SMA AND SMA rising over 20 bars).
    ``vol_ignition``: minimum true-range/ATR ratio of bar i (a volatility-expansion bar).
    ``down_atr``: minimum DOWN body (open_mid - close_mid)/ATR of bar i (a big-red ignition bar).
    """

    sma_period: int = 100
    vol_ignition: float = 1.5
    down_atr: float = 1.0
    atr_period: int = 14
    signal_name: str = "carry_unwind_cascade_short_v0.1"
    primary_tf: str = "H4"
    auxiliary_tfs: tuple[str, ...] = ()
    causal_lineage: str = "clean"
    direction: str = "short"

    def required_aux_data(self) -> list[str]:
        return list(self.auxiliary_tfs)

    def _name(self) -> str:
        return f"carry_unwind_short_sma{self.sma_period}_vi{self.vol_ignition:.2f}_d{self.down_atr:.2f}_v0.1"

    def evaluate(self, panels: Mapping[str, Panel]) -> SignalEvaluation:
        primary = panels[self.primary_tf]
        per_pair: dict[str, PerPairSignalState] = {}
        for pair in sorted(primary.pairs):
            df = primary.pair_dfs[pair]
            n = len(df)
            idx = df.index
            open_mid = (df["open_bid"].to_numpy(float) + df["open_ask"].to_numpy(float)) / 2.0
            high_mid = (df["high_bid"].to_numpy(float) + df["high_ask"].to_numpy(float)) / 2.0
            low_mid = (df["low_bid"].to_numpy(float) + df["low_ask"].to_numpy(float)) / 2.0
            close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
            atr = _atr_shift1_mid(df, self.atr_period)

            c = pd.Series(close_mid, index=idx)
            sma = c.rolling(self.sma_period).mean()
            sma_slope = (sma - sma.shift(20)).to_numpy(float)
            sma_v = sma.to_numpy(float)
            prev_close = c.shift(1).to_numpy(float)
            tr = np.maximum(high_mid, prev_close) - np.minimum(low_mid, prev_close)  # true range bar i

            with np.errstate(invalid="ignore"):
                vol_ratio = tr / atr
                body_down = (open_mid - close_mid) / atr  # positive = red bar

            fire = (
                (close_mid > sma_v)                  # carry built up (cross in uptrend)
                & (sma_slope > 0)                    # trend rising
                & (vol_ratio >= self.vol_ignition)   # vol-expansion ignition bar
                & (body_down >= self.down_atr)       # big-red ignition body
                & np.isfinite(atr) & (atr > 0)
                & np.isfinite(sma_v) & np.isfinite(sma_slope)
            )
            mask = np.zeros(n, dtype=bool)
            mask[fire] = True

            per_pair[pair] = PerPairSignalState(
                signal_mask=pd.Series(mask, index=idx, name="signal_mask"),
                atr=pd.Series(atr, index=idx, name="atr_14_shift1"),
                additional_gates={},
                exit_predicate=None,
                path_feature_anchor=None,
                direction=Direction.SHORT,
            )
        return SignalEvaluation(
            primary_tf=self.primary_tf,
            per_pair=per_pair,
            signal_name=self._name(),
            causal_lineage=self.causal_lineage,
            direction=Direction.SHORT,
        )


assert isinstance(CarryUnwindCascadeShortSignal(), SignalModule)

__all__ = ("CarryUnwindCascadeShortSignal",)
