"""Failed-breakdown RECLAIM (stop-run reversal) long SignalModule (EXPERIMENT tool).

Built by arc 1013 (chat 1000s). EXPERIMENT tool per `discovery/TOOL_REGISTRY.md`: defines an entry
MASK + ATR (price geometry) ONLY; never realizes P&L. Scoring routes through the canonical apparatus
(`build_arc_pool`, `ArcFoldRunner` -> `MultiPairBacktester`). Conforms to
`core.arc.signal_protocol.SignalModule`.

Mechanism (arc 1013 observation + control): resting sell-stops cluster just below visible swing lows.
A bar that PIERCES a K-bar swing low (low_bid < prior K-bar min low — sweeping the stops) and then
CLOSES BACK ABOVE it (close_mid > that level — the breakdown FAILED), with a LARGE lower rejection
shadow (>= min_shadow_atr ATR), is a stop-run / liquidity grab: the down-move was liquidity-driven,
not informational, and the reclaim is the confirmation the adverse excursion is OVER *at entry* (the
fix to the gap-fill capturability wall, arcs 2001/2002, which entered INSIDE the continuation).

The arc-1013 CONTROL proved the STRUCTURE is load-bearing, not the generic wick: a same-magnitude
rejection wick AT a swept swing low captures 0.52-0.61 (grows with shadow), while the SAME wick NOT at
a swing low stays a coin-flip ~0.49-0.51 with negative drift (generic reversion is dead, arcs
3000/3001). So this is NOT "buy a big down-wick" — it is a structural stop-run reclaim.

Ex-ante (no-lookahead): the swing low uses low_bid.shift(1).rolling(K).min() (strictly prior K bars,
excludes bar i). pierce / reclaim / shadow all read bar i's own OHLC (known at bar i close). ATR is
Wilder(14) on MID, shift(1) (strictly prior bars). The signal fires at bar i close; the pool/engine
enters at the next bar's open. No future bar is read. Intended TF = H4, USD majors.
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
from core.sim.panel import Panel
from discovery.tools.trend_entry_signals import _atr_shift1_mid


@dataclass(frozen=True)
class FailedBreakdownReclaimLongSignal:
    """Long a deep failed-breakdown reclaim (stop-run reversal) at a K-bar swing low.

    ``swing_lookback`` (K): bars in the swing-low window (prior K-bar min low). Larger = more
        significant level / denser stops (arc 1013: K=40-60 best).
    ``min_shadow_atr``: minimum lower-rejection-shadow (min(open_mid,close_mid) - low_bid)/atr to
        fire — the rejection-strength gate (arc 1013: control excess grows with this; 1.0-1.5).
    """

    swing_lookback: int = 40
    min_shadow_atr: float = 1.25
    atr_period: int = 14
    signal_name: str = "failed_breakdown_reclaim_long_v0.1"
    primary_tf: str = "H4"
    auxiliary_tfs: tuple[str, ...] = ()
    causal_lineage: str = "clean"

    def required_aux_data(self) -> list[str]:
        return list(self.auxiliary_tfs)

    def _name(self) -> str:
        return f"fbr_long_K{self.swing_lookback}_sh{self.min_shadow_atr:.2f}_v0.1"

    def evaluate(self, panels: Mapping[str, Panel]) -> SignalEvaluation:
        primary = panels[self.primary_tf]
        per_pair: dict[str, PerPairSignalState] = {}
        for pair in sorted(primary.pairs):
            df = primary.pair_dfs[pair]
            n = len(df)
            idx = df.index
            low_bid = df["low_bid"].to_numpy(float)
            open_mid = (df["open_bid"].to_numpy(float) + df["open_ask"].to_numpy(float)) / 2.0
            close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
            atr = _atr_shift1_mid(df, self.atr_period)

            prior_low = pd.Series(low_bid, index=idx).shift(1).rolling(self.swing_lookback).min().to_numpy(float)
            with np.errstate(invalid="ignore"):
                shadow = (np.minimum(open_mid, close_mid) - low_bid) / atr

            fire = (
                (low_bid < prior_low)             # pierced the swing low (swept stops)
                & (close_mid > prior_low)         # reclaimed above it (failed breakdown)
                & (shadow >= self.min_shadow_atr) # deep rejection wick
                & np.isfinite(atr) & (atr > 0)
                & np.isfinite(prior_low)
            )
            mask = np.zeros(n, dtype=bool)
            mask[fire] = True

            per_pair[pair] = PerPairSignalState(
                signal_mask=pd.Series(mask, index=idx, name="signal_mask"),
                atr=pd.Series(atr, index=idx, name="atr_14_shift1"),
                additional_gates={},
                exit_predicate=None,
                path_feature_anchor=None,
            )
        return SignalEvaluation(
            primary_tf=self.primary_tf,
            per_pair=per_pair,
            signal_name=self._name(),
            causal_lineage=self.causal_lineage,
        )


assert isinstance(FailedBreakdownReclaimLongSignal(), SignalModule)

__all__ = ("FailedBreakdownReclaimLongSignal",)
