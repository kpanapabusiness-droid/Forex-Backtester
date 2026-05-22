"""SHBSignalModule — adapts the SHB (swing-high breakout in trend, long, 4H)
signal layer to the :class:`core.arc.signal_protocol.SignalModule` Protocol.

The signal producer at ``signals.lchar_swing_high_breakout_trend.compute_signal``
already implements the SHB rule with causal 3-bar swing detection and
right-edge-at-t-4 constraint (see
``docs/archive/signal_specs/signal_swing_high_breakout_trend_long_v0.1.md``
for the full spec; intent-doc §3 contains the producer-level causal trace).

This module wraps that producer to expose the SignalModule contract for the
v3 canonical orchestrator (``core/arc/arc_orchestrator.py``). It does NOT
duplicate signal logic — it constructs a bid-OHLC view from the v3 H4
panel (per anchor convention: long-side signal evaluated on bid prices)
and surfaces the producer's ``signal`` + ``atr14`` columns.

SHB has no signal-class-inherent filter beyond its built-in trend filter
(swing-low → close[t-1] > min) and no signal-class exit predicate beyond
the time-exit + hard SL (those live in the architecture config).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from core.arc.signal_protocol import (
    PerPairSignalState,
    SignalEvaluation,
    SignalModule,
    validate_panels,
)
from core.sim.panel import Panel
from signals.lchar_swing_high_breakout_trend import compute_signal


@dataclass(frozen=True)
class SHBSignalModule(SignalModule):
    """SignalModule implementation for the SHB long signal.

    Locked params live inside the producer module
    (``signals.lchar_swing_high_breakout_trend``): SWING_K=3,
    RIGHT_EDGE_OFFSET=4, BREAK_BUFFER_ATR=0.10, CLOSE_UPPER_HALF_MIN=0.5,
    REFRACTORY_BARS=20, ATR_PERIOD=14 (Wilder).
    """

    signal_name: str = "shb_swing_high_breakout_trend_long"
    primary_tf: str = "H4"
    auxiliary_tfs: tuple[str, ...] = ()
    causal_lineage: str = "clean"

    def evaluate(self, panels: Mapping[str, Panel]) -> SignalEvaluation:
        validate_panels(self, panels)
        h4 = panels[self.primary_tf]
        per_pair: dict[str, PerPairSignalState] = {}
        for pair in sorted(h4.pairs):
            df_h4 = h4.pair_dfs[pair]
            # Build bid-OHLC view for the producer (signal evaluated on bid
            # per v3 anchor convention — long-side signal needs the bid to
            # break above H_ref + buffer × ATR).
            bid_df = pd.DataFrame(
                {
                    "open": df_h4["open_bid"].astype(float),
                    "high": df_h4["high_bid"].astype(float),
                    "low": df_h4["low_bid"].astype(float),
                    "close": df_h4["close_bid"].astype(float),
                },
                index=df_h4.index,
            )
            bid_df["date"] = bid_df.index
            sig_out = compute_signal(bid_df, signal_col="signal")
            # sig_out has int 0..n-1 index after compute_signal's reset_index;
            # re-align to df_h4.index by position.
            sig_out.index = df_h4.index
            signal_mask = sig_out["signal"].astype(bool)
            atr = sig_out["atr14"].astype(float)
            per_pair[pair] = PerPairSignalState(
                signal_mask=signal_mask,
                atr=atr,
                additional_gates={},  # SHB has no extra gates
                exit_predicate=None,   # SHB has no signal-class exit
                path_feature_anchor=df_h4.index,
            )
        return SignalEvaluation(
            primary_tf=self.primary_tf,
            per_pair=per_pair,
            signal_name=self.signal_name,
            causal_lineage=self.causal_lineage,
        )


__all__ = ("SHBSignalModule",)
