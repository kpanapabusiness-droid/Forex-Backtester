"""Failed-breakout REJECTION (stop-run reversal) SHORT SignalModule (EXPERIMENT tool).

Built by arc 3011 (chat 3000s). The SHORT MIRROR of arc 1013's
``FailedBreakdownReclaimLongSignal`` (the strongest directional edge in the corpus). EXPERIMENT
tool per `discovery/TOOL_REGISTRY.md`: defines an entry MASK + ATR (price geometry) ONLY; never
realizes P&L. Scoring routes through the canonical apparatus (`build_arc_pool`, `ArcFoldRunner` ->
`MultiPairBacktester`). Conforms to `core.arc.signal_protocol.SignalModule`.

Mechanism (the mirror of arc 1013): resting buy-stops cluster just ABOVE visible swing highs. A bar
that PIERCES a K-bar swing high (high_ask > prior K-bar max high — sweeping the stops above) and then
CLOSES BACK BELOW it (close_mid < that level — the breakOUT FAILED), with a LARGE upper rejection
shadow (>= min_shadow_atr ATR), is a stop-run / liquidity grab on the upside: the up-move was
liquidity-driven, not informational, and the rejection is the confirmation the favorable-for-short
move is starting *at entry*. Arc 1013 explicitly flagged "the up-sweep is arguably the stronger leg."

Whether the SWING-HIGH structure is load-bearing for the SHORT (the way the swing-low was for the
1013 long) is the empirical question arc 3011 tests with a structure control (big-reject-AT-swept-high
vs the SAME wick elsewhere) — mirroring arcs 1014/2009, which found the swing-LOW sweep has no
tradeable short. This is the opposite structure (swing HIGH), so it is a genuinely fresh test.

Ex-ante (no-lookahead): the swing high uses high_ask.shift(1).rolling(K).max() (strictly prior K bars,
excludes bar i). pierce / reject / shadow all read bar i's own OHLC (known at bar i close). ATR is
Wilder(14) on MID, shift(1) (strictly prior bars). The signal fires at bar i close; the pool/engine
enters at the next bar's open (sell at open_bid). No future bar is read. Intended TF = H4, USD majors.
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
class FailedBreakoutRejectionShortSignal:
    """Short a deep failed-breakout rejection (stop-run reversal) at a K-bar swing high.

    ``swing_lookback`` (K): bars in the swing-high window (prior K-bar max high). Larger = more
        significant level / denser stops (mirror of arc 1013: K=40-60 best for the long).
    ``min_shadow_atr``: minimum upper-rejection-shadow (high_ask - max(open_mid,close_mid))/atr to
        fire — the rejection-strength gate (mirror of arc 1013's 1.0-1.5).
    """

    swing_lookback: int = 40
    min_shadow_atr: float = 1.25
    atr_period: int = 14
    signal_name: str = "failed_breakout_rejection_short_v0.1"
    primary_tf: str = "H4"
    auxiliary_tfs: tuple[str, ...] = ()
    causal_lineage: str = "clean"
    direction: str = "short"

    def required_aux_data(self) -> list[str]:
        return list(self.auxiliary_tfs)

    def _name(self) -> str:
        return f"fbr_short_K{self.swing_lookback}_sh{self.min_shadow_atr:.2f}_v0.1"

    def evaluate(self, panels: Mapping[str, Panel]) -> SignalEvaluation:
        primary = panels[self.primary_tf]
        per_pair: dict[str, PerPairSignalState] = {}
        for pair in sorted(primary.pairs):
            df = primary.pair_dfs[pair]
            n = len(df)
            idx = df.index
            high_ask = df["high_ask"].to_numpy(float)
            open_mid = (df["open_bid"].to_numpy(float) + df["open_ask"].to_numpy(float)) / 2.0
            close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
            atr = _atr_shift1_mid(df, self.atr_period)

            prior_high = pd.Series(high_ask, index=idx).shift(1).rolling(self.swing_lookback).max().to_numpy(float)
            with np.errstate(invalid="ignore"):
                shadow = (high_ask - np.maximum(open_mid, close_mid)) / atr

            fire = (
                (high_ask > prior_high)            # pierced the swing high (swept stops above)
                & (close_mid < prior_high)         # rejected back below it (failed breakout)
                & (shadow >= self.min_shadow_atr)  # deep upper rejection wick
                & np.isfinite(atr) & (atr > 0)
                & np.isfinite(prior_high)
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


assert isinstance(FailedBreakoutRejectionShortSignal(), SignalModule)

__all__ = ("FailedBreakoutRejectionShortSignal",)
