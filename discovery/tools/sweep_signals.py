"""Liquidity-sweep reversal long SignalModule (EXPERIMENT tool, BUILT by arc 2007, chat 2000s).

EXPERIMENT tool per `discovery/TOOL_REGISTRY.md`: defines an entry MASK + ATR (price geometry)
ONLY; never realizes P&L. Scoring routes through the canonical apparatus (`build_arc_pool`,
`ArcFoldRunner` -> `MultiPairBacktester`). Conforms to `core.arc.signal_protocol.SignalModule`.

BECAUSE (order-flow / structural): a DOWN move that pierces a well-defined prior swing low
(running resting sell-stops + triggering breakout-shorts) and then IMMEDIATELY reclaims that low
within the same bar is a classic liquidity grab -- a large buyer used the stop-cascade liquidity to
fill, so the path should revert UP. Each single condition is a coin-flip (corpus-proven). The DEEP
claim under test: the multi-factor CONJUNCTION
  (sweep below prior L-bar low) x (close back above it) x (fast 3-bar drop into it)
  x (higher-TF uptrend context, close > SMA200)
isolates the genuine forced-sweep-in-uptrend cell that single conditions cannot.

arc-2007 observation finding (the reason for THIS exact construction): the CLIMAX (big-range) sweep
is a FALLING KNIFE (forward drift -0.33 ATR, worse the bigger -> the strong SHORT leg, blocked by
long-only); the only non-knife long cell is the FAST-but-not-climactic drop that reclaims in an
uptrend (cap ~0.51-0.52, drift ~0 to +0.03). This signal encodes that best long cell.

Ex-ante (no-lookahead): prior swing low = rolling-min of MID low over [i-L, i-1] (shift1 excludes
bar i). sweep/reclaim/drop3/sma use only close/high/low at bar i and earlier; ATR = Wilder(14) on
MID, shift(1). Signal fires at bar i close; pool/engine enters next bar's open. Intended TF = H4.
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
class SweepReclaimReversalLongSignal:
    """Long a fast sweep-and-reclaim of a prior swing low in an uptrend.

    ``swing_lookback`` (L): bars for the prior swing low (rolling-min of MID low, shift1).
    ``drop3_atr``: require a fast move in, (close[i]-close[i-3])/atr <= -drop3_atr (speed/non-knife).
    ``sma_period``: higher-TF trend context, fire only when close_mid > SMA(sma_period). 0 disables.
    ``require_reclaim``: fire only when close_mid[i] > prior swing low (reclaimed). Default True.
    ``max_depth_atr``: if >0, also require the pierce be shallow ((swing_low-low)/atr < max_depth_atr)
        -- a deep pierce is the falling-knife climax. Default 0.0 (disabled; drop3 already excludes
        the slow grind, and the observation showed depth<0.5 did not add lift).
    """

    swing_lookback: int = 20
    drop3_atr: float = 1.0
    sma_period: int = 200
    require_reclaim: bool = True
    max_depth_atr: float = 0.0
    atr_period: int = 14
    signal_name: str = "sweep_reclaim_reversal_long_v0.1"
    primary_tf: str = "H4"
    auxiliary_tfs: tuple[str, ...] = ()
    causal_lineage: str = "clean"

    def required_aux_data(self) -> list[str]:
        return list(self.auxiliary_tfs)

    def _name(self) -> str:
        return (f"sweep_reclaim_long_L{self.swing_lookback}_d3{self.drop3_atr:.2f}"
                f"_sma{self.sma_period}_rec{int(self.require_reclaim)}"
                f"_dep{self.max_depth_atr:.2f}_v0.1")

    def evaluate(self, panels: Mapping[str, Panel]) -> SignalEvaluation:
        primary = panels[self.primary_tf]
        per_pair: dict[str, PerPairSignalState] = {}
        for pair in sorted(primary.pairs):
            df = primary.pair_dfs[pair]
            n = len(df)
            idx = df.index
            close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
            low_mid = (df["low_bid"].to_numpy(float) + df["low_ask"].to_numpy(float)) / 2.0
            atr = _atr_shift1_mid(df, self.atr_period)

            cm = pd.Series(close_mid, index=idx)
            lm = pd.Series(low_mid, index=idx)
            # prior swing low over [i-L, i-1]
            swing_low = lm.rolling(self.swing_lookback).min().shift(1).to_numpy()
            sma = cm.rolling(self.sma_period).mean().to_numpy() if self.sma_period > 0 else None

            drop3 = np.full(n, np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                drop3[3:] = (close_mid[3:] - close_mid[:-3]) / atr[3:]
                depth = (swing_low - low_mid) / atr  # how far below the swing low

            sweep = low_mid < swing_low
            reclaim = close_mid > swing_low if self.require_reclaim else np.ones(n, dtype=bool)
            uptrend = (close_mid > sma) if sma is not None else np.ones(n, dtype=bool)
            fast = drop3 <= -self.drop3_atr
            shallow = (depth < self.max_depth_atr) if self.max_depth_atr > 0 else np.ones(n, dtype=bool)

            fire = (
                np.isfinite(atr) & (atr > 0)
                & np.isfinite(swing_low)
                & sweep & reclaim & uptrend & fast & shallow
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


assert isinstance(SweepReclaimReversalLongSignal(), SignalModule)

__all__ = ("SweepReclaimReversalLongSignal",)
