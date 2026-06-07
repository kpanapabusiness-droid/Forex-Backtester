"""Trend-CONTINUATION Donchian-breakout SignalModule (EXPERIMENT tool) — both directions.

Built by arc 1074 (chat 1000s). EXPERIMENT tool per `discovery/TOOL_REGISTRY.md`: defines an
entry MASK + ATR (price geometry) + a per-pair `Direction` ONLY; it NEVER realizes P&L. All
scoring routes through the canonical apparatus (`build_arc_pool` / `ArcFoldRunner` ->
`MultiPairBacktester`). Conforms to `core.arc.signal_protocol.SignalModule`.

Why this exists (arc 1074 thesis). Arc 1064 tested the textbook trend-continuation entry (a fresh
Donchian-N break inside an established dual-SMA trend, both directions) under the positive-skew lens,
but cheap-killed it on GROSS `fwd_drift_atr` WITHOUT spending the honest engine, arguing the
SL-first engine is "strictly worse" than gross drift. That bound is wrong for a take-the-loss
trailing-runner exit: the -1R stop CAPS the gross left-tail losers that drove arc-1064's negative
tail-removed gross mean. This tool re-expresses arc-1064's EXACT entry as a proper `SignalModule`
so the honest engine can be spent under the runner exits (the deferred test). The dual-SMA
trend filter + Donchian crossing-bar construction MATCH arc-1064 `build_obs` bar-for-bar (mid OHLC,
shift(1), crossing-bar-only) so the engine run is apples-to-apples with the gross-drift screen.

`DonchianBreakoutLongSignal` (arc 2000) is long-only with a single SMA filter; this adds the
dual-SMA trend-CONTINUATION condition AND the SHORT mirror (shorts enabled, PR #273).

Ex-ante (no-lookahead): Donchian high/low = rolling max/min of prior N MID highs/lows shift(1);
SMA_fast/SMA_slow on MID close shift(1); break detected on the CROSSING bar only; ATR(14) Wilder
on MID OHLC shift(1). Entry fills next-bar open in the engine.
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


def _mid(df: pd.DataFrame, field: str) -> np.ndarray:
    return (df[f"{field}_bid"].to_numpy(float) + df[f"{field}_ask"].to_numpy(float)) / 2.0


@dataclass(frozen=True)
class TrendContinuationBreakoutSignal:
    """Fresh Donchian-N break inside an established dual-SMA trend; long OR short.

    LONG  : (close_mid > SMA_slow AND SMA_fast > SMA_slow) AND a FRESH break above the
            prior-N-bar Donchian high (crossing bar only).
    SHORT : (close_mid < SMA_slow AND SMA_fast < SMA_slow) AND a FRESH break below the
            prior-N-bar Donchian low.
    """

    direction: str = "long"            # "long" | "short"
    lookback: int = 20                 # Donchian channel length (prior N bars)
    sma_fast: int = 50
    sma_slow: int = 200
    spacing_bars: int = 6              # refractory between accepted breakouts
    atr_period: int = 14
    signal_name: str = "trend_continuation_breakout_v0.1"
    primary_tf: str = "H4"
    auxiliary_tfs: tuple[str, ...] = ()
    causal_lineage: str = "clean"

    def required_aux_data(self) -> list[str]:
        return list(self.auxiliary_tfs)

    @property
    def _dir(self) -> Direction:
        return Direction.LONG if self.direction == "long" else Direction.SHORT

    def _name(self) -> str:
        return (f"trendcont_{self.direction}_d{self.lookback}_sma{self.sma_fast}-{self.sma_slow}"
                f"_sp{self.spacing_bars}_v0.1")

    def evaluate(self, panels: Mapping[str, Panel]) -> SignalEvaluation:
        primary = panels[self.primary_tf]
        per_pair: dict[str, PerPairSignalState] = {}
        long_side = self.direction == "long"
        for pair in sorted(primary.pairs):
            df = primary.pair_dfs[pair]
            n = len(df)
            idx = df.index
            hi, lo, mc = _mid(df, "high"), _mid(df, "low"), _mid(df, "close")
            atr = _atr_shift1_mid(df, self.atr_period)

            sma_f = pd.Series(mc, index=idx).rolling(self.sma_fast).mean().shift(1).to_numpy()
            sma_s = pd.Series(mc, index=idx).rolling(self.sma_slow).mean().shift(1).to_numpy()
            prior_hi = pd.Series(hi, index=idx).rolling(self.lookback).max().shift(1).to_numpy()
            prior_lo = pd.Series(lo, index=idx).rolling(self.lookback).min().shift(1).to_numpy()

            if long_side:
                trend = (mc > sma_s) & (sma_f > sma_s)
                brk = mc > prior_hi
            else:
                trend = (mc < sma_s) & (sma_f < sma_s)
                brk = mc < prior_lo
            brk = np.where(np.isfinite(prior_hi if long_side else prior_lo), brk, False)
            prev_brk = np.r_[False, brk[:-1]]
            fire = trend & brk & (~prev_brk) & np.isfinite(sma_s) & np.isfinite(sma_f) & np.isfinite(atr)

            # spacing refractory (greedy left-to-right)
            kept = np.zeros(n, dtype=bool)
            last = -10_000
            for t in np.flatnonzero(fire):
                if (t - last) >= self.spacing_bars:
                    kept[t] = True
                    last = t

            per_pair[pair] = PerPairSignalState(
                signal_mask=pd.Series(kept, index=idx, name="signal_mask"),
                atr=pd.Series(atr, index=idx, name="atr_14_shift1"),
                additional_gates={},
                exit_predicate=None,
                path_feature_anchor=None,
                direction=self._dir,
            )
        return SignalEvaluation(
            primary_tf=self.primary_tf,
            per_pair=per_pair,
            signal_name=self._name(),
            causal_lineage=self.causal_lineage,
            direction=self._dir,
        )


assert isinstance(TrendContinuationBreakoutSignal(), SignalModule)

__all__ = ("TrendContinuationBreakoutSignal",)
