"""Trend-PULLBACK-RESUME continuation SignalModule (EXPERIMENT tool) — both directions.

Built by arc 1075 (chat 1000s). EXPERIMENT tool per `discovery/TOOL_REGISTRY.md`: entry MASK + ATR +
per-pair `Direction` ONLY; never realizes P&L. Scoring routes through the canonical apparatus.

Why this exists (arc 1075 thesis, motivated by arc 1074's diagnosis). Arc 1074 spent the honest engine
on the Donchian-breakout-in-trend continuation entry under runner exits and found it decisively
negative (mean per-trade R ~-0.5, median ~-0.9) because a breakout-at-the-channel-edge is ADVERSE-FIRST
— you buy the exact high, price retraces through the -1R stop on the majority, and take-the-loss fires
before any runner can develop. The natural mechanistic complement (DISCOVERY_DIRECTION item S1, the
take-the-loss-tax entry geometry; the fbr edge is its reversion instance): enter the continuation AFTER
the adverse pullback, on the RESUMPTION bar, so the path is FAVORABLE-FIRST and the -1R tax falls on the
noise trades, not the trend trades. This is the last continuation entry GEOMETRY not yet run on the
engine + positive-skew (mean + tail-removed) lens.

Entry (LONG): established uptrend (close_mid > SMA_slow AND SMA_fast > SMA_slow) THAT recently pulled
back (a bar in the last `pullback_lookback` whose low_mid dipped below SMA_fast — a dip into the fast
MA) AND a RESUMPTION now: close_mid > prior-bar high_mid (resumes up) AND close_mid > SMA_fast (back
above the fast MA). Fires on the resumption crossing bar; spacing refractory. SHORT = the mirror.
Ex-ante: SMAs on MID close shift1; pullback uses low_mid over a prior window (shift1); resumption uses
the prior bar's high (shift1); ATR(14) Wilder MID shift1. Entry fills next-bar open in the engine.
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
class TrendPullbackResumeSignal:
    """Continuation entry on a pullback-then-resumption inside an established dual-SMA trend; long OR short."""

    direction: str = "long"
    sma_fast: int = 20
    sma_slow: int = 100
    pullback_lookback: int = 10        # window in which a dip into the fast MA must have occurred
    spacing_bars: int = 6
    atr_period: int = 14
    signal_name: str = "trend_pullback_resume_v0.1"
    primary_tf: str = "H4"
    auxiliary_tfs: tuple[str, ...] = ()
    causal_lineage: str = "clean"

    def required_aux_data(self) -> list[str]:
        return list(self.auxiliary_tfs)

    @property
    def _dir(self) -> Direction:
        return Direction.LONG if self.direction == "long" else Direction.SHORT

    def _name(self) -> str:
        return (f"pullbackresume_{self.direction}_sma{self.sma_fast}-{self.sma_slow}"
                f"_pb{self.pullback_lookback}_sp{self.spacing_bars}_v0.1")

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
            prev_hi = np.r_[np.nan, hi[:-1]]
            prev_lo = np.r_[np.nan, lo[:-1]]

            if long_side:
                trend = (mc > sma_s) & (sma_f > sma_s)
                dipped = lo < sma_f                              # bar's low pierced the fast MA (a pullback)
                pulled = (pd.Series(dipped, index=idx)
                          .rolling(self.pullback_lookback).max().shift(1).fillna(0).to_numpy() > 0)
                resume = (mc > prev_hi) & (mc > sma_f)
            else:
                trend = (mc < sma_s) & (sma_f < sma_s)
                popped = hi > sma_f                              # bar's high pierced the fast MA (a pullback up)
                pulled = (pd.Series(popped, index=idx)
                          .rolling(self.pullback_lookback).max().shift(1).fillna(0).to_numpy() > 0)
                resume = (mc < prev_lo) & (mc < sma_f)

            fire = (trend & pulled & resume
                    & np.isfinite(sma_s) & np.isfinite(sma_f) & np.isfinite(atr)
                    & np.isfinite(prev_hi if long_side else prev_lo))
            prev_fire = np.r_[False, fire[:-1]]
            fire = fire & (~prev_fire)                           # crossing/first resumption bar only

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


assert isinstance(TrendPullbackResumeSignal(), SignalModule)

__all__ = ("TrendPullbackResumeSignal",)
