"""UNCONDITIONAL Donchian-breakout SignalModule (EXPERIMENT tool) — both directions, NO trend filter.

Built by arc 1085 (chat 1000s). EXPERIMENT tool per `discovery/TOOL_REGISTRY.md`: defines an entry
MASK + ATR (price geometry) + a per-pair `Direction` ONLY; it NEVER realizes P&L. All scoring routes
through the canonical apparatus (`ArcFoldRunner` -> `A1Architecture` -> `MultiPairBacktester`).
Conforms to `core.arc.signal_protocol.SignalModule`.

Why this exists (the gap arc 1085 closes). The positive-skew continuation frontier was closed under
the mandated mean+median+tail-removed lens across entry geometry (arcs 1074/1075/1078/2081/2082/2083)
and timeframe (1076 D1, 1077 W1) — but ALL of those used a TREND-FILTERED entry
(`TrendContinuationBreakoutSignal`: a Donchian break INSIDE an established dual-SMA trend). The pure,
UNCONDITIONAL Donchian breakout — the canonical CTA / time-series-momentum entry where the breakout
IS the trend signal (no prior-trend confirmation) — was run only by arc 2000, and there only
LONG-ONLY, on MAJORS, at H4, and CHEAP-KILLED at triage (no §5f exit selection, no formal
median-per-fold + tail-removed guard, no both-directions, no D1, no full 8-fold WFO). Arc 2000's
decisive finding was that the right tail is GENERIC (a random/periodic long has the same tail — the
breakout does not SELECT it). This tool re-expresses that exact unconditional entry as both-direction
so arc 1085 can close the cell airtight under the full mandated lens at D1.

This is the long-only `DonchianBreakoutLongSignal` (arc 2000) with the SHORT mirror added and the SMA
filter dropped; equivalently it is `TrendContinuationBreakoutSignal` with the dual-SMA trend gate
removed (so the construction is bar-for-bar identical to that corpus tool minus `trend`).

Ex-ante (no-lookahead): Donchian high/low = rolling max/min of prior N MID highs/lows shift(1); break
detected on the CROSSING bar only; ATR(14) Wilder on MID OHLC shift(1). Entry fills next-bar open.
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
class DonchianBreakoutSignal:
    """Fresh Donchian-N break, NO trend filter; long OR short (pure CTA trend-following entry).

    LONG  : a FRESH break above the prior-N-bar Donchian high (crossing bar only).
    SHORT : a FRESH break below the prior-N-bar Donchian low (crossing bar only).

    Identical to `TrendContinuationBreakoutSignal` minus the dual-SMA `trend` gate.
    """

    direction: str = "long"            # "long" | "short"
    lookback: int = 20                 # Donchian channel length (prior N bars)
    spacing_bars: int = 6              # refractory between accepted breakouts
    atr_period: int = 14
    signal_name: str = "donchian_breakout_uncond_v0.1"
    primary_tf: str = "H4"
    auxiliary_tfs: tuple[str, ...] = ()
    causal_lineage: str = "clean"

    def required_aux_data(self) -> list[str]:
        return list(self.auxiliary_tfs)

    @property
    def _dir(self) -> Direction:
        return Direction.LONG if self.direction == "long" else Direction.SHORT

    def _name(self) -> str:
        return f"donch_uncond_{self.direction}_d{self.lookback}_sp{self.spacing_bars}_v0.1"

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

            prior_hi = pd.Series(hi, index=idx).rolling(self.lookback).max().shift(1).to_numpy()
            prior_lo = pd.Series(lo, index=idx).rolling(self.lookback).min().shift(1).to_numpy()

            if long_side:
                brk = mc > prior_hi
            else:
                brk = mc < prior_lo
            brk = np.where(np.isfinite(prior_hi if long_side else prior_lo), brk, False)
            prev_brk = np.r_[False, brk[:-1]]
            fire = brk & (~prev_brk) & np.isfinite(atr)

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


assert isinstance(DonchianBreakoutSignal(), SignalModule)

__all__ = ("DonchianBreakoutSignal",)
