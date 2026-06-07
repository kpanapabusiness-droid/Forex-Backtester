"""Compression-COIL breakout SignalModule (EXPERIMENT tool) — the mechanistic OPPOSITE of
arc-2082's `VolExpansionBreakoutSignal`, and the purest positive-skew entry geometry the apparatus
can express.

The positive-skew premise (LESSONS 2026-06-06; the operator's ONE open in-charter thread): a genuine
continuation/trend edge loses often (low capture, ~-1R median) but is mean-positive from a FAT RIGHT
TAIL of large winners. A breakout out of a TIGHT COIL is the strongest structural candidate for that
shape: the coil's small recent range means a small ATR, so the engine's -1R stop is a small ABSOLUTE
move sitting just back inside the coil; if the post-break expansion runs, the trailing runner harvests
a LARGE R-multiple (move / small-stop-distance) -> exactly the bounded-loss / unbounded-win skew. This
was killed only on the BLIND lens: arc 1001 (long-only, capture + single-exit triage, pre-redirect) and
arc 1062 (both dirs, gross capture/drift, "no engine"). Per the arc-1064 -> 1074 precedent the operator
sanctioned, it must be re-run on the honest engine under the tail-preserving runner exits + the mandated
mean / median-per-fold / tail-removed lens. That is arc 1078.

Distinct from `VolExpansionBreakoutSignal` (arc 2082): that fires only when the CURRENT bar is a vol
IGNITION bar (TR/ATR >= vol_mult) -> the breakout happens on an already-LARGE bar -> the 2*ATR stop is
WIDE -> small R-multiples on the tail. THIS fires only when the prior-N range was COMPRESSED
((Donchian-high - Donchian-low)/ATR <= coil_thresh) -> small stop -> large R-multiple tail. Opposite
gate, opposite skew geometry.

Mask + ATR geometry ONLY; NEVER realizes P&L — all scoring routes through the canonical apparatus
(`build_arc_pool` / `ArcFoldRunner` -> `MultiPairBacktester`). Conforms to
`core.arc.signal_protocol.SignalModule`.

Ex-ante (no-lookahead): ATR(14) Wilder on MID, shift(1); Donchian high/low = rolling max/min of the
prior N bars, shift(1); the coil width uses those shift1 Donchian extremes (so it is measured strictly
on bars BEFORE the entry-decision bar); crossing-bar only + spacing refractory; entry fills next-bar
open in the engine.

Built by arc 1078 (chat 1000s).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from core.arc.signal_protocol import PerPairSignalState, SignalEvaluation, SignalModule
from core.sim.account import Direction
from discovery.tools.trend_entry_signals import _apply_spacing, _atr_shift1_mid


@dataclass(frozen=True)
class CompressionCoilBreakoutSignal:
    """Long/short breakout that fires only out of a TIGHT (compressed) coil.

    Fires at bar t when (a) the prior-N coil width ``(Donchian_high - Donchian_low) / shift1_ATR <=
    coil_thresh`` (compression) AND (b) close_mid[t] breaks the prior-N Donchian extreme (high for
    long, low for short) for the FIRST time (crossing bar), subject to a spacing refractory.
    ``direction`` = "long" or "short".
    """

    lookback: int = 20             # coil / Donchian window (bars)
    coil_thresh: float = 5.0       # (Donchian range)/ATR <= this => tight coil
    spacing_bars: int = 6
    atr_period: int = 14
    direction: str = "long"
    signal_name: str = "compression_coil_breakout_v0.1"
    primary_tf: str = "H4"
    auxiliary_tfs: tuple[str, ...] = ()
    causal_lineage: str = "clean"

    def required_aux_data(self) -> list[str]:
        return list(self.auxiliary_tfs)

    def _name(self) -> str:
        return f"coil{self.lookback}_t{self.coil_thresh}_{self.direction}_v0.1"

    def evaluate(self, panels: Mapping[str, Panel]) -> SignalEvaluation:  # type: ignore[name-defined]
        primary = panels[self.primary_tf]
        dirn = Direction.SHORT if self.direction == "short" else Direction.LONG
        per_pair: dict[str, PerPairSignalState] = {}
        for pair in sorted(primary.pairs):
            df = primary.pair_dfs[pair]
            n = len(df)
            idx = df.index
            close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
            high_mid = (df["high_bid"].to_numpy(float) + df["high_ask"].to_numpy(float)) / 2.0
            low_mid = (df["low_bid"].to_numpy(float) + df["low_ask"].to_numpy(float)) / 2.0
            atr = _atr_shift1_mid(df, self.atr_period)

            hi = pd.Series(high_mid, index=idx).rolling(self.lookback, min_periods=self.lookback).max().shift(1).to_numpy()
            lo = pd.Series(low_mid, index=idx).rolling(self.lookback, min_periods=self.lookback).min().shift(1).to_numpy()

            # coil width (prior-N range) over shift1 ATR -- all known strictly before bar t (ex-ante)
            with np.errstate(invalid="ignore", divide="ignore"):
                coil_ratio = (hi - lo) / atr
            tight = np.isfinite(coil_ratio) & (coil_ratio <= self.coil_thresh)

            if dirn is Direction.LONG:
                raw = np.isfinite(hi) & (close_mid > hi)
            else:
                raw = np.isfinite(lo) & (close_mid < lo)
            prev = np.zeros(n, dtype=bool); prev[1:] = raw[:-1]
            fire = raw & (~prev) & np.isfinite(atr) & (atr > 0) & tight

            kept = _apply_spacing(np.flatnonzero(fire), self.spacing_bars)
            mask = np.zeros(n, dtype=bool)
            if len(kept):
                mask[kept] = True
            per_pair[pair] = PerPairSignalState(
                signal_mask=pd.Series(mask, index=idx, name="signal_mask"),
                atr=pd.Series(atr, index=idx, name="atr_14_shift1"),
                additional_gates={}, exit_predicate=None, path_feature_anchor=None,
                direction=dirn,
            )
        return SignalEvaluation(
            primary_tf=self.primary_tf, per_pair=per_pair, signal_name=self._name(),
            causal_lineage=self.causal_lineage, direction=dirn,
        )


from core.sim.panel import Panel  # noqa: E402  (after dataclass to keep the annotation lazy)

assert isinstance(CompressionCoilBreakoutSignal(), SignalModule)
__all__ = ("CompressionCoilBreakoutSignal",)
