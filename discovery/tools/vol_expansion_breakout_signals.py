"""Volatility-expansion breakout SignalModule (EXPERIMENT tool) — the dispatch's explicit
positive-skew example (LESSONS 2026-06-06): a breakout that occurs DURING a volatility-regime
EXPANSION, where forced-flow trend-INITIATION (and thus a positive-skew right tail) is most plausible.

Distinct from `DonchianBreakoutLongSignal` (arc 2000), which is an unconditional N-bar-high break: this
ADDS a vol-ignition gate (`TR/ATR >= vol_mult`) so the breakout fires only on expansion bars. Direction-
aware (long = up-break, short = down-break). Mask + ATR geometry ONLY; NEVER realizes P&L — all scoring
routes through the canonical apparatus (`build_arc_pool` / `ArcFoldRunner` -> `MultiPairBacktester`).
Conforms to `core.arc.signal_protocol.SignalModule`.

Ex-ante (no-lookahead): ATR(14) Wilder on MID, shift(1); Donchian high/low = rolling max/min of prior N
bars, shift(1); the vol-ignition uses the CURRENT bar's true range (known at bar close) / shift1 ATR;
crossing-bar only + spacing refractory; entry fills next-bar open in the engine.

Built by arc 2082 (chat 2000s).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from core.arc.signal_protocol import PerPairSignalState, SignalEvaluation, SignalModule
from core.sim.account import Direction
from discovery.tools.trend_entry_signals import _atr_shift1_mid, _apply_spacing


@dataclass(frozen=True)
class VolExpansionBreakoutSignal:
    """Long/short breakout that fires only on a volatility-EXPANSION (ignition) bar.

    Fires at bar t when (a) the bar's true range exceeds ``vol_mult`` x shift1 ATR (vol ignition) AND
    (b) close_mid[t] breaks the prior-N Donchian extreme (high for long, low for short) for the FIRST
    time (crossing bar), subject to a spacing refractory. ``direction`` = "long" or "short".
    """

    lookback: int = 40
    vol_mult: float = 1.5          # TR/ATR ignition threshold
    spacing_bars: int = 6
    atr_period: int = 14
    direction: str = "long"
    signal_name: str = "vol_expansion_breakout_v0.1"
    primary_tf: str = "H4"
    auxiliary_tfs: tuple[str, ...] = ()
    causal_lineage: str = "clean"

    def required_aux_data(self) -> list[str]:
        return list(self.auxiliary_tfs)

    def _name(self) -> str:
        return f"volexp{self.lookback}_v{self.vol_mult}_{self.direction}_v0.1"

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

            # current-bar true range (known at bar close) over shift1 ATR = vol ignition (ex-ante)
            prev_close = np.empty(n); prev_close[0] = np.nan; prev_close[1:] = close_mid[:-1]
            tr = np.maximum.reduce([high_mid - low_mid, np.abs(high_mid - prev_close),
                                    np.abs(low_mid - prev_close)])
            with np.errstate(invalid="ignore", divide="ignore"):
                ignition = (tr / atr) >= self.vol_mult

            hi = pd.Series(high_mid, index=idx).rolling(self.lookback, min_periods=self.lookback).max().shift(1).to_numpy()
            lo = pd.Series(low_mid, index=idx).rolling(self.lookback, min_periods=self.lookback).min().shift(1).to_numpy()
            if dirn is Direction.LONG:
                raw = np.isfinite(hi) & (close_mid > hi)
            else:
                raw = np.isfinite(lo) & (close_mid < lo)
            prev = np.zeros(n, dtype=bool); prev[1:] = raw[:-1]
            fire = raw & (~prev) & np.isfinite(atr) & (atr > 0) & ignition

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

assert isinstance(VolExpansionBreakoutSignal(), SignalModule)
__all__ = ("VolExpansionBreakoutSignal",)
