"""Trend-entry long SignalModules (EXPERIMENT tools) — Donchian breakout + periodic base.

Built by arc 2000 (chat 2000s). These are EXPERIMENT tools per
`discovery/TOOL_REGISTRY.md`: they only define an entry MASK + ATR (price geometry);
they NEVER realize P&L. All scoring routes through the canonical apparatus
(`build_arc_pool` for path/MFE characterization, `ArcFoldRunner` → `MultiPairBacktester`
for gate verdicts). Conform to `core.arc.signal_protocol.SignalModule`.

Why these exist (arc 2000 thesis): the prior arcs screened on a WIN-RATE lens
(+1R-before-SL capture), which is structurally blind to a low-win-rate / fat-tailed
(convex) payoff. A Donchian breakout is the canonical time-series-momentum / trend
entry whose edge lives in the right tail. `DonchianBreakoutLongSignal` puts us long on
a fresh N-bar-high break so a tail-preserving full-size trailing exit
(`sl_plus_trailing_atr`) can be tested for convexity. `PeriodicLongSignal` is a
time-based unconditional base for the MFE-distribution comparison (does the breakout
fatten the harvestable tail vs a typical long?).

Ex-ante (no-lookahead) construction:
  - Donchian channel high = rolling max of prior N bar highs, shift(1).
  - Breakout fires only on the *crossing* bar (raw & ~raw.shift(1)), then spacing.
  - Optional SMA regime filter uses shift(1) SMA (strictly prior closes).
  - ATR(14) is Wilder on MID OHLC, shift(1) (matches the v3 convention in
    core.signals.pullback_resume_hhhl). Entry fills next-bar open in the pool/engine.
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


def _wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's ATR. Returns NaN for the first ``period`` bars.

    Verbatim convention from core.signals.pullback_resume_hhhl._wilder_atr so the
    R-frame matches the rest of the apparatus.
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    prev_close = np.empty(n)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    atr = np.full(n, np.nan)
    seed = np.nanmean(tr[1:period + 1])
    atr[period] = seed
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _atr_shift1_mid(df: pd.DataFrame, period: int) -> np.ndarray:
    """Wilder ATR(period) on MID OHLC, shifted 1 bar (ex-ante)."""
    high_mid = (df["high_bid"].to_numpy(float) + df["high_ask"].to_numpy(float)) / 2.0
    low_mid = (df["low_bid"].to_numpy(float) + df["low_ask"].to_numpy(float)) / 2.0
    close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    atr_raw = _wilder_atr(high_mid, low_mid, close_mid, period)
    n = len(df)
    out = np.empty(n)
    out[0] = np.nan
    out[1:] = atr_raw[:-1]
    return out


def _apply_spacing(fire_idx: np.ndarray, spacing_bars: int) -> np.ndarray:
    """Keep only fires >= spacing_bars apart (per pair). Greedy left-to-right."""
    last = -10_000
    kept = []
    for t in fire_idx:
        if (t - last) >= spacing_bars:
            kept.append(int(t))
            last = int(t)
    return np.asarray(kept, dtype=int)


@dataclass(frozen=True)
class DonchianBreakoutLongSignal:
    """Long on a fresh N-bar-high breakout (canonical trend / time-series-momentum entry).

    Fires at bar t when close_bid[t] breaks above the prior-N-bar Donchian high
    (max high_bid over t-N..t-1) for the FIRST time (crossing bar), subject to
    optional SMA-regime confirmation and a spacing refractory. Long-only.
    """

    lookback: int = 40                 # Donchian channel length (prior N highs)
    spacing_bars: int = 6              # refractory between accepted breakouts
    atr_period: int = 14
    sma_filter: int | None = None      # if set, also require close_mid > SMA(sma_filter) shift1
    signal_name: str = "donchian_breakout_long_v0.1"
    primary_tf: str = "H4"
    auxiliary_tfs: tuple[str, ...] = ()
    causal_lineage: str = "clean"

    def required_aux_data(self) -> list[str]:
        return list(self.auxiliary_tfs)

    def _name(self) -> str:
        sma = f"_sma{self.sma_filter}" if self.sma_filter else ""
        return f"donchian{self.lookback}{sma}_sp{self.spacing_bars}_long_v0.1"

    def evaluate(self, panels: Mapping[str, Panel]) -> SignalEvaluation:
        primary = panels[self.primary_tf]
        per_pair: dict[str, PerPairSignalState] = {}
        for pair in sorted(primary.pairs):
            df = primary.pair_dfs[pair]
            n = len(df)
            idx = df.index
            close_bid = df["close_bid"].to_numpy(float)
            high_bid = df["high_bid"].to_numpy(float)
            atr = _atr_shift1_mid(df, self.atr_period)

            # Donchian high of the prior N bars (ex-ante via shift 1).
            high_ser = pd.Series(high_bid, index=idx)
            donchian_high = high_ser.rolling(self.lookback, min_periods=self.lookback).max().shift(1).to_numpy()

            raw = np.zeros(n, dtype=bool)
            valid = np.isfinite(donchian_high)
            raw[valid] = close_bid[valid] > donchian_high[valid]
            # Crossing bar only: True now, not True on the prior bar.
            prev = np.zeros(n, dtype=bool)
            prev[1:] = raw[:-1]
            fire = raw & (~prev)

            if self.sma_filter:
                close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
                sma = pd.Series(close_mid, index=idx).rolling(
                    self.sma_filter, min_periods=self.sma_filter
                ).mean().shift(1).to_numpy()
                fire = fire & np.isfinite(sma) & (close_mid > sma)

            fire_idx = np.flatnonzero(fire)
            kept = _apply_spacing(fire_idx, self.spacing_bars)
            mask = np.zeros(n, dtype=bool)
            if len(kept):
                mask[kept] = True

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


@dataclass(frozen=True)
class PeriodicLongSignal:
    """Time-based unconditional base: fire every ``period`` bars per pair (after warmup).

    A deterministic, condition-free long entry used ONLY as an MFE-distribution
    reference (does a real trend entry fatten the harvestable right tail vs a typical
    long?). Not a deployable signal — a characterization yardstick.
    """

    period: int = 30
    warmup: int = 120
    atr_period: int = 14
    signal_name: str = "periodic_long_base_v0.1"
    primary_tf: str = "H4"
    auxiliary_tfs: tuple[str, ...] = ()
    causal_lineage: str = "clean"

    def required_aux_data(self) -> list[str]:
        return list(self.auxiliary_tfs)

    def _name(self) -> str:
        return f"periodic{self.period}_base_v0.1"

    def evaluate(self, panels: Mapping[str, Panel]) -> SignalEvaluation:
        primary = panels[self.primary_tf]
        per_pair: dict[str, PerPairSignalState] = {}
        for pair in sorted(primary.pairs):
            df = primary.pair_dfs[pair]
            n = len(df)
            idx = df.index
            atr = _atr_shift1_mid(df, self.atr_period)
            mask = np.zeros(n, dtype=bool)
            fire_idx = np.arange(self.warmup, n, self.period)
            mask[fire_idx] = True
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


# Protocol conformance (runtime).
assert isinstance(DonchianBreakoutLongSignal(), SignalModule)
assert isinstance(PeriodicLongSignal(), SignalModule)

__all__ = ("DonchianBreakoutLongSignal", "PeriodicLongSignal")
