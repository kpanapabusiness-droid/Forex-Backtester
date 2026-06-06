"""Touch-count-filtered failed-breakdown reclaim long SignalModule (EXPERIMENT tool, BUILT).

Built by arc 2030 (chat 2000s). EXPERIMENT tool per `discovery/TOOL_REGISTRY.md`: entry MASK + ATR
geometry ONLY; never realizes P&L. Scoring routes through the canonical apparatus. Conforms to
`core.arc.signal_protocol.SignalModule`.

Mechanism (arc 2029 -> arc 2030). Arc 2029 established that fbr's load-bearing liquidity feature is
*significance-by-survival* — a rolling-K swing low that is STILL INTACT has survived K bars of testing,
which is what concentrates resting stops (calendar-anchored prior-day/week lows, mechanically refreshed,
do NOT — they are a coin-flip once fbr's survived-swing subset is removed). Arc 2030 sharpens that: a
swing low that was *repeatedly TESTED* (approached within a tight band, without breaking) before the
sweep holds even DENSER resting liquidity than one that merely survived passively. Conditioning the fbr
population on prior touch-count separates strongly on IS H4 majors (gross honest +1R capture):
0-1 touches 0.480 (sub-coin-flip, drift -0.54) | 2-3 touches 0.614 (+0.61) | >=4 0.567. A `min_touches>=2`
filter removes the weak ~43% 0-1-touch tail and lifts fbr capture 0.547->0.598, drift +0.086->+0.555,
per-pair balanced (all 4 majors >=0.548). NOTE (arc 2030): the lift is a MEAN lift — the strong-USD
risk-off binding folds (2016/2018/2019) keep negative forward drift, so this is a CLEANER PORTFOLIO
component, NOT a path to solo all-folds-positive.

`min_touches`: minimum prior-touch count of the swept level required to fire. A "touch" = a bar in the
prior `swing_lookback` window whose `low_bid` sits within `[level, level + touch_band_atr*ATR]`
(approached the level from above without breaking it). Default 2.
`touch_band_atr`: tightness of the touch band (default 0.25 ATR).
Everything else mirrors `FailedBreakdownReclaimLongSignal` exactly (same swing/pierce/reclaim/shadow/ATR,
all ex-ante shift1; entry next bar). Intended TF=H4, USD majors.
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
class FailedBreakdownReclaimTouchFilteredLongSignal:
    """fbr (failed-breakdown reclaim long) restricted to swept levels tested >= ``min_touches`` times."""

    swing_lookback: int = 40
    min_shadow_atr: float = 1.0
    min_touches: int = 2
    touch_band_atr: float = 0.25
    atr_period: int = 14
    signal_name: str = "fbr_touch_filtered_long_v0.1"
    primary_tf: str = "H4"
    auxiliary_tfs: tuple[str, ...] = ()
    causal_lineage: str = "clean"

    def required_aux_data(self) -> list[str]:
        return list(self.auxiliary_tfs)

    def _name(self) -> str:
        return (
            f"fbr_touch_long_K{self.swing_lookback}_sh{self.min_shadow_atr:.2f}"
            f"_t{self.min_touches}_v0.1"
        )

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

            prior_low = (
                pd.Series(low_bid, index=idx).shift(1).rolling(self.swing_lookback).min().to_numpy(float)
            )
            with np.errstate(invalid="ignore"):
                shadow = (np.minimum(open_mid, close_mid) - low_bid) / atr

            base = (
                (low_bid < prior_low)
                & (close_mid > prior_low)
                & (shadow >= self.min_shadow_atr)
                & np.isfinite(atr) & (atr > 0)
                & np.isfinite(prior_low)
            )

            mask = np.zeros(n, dtype=bool)
            for i in np.flatnonzero(base):
                level = prior_low[i]
                a = atr[i]
                lo = max(0, i - self.swing_lookback)
                win = low_bid[lo:i]  # strictly prior bars (ex-ante)
                touches = int(np.sum((win >= level) & (win <= level + self.touch_band_atr * a)))
                if touches >= self.min_touches:
                    mask[i] = True

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


assert isinstance(FailedBreakdownReclaimTouchFilteredLongSignal(), SignalModule)

__all__ = ("FailedBreakdownReclaimTouchFilteredLongSignal",)
