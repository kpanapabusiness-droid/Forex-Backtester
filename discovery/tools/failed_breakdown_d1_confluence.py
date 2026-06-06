"""D1-level-confluent failed-breakdown reclaim long (EXPERIMENT tool, arc 1040).

`FailedBreakdownReclaimLongSignal` (fbr, arc 1013) restricted to H4 reclaims whose swept extreme sits AT
a Daily (D1) swing-low support level — a CROSS-TIMEFRAME structural-confluence filter (the "multi-timeframe
structure as a setup" sub-lane the corpus flagged untested). HYPOTHESIS (arc 1040): fbr's sole negative IS
fold is 2018 (failed breakdown -> real breakdown in a strong-USD regime, arc 2014); a reclaim at a D1 level
— defended on a higher timeframe — may hold even in 2018, rescuing the fold.

Mask + ATR GEOMETRY ONLY; never realizes P&L (scoring routes through the canonical apparatus). Conforms to
`core.arc.signal_protocol.SignalModule`.

No-lookahead: the H4 fire is fbr's own ex-ante fire (shift1 swing low, bar-i OHLC). The D1 level uses only
D1 bars strictly PRIOR to the H4 fire's calendar day (`D1 low_bid.shift(1).rolling(d1_swing_n).min()`,
read from the most recent D1 bar dated before the fire day) — no future D1 bar is read. D1 ATR = Wilder(14)
MID shift1. Confluence = the H4 sweep extreme `low_bid[i]` is within `band` * D1-ATR of that D1 level.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from core.arc.signal_protocol import PerPairSignalState, SignalEvaluation, SignalModule
from core.features._helpers import wilder_atr, mid_high, mid_low, mid_close
from core.sim.panel import Panel
from discovery.tools.failed_breakdown_signals import FailedBreakdownReclaimLongSignal


@dataclass(frozen=True)
class FailedBreakdownReclaimD1ConfluenceLongSignal:
    """fbr (K=swing_lookback / shadow=min_shadow_atr) AND the H4 sweep tags a prior D1 swing-low level.

    ``band``: confluence tolerance in D1-ATR units (|sweep_low - d1_level| <= band * d1_atr). 0.5 default.
    ``d1_swing_n``: D1 swing-low window length in days.
    """

    swing_lookback: int = 40
    min_shadow_atr: float = 1.25
    band: float = 0.5
    d1_swing_n: int = 20
    atr_period: int = 14
    signal_name: str = "fbr_d1_confluence_long_v0.1"
    primary_tf: str = "H4"
    auxiliary_tfs: tuple[str, ...] = ("D1",)
    causal_lineage: str = "clean"

    def required_aux_data(self) -> list[str]:
        return list(self.auxiliary_tfs)

    def _name(self) -> str:
        return f"fbr_d1conf_K{self.swing_lookback}_sh{self.min_shadow_atr:.2f}_b{self.band:.2f}_v0.1"

    def evaluate(self, panels: Mapping[str, Panel]) -> SignalEvaluation:
        h4 = panels["H4"]
        d1 = panels["D1"]
        base = FailedBreakdownReclaimLongSignal(
            swing_lookback=self.swing_lookback, min_shadow_atr=self.min_shadow_atr,
            atr_period=self.atr_period,
        ).evaluate({"H4": h4})

        per_pair: dict[str, PerPairSignalState] = {}
        for pair in sorted(h4.pairs):
            state = base.per_pair[pair]
            mask = state.signal_mask.to_numpy(bool).copy()
            idx = state.signal_mask.index
            low_bid_h4 = h4.pair_dfs[pair]["low_bid"]

            ddf = d1.pair_dfs[pair]
            d1_level = ddf["low_bid"].shift(1).rolling(self.d1_swing_n).min()
            d1_atr = wilder_atr(mid_high(ddf), mid_low(ddf), mid_close(ddf), self.atr_period).shift(1)
            d1_dates = d1_level.index

            for k in np.where(mask)[0]:
                t = idx[k]
                day = t.normalize()
                prior = d1_dates < day               # D1 bars strictly before the fire day (ex-ante)
                if not prior.any():
                    mask[k] = False
                    continue
                j = np.where(prior)[0][-1]
                lvl = d1_level.iloc[j]
                datr = d1_atr.iloc[j]
                if not (np.isfinite(lvl) and np.isfinite(datr) and datr > 0):
                    mask[k] = False
                    continue
                sweep_low = float(low_bid_h4.loc[t])
                if abs(sweep_low - lvl) / datr > self.band:
                    mask[k] = False                  # not at a D1 level -> drop the fire

            per_pair[pair] = PerPairSignalState(
                signal_mask=pd.Series(mask, index=idx, name="signal_mask"),
                atr=state.atr,
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


assert isinstance(FailedBreakdownReclaimD1ConfluenceLongSignal(), SignalModule)

__all__ = ("FailedBreakdownReclaimD1ConfluenceLongSignal",)
