"""Month-end reversion SignalModules (EXPERIMENT tools).

Built by arc 1011 (chat 1000s, the LONG) and arc 3017 (chat 3000s, the SHORT mirror).
EXPERIMENT tools per `discovery/TOOL_REGISTRY.md`: define an entry MASK + ATR (price geometry)
ONLY; never realize P&L. Scoring routes through the canonical apparatus (`build_arc_pool`,
`ArcFoldRunner` -> `MultiPairBacktester`). Conform to `core.arc.signal_protocol.SignalModule`.

Mechanism (arc 1011 observation + control): month-end mechanical rebalancing flows (the WMR 4pm London
fix on the last business day) are large and INELASTIC -- a big move INTO month-end over-extends and
REVERSES once the flow completes. The effect is DIRECTION-SYMMETRIC:
  - LONG side (arc 1011): a big DOWN move into month-end (a currency sold into the fix) -> buy it,
    expecting reversion UP. The arc-1011 control proved the timing is load-bearing: a big 2-day down
    move on a RANDOM day CONTINUES down (fwd2 -0.063, the dead generic-reversion finding of arcs
    3000/3001), while the SAME move into MONTH-END reverses (+0.186, +0.249 ATR month-end excess).
  - SHORT side (arc 3017): a big UP move into month-end (a currency bought into the fix) -> sell it,
    expecting reversion DOWN. Arc-3017 obs: capture 0.5508 (>0.50), month-end EXCESS drift +0.0996 ATR
    over the random-day big-UP control (generic big-up CONTINUES up -0.0405). Motivation: the long side
    is NEGATIVE in the strong-USD block (2014/15/16) because EURUSD-type down-moves continue; the short
    side fires on USDXXX over-extensions in those years -> candidate 2015 & 2018-positive 4th portfolio
    leg.

Ex-ante (no-lookahead): bar i is flagged the last trading day of its month when bar i+1 is in a new
month (calendar knowledge -- the last business day is date-determinable in advance; same convention as
arc 1005). The move INTO month-end uses close_mid[i] and close_mid[i-`into_bars`] (both known at bar i
close); ATR is Wilder(14) on MID, shift(1) (strictly prior bars, excludes bar i). The signal fires at
bar i close; the pool/engine enters at the next bar's open (the first trading day of the next month).
No future bar is read for the entry decision. Intended TF = D1.
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


def _month_end_into_move(df: pd.DataFrame, into_bars: int, atr_period: int):
    """Return (idx, is_last_month_end, into_move_in_atr, atr) -- shared ex-ante geometry."""
    n = len(df)
    idx = df.index
    close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
    atr = _atr_shift1_mid(df, atr_period)
    ym = pd.PeriodIndex(idx, freq="M")
    is_last = np.zeros(n, dtype=bool)
    if n >= 2:
        is_last[:-1] = ym[1:] != ym[:-1]
    into = np.full(n, np.nan)
    k = into_bars
    with np.errstate(invalid="ignore", divide="ignore"):
        into[k:] = (close_mid[k:] - close_mid[:-k]) / atr[k:]
    return idx, is_last, into, atr


@dataclass(frozen=True)
class MonthEndReversionLongSignal:
    """Long a big DOWN move into month-end, betting on the post-fix rebalancing reversion UP.

    ``threshold_atr``: minimum |down-move| into month-end in ATR units to fire
        (into_atr <= -threshold_atr, where into = (close[i] - close[i-into_bars])/atr).
    ``into_bars``: lookback (bars) for the move into month-end (default 2 on D1 = last 2 days).
    """

    threshold_atr: float = 1.0
    into_bars: int = 2
    atr_period: int = 14
    signal_name: str = "month_end_reversion_long_v0.1"
    primary_tf: str = "D1"
    auxiliary_tfs: tuple[str, ...] = ()
    causal_lineage: str = "clean"

    def required_aux_data(self) -> list[str]:
        return list(self.auxiliary_tfs)

    def _name(self) -> str:
        return f"month_end_rev_long_thr{self.threshold_atr:.2f}_in{self.into_bars}_v0.1"

    def evaluate(self, panels: Mapping[str, Panel]) -> SignalEvaluation:
        primary = panels[self.primary_tf]
        per_pair: dict[str, PerPairSignalState] = {}
        for pair in sorted(primary.pairs):
            df = primary.pair_dfs[pair]
            idx, is_last, into, atr = _month_end_into_move(df, self.into_bars, self.atr_period)
            fire = (
                is_last
                & np.isfinite(into)
                & np.isfinite(atr)
                & (atr > 0)
                & (into <= -self.threshold_atr)
            )
            mask = np.zeros(len(df), dtype=bool)
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


@dataclass(frozen=True)
class MonthEndReversionShortSignal:
    """SHORT a big UP move into month-end, betting on the post-fix rebalancing reversion DOWN.

    The direction-mirror of ``MonthEndReversionLongSignal``, built INDEPENDENTLY and concurrently by
    arc 1019 (chat 1000s) and arc 3017 (chat 3000s) — same logic, same pool (n=116, gross +0.1713).
    Same ex-ante month-end detection + ATR geometry; fires when the move INTO month-end is a big UP
    move (``into >= +threshold_atr``, where into = (close[i] - close[i-into_bars])/atr) and declares
    ``Direction.SHORT`` on both the per-pair state and the evaluation so the canonical Step-1 pool
    + architecture emit a short (entry next bar at open_bid, SL ABOVE entry, ``final_r`` short-signed).

    Mechanism (observation + control, mirror of arc 1011): month-end mechanical rebalancing reverts
    BOTH directions; arc 1011 captured only the long/down side and is 2015-negative (in a strong-USD
    trend a big down move into month-end IS the trend → continues, doesn't revert). The SHORT side
    fades a big UP move into month-end — a counter-trend bounce that mechanical reversion +
    trend-resumption pushes back down — positive precisely in strong-USD years (2015 +0.46 ATR, 2018
    +0.37 ATR gross; month-end excess +0.089 vs the random-day control; honest short capture 0.5508,
    the first corpus short >0.50). Conforms to ``core.arc.signal_protocol.SignalModule``.

    **Disposition (independent-reproduction resolution, arc 3017):** PORTFOLIO. The honest-engine
    result is exit-sensitive — with a reversion-horizon time exit it is mean-positive every exit and
    beats the fair same-side null by ~+0.8pp (arc 1019: partial-runner mean +0.683%, 7/10, robustly
    2018-positive); a long 120-bar hold washes it toward noise. Intended TF = D1, USD majors, SHORT
    reversion exit (≤5-bar time exit or partial-runner). Not all-folds-positive (best 7/10) → PORTFOLIO.
    """

    threshold_atr: float = 1.0
    into_bars: int = 2
    atr_period: int = 14
    signal_name: str = "month_end_reversion_short_v0.1"
    primary_tf: str = "D1"
    auxiliary_tfs: tuple[str, ...] = ()
    causal_lineage: str = "clean"

    def required_aux_data(self) -> list[str]:
        return list(self.auxiliary_tfs)

    def _name(self) -> str:
        return f"month_end_rev_short_thr{self.threshold_atr:.2f}_in{self.into_bars}_v0.1"

    def evaluate(self, panels: Mapping[str, Panel]) -> SignalEvaluation:
        primary = panels[self.primary_tf]
        per_pair: dict[str, PerPairSignalState] = {}
        for pair in sorted(primary.pairs):
            df = primary.pair_dfs[pair]
            idx, is_last, into, atr = _month_end_into_move(df, self.into_bars, self.atr_period)
            fire = (
                is_last
                & np.isfinite(into)
                & np.isfinite(atr)
                & (atr > 0)
                & (into >= self.threshold_atr)
            )
            mask = np.zeros(len(df), dtype=bool)
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


assert isinstance(MonthEndReversionLongSignal(), SignalModule)
assert isinstance(MonthEndReversionShortSignal(), SignalModule)

__all__ = ("MonthEndReversionLongSignal", "MonthEndReversionShortSignal")
