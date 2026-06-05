"""Weekend-gap-fill long SignalModule (EXPERIMENT tool).

Built by arc 2001 (chat 2000s). EXPERIMENT tool per `discovery/TOOL_REGISTRY.md`: defines an entry
MASK + ATR (price geometry) ONLY; never realizes P&L. Scoring routes through the canonical apparatus
(`build_arc_pool`, `ArcFoldRunner`→`MultiPairBacktester`). Conforms to
`core.arc.signal_protocol.SignalModule`.

Mechanism (arc 2001 observation): at the weekly open, FX prices gap from the prior Friday close. Big
gaps tend to FILL (revert toward the prior close) — monotone & symmetric in gap size. The long-only
tradeable side is a significant DOWN gap (open below prior close): buy it, expecting reversion UP.

Ex-ante (no-lookahead): the weekly-open bar i is flagged by a > ``gap_hours`` index gap. The gap size
uses open_mid[i] and close_mid[i-1] (both known at bar i close); ATR is Wilder(14) on MID, shift(1)
(strictly prior bars, excludes bar i). The signal fires at bar i close; the pool/engine enters at the
next bar's open. No future bar is read.
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


@dataclass(frozen=True)
class WeekendGapFillLongSignal:
    """Long a significant weekly-open DOWN gap, betting on reversion toward the prior close.

    ``threshold_atr``: minimum |down-gap| in ATR units to fire (gap_atr <= -threshold_atr).
    ``gap_hours``: index time-gap (h) that flags a weekly/holiday open (normal H4 step = 4h).
    """

    threshold_atr: float = 1.0
    gap_hours: float = 20.0
    atr_period: int = 14
    signal_name: str = "weekend_gap_fill_long_v0.1"
    primary_tf: str = "H4"
    auxiliary_tfs: tuple[str, ...] = ()
    causal_lineage: str = "clean"

    def required_aux_data(self) -> list[str]:
        return list(self.auxiliary_tfs)

    def _name(self) -> str:
        return f"weekend_gapfill_long_thr{self.threshold_atr:.2f}_v0.1"

    def evaluate(self, panels: Mapping[str, Panel]) -> SignalEvaluation:
        primary = panels[self.primary_tf]
        per_pair: dict[str, PerPairSignalState] = {}
        for pair in sorted(primary.pairs):
            df = primary.pair_dfs[pair]
            n = len(df)
            idx = df.index
            open_mid = (df["open_bid"].to_numpy(float) + df["open_ask"].to_numpy(float)) / 2.0
            close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
            atr = _atr_shift1_mid(df, self.atr_period)

            prev_close = np.empty(n)
            prev_close[0] = np.nan
            prev_close[1:] = close_mid[:-1]
            dt_hours = idx.to_series().diff().dt.total_seconds().to_numpy() / 3600.0
            gap_open = dt_hours > self.gap_hours  # bar i is a weekly/holiday open

            with np.errstate(invalid="ignore", divide="ignore"):
                gap_atr = (open_mid - prev_close) / atr

            fire = (
                gap_open
                & np.isfinite(gap_atr)
                & np.isfinite(atr)
                & (atr > 0)
                & (gap_atr <= -self.threshold_atr)
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


assert isinstance(WeekendGapFillLongSignal(), SignalModule)


@dataclass(frozen=True)
class WeekendUpGapShortSignal:
    """SHORT a significant weekly-open UP gap, betting on reversion (fill) DOWN toward the prior close.

    The direction-mirror of ``WeekendGapFillLongSignal`` (arc 2013, chat 2000s). Same ex-ante
    weekly-open detection and ATR; fires when ``gap_atr >= +threshold_atr`` (open ABOVE prior close)
    and declares ``Direction.SHORT`` on both the per-pair state and the evaluation so the canonical
    Step-1 pool builder + architecture emit a short (entry next-bar, SL ABOVE entry, ``final_r``
    short-signed). EXPERIMENT tool: mask + ATR (price geometry) + side declaration ONLY; never
    realizes P&L — scoring routes through `build_arc_pool` / `ArcFoldRunner` → `MultiPairBacktester`.

    NOTE (arc 2013): honest i+1 short capture clears 0.50 only on JPY CROSSES (majors continue up),
    and the up-gap leg is the WEAKER honest leg vs the down-gap fill long (the JPY-basket upward drift
    helps the long, fights the short); thin (~12 ev/yr at thr 1.0) and non-monotone past 1.0.
    """

    threshold_atr: float = 1.0
    gap_hours: float = 20.0
    atr_period: int = 14
    signal_name: str = "weekend_up_gap_short_v0.1"
    primary_tf: str = "H4"
    auxiliary_tfs: tuple[str, ...] = ()
    causal_lineage: str = "clean"

    def required_aux_data(self) -> list[str]:
        return list(self.auxiliary_tfs)

    def _name(self) -> str:
        return f"weekend_upgap_short_thr{self.threshold_atr:.2f}_v0.1"

    def evaluate(self, panels: Mapping[str, Panel]) -> SignalEvaluation:
        primary = panels[self.primary_tf]
        per_pair: dict[str, PerPairSignalState] = {}
        for pair in sorted(primary.pairs):
            df = primary.pair_dfs[pair]
            n = len(df)
            idx = df.index
            open_mid = (df["open_bid"].to_numpy(float) + df["open_ask"].to_numpy(float)) / 2.0
            close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
            atr = _atr_shift1_mid(df, self.atr_period)

            prev_close = np.empty(n)
            prev_close[0] = np.nan
            prev_close[1:] = close_mid[:-1]
            dt_hours = idx.to_series().diff().dt.total_seconds().to_numpy() / 3600.0
            gap_open = dt_hours > self.gap_hours

            with np.errstate(invalid="ignore", divide="ignore"):
                gap_atr = (open_mid - prev_close) / atr

            fire = (
                gap_open
                & np.isfinite(gap_atr)
                & np.isfinite(atr)
                & (atr > 0)
                & (gap_atr >= self.threshold_atr)
            )
            mask = np.zeros(n, dtype=bool)
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


assert isinstance(WeekendUpGapShortSignal(), SignalModule)

__all__ = ("WeekendGapFillLongSignal", "WeekendUpGapShortSignal")
