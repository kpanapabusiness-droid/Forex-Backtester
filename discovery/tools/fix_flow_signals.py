"""Month-end London-4pm-fix reversion long SignalModule (EXPERIMENT tool, BUILT).

Built by arc 3008 (chat 3000s). EXPERIMENT tool per `discovery/TOOL_REGISTRY.md`: defines an
entry MASK + ATR (price geometry) ONLY; never realizes P&L. Scoring routes through the canonical
apparatus (`build_arc_pool`, `ArcFoldRunner` -> `MultiPairBacktester`). Conforms to
`core.arc.signal_protocol.SignalModule`.

Mechanism (documented *because*, arc 3008 observation): at month-end, equity-index investors
rebalance FX hedges at the 16:00 London WM/Reuters benchmark fix; the aggregate, mechanical,
predictable flow pushes price INTO the fix and partially REVERSES afterward (Melvin & Prins 2015;
Evans 2018). The long-only tradeable side is an abnormal DOWN push into the fix: buy it, betting
on the post-fix reversion UP. This is the weekend-gap-fill archetype (discrete dislocation ->
reversion) on a decorrelated, calendar-timed intraday event.

Ex-ante (no-lookahead): primary TF is H1. The "run-up into the fix" bar is the bar whose
London-local hour == ``fix_hour_london`` (default 15 -> the 15:00-16:00 London bar that ENDS at the
16:00 fix; DST-robust via Europe/London tz). It fires at that bar's close when (a) the date is the
last weekday of its calendar month (the canonical rebalancing fix day) and (b) the bar's own return
``(close_mid - open_mid)/ATR <= -threshold_atr`` (a down-push). ATR is Wilder(14) on MID, shift(1)
(strictly prior bars). The engine enters at the NEXT bar's open (the post-fix hour). No future bar
is read.
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


def _month_end_dates(index_utc) -> set:
    """Set of last-weekday-of-month dates (London-local), observed in the index."""
    ldn = index_utc.tz_convert("Europe/London")
    days = pd.DatetimeIndex(sorted(set(ldn.normalize())))
    s = pd.Series(days, index=days)
    grp = s.groupby([days.year, days.month]).last()
    return set(pd.DatetimeIndex(grp.values).date)


@dataclass(frozen=True)
class MonthEndFixReversionLongSignal:
    """Long an abnormal DOWN push into the month-end 16:00 London fix, betting on post-fix reversion.

    ``threshold_atr``: minimum |down-push| (fix-run-up-bar return in ATR units) to fire.
    ``fix_hour_london``: London-local hour of the run-up bar that ends at the fix (default 15).
    """

    threshold_atr: float = 0.5
    fix_hour_london: int = 15
    atr_period: int = 14
    signal_name: str = "month_end_fix_reversion_long_v0.1"
    primary_tf: str = "H1"
    auxiliary_tfs: tuple[str, ...] = ()
    causal_lineage: str = "clean"

    def required_aux_data(self) -> list[str]:
        return list(self.auxiliary_tfs)

    def _name(self) -> str:
        return f"me_fix_reversion_long_thr{self.threshold_atr:.2f}_v0.1"

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

            ldn = idx.tz_convert("Europe/London")
            hour = np.asarray(ldn.hour)
            dts = np.asarray(pd.DatetimeIndex(ldn.normalize()).date)
            me = _month_end_dates(idx)
            is_me = np.array([d in me for d in dts])

            with np.errstate(invalid="ignore", divide="ignore"):
                push_atr = (close_mid - open_mid) / atr

            fire = (
                (hour == self.fix_hour_london)
                & is_me
                & np.isfinite(push_atr)
                & np.isfinite(atr)
                & (atr > 0)
                & (push_atr <= -self.threshold_atr)
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


assert isinstance(MonthEndFixReversionLongSignal(), SignalModule)

__all__ = ("MonthEndFixReversionLongSignal",)
