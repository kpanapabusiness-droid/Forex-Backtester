"""Turn-of-quarter USD-reversion SignalModule (EXPERIMENT tool).

Built by arc 2058 (chat 2000s). EXPERIMENT tool per `discovery/TOOL_REGISTRY.md`: defines an entry MASK +
ATR (price geometry) + per-pair `Direction` ONLY; never realizes P&L. Scoring routes through the canonical
apparatus (`build_arc_pool`, `ArcFoldRunner` -> `MultiPairBacktester`). Conforms to
`core.arc.signal_protocol.SignalModule`.

Mechanism (arc 2058 observation): across the 7 USD majors, D1, a long-USD position WEAKENS into the
quarter-end turn (5d into-move usd_long -0.255 ATR) and STRENGTHENS in the 3 days AFTER (fwd usd_long
+0.296 ATR, median +0.251, frac+ 0.601), vs a control of other-month-ends (+/-0 fwd) and all-days
baseline (~0). The post-turn USD strength is carried by the European/commodity XXXUSD majors
(EUR/GBP/AUD/NZD-USD all fwd>0: +0.44/+0.56/+0.35/+0.18) — i.e. those pairs FALL after the quarter-end ->
a uniform SHORT-the-major-after-quarter-end. Candidate quarter-end rebalancing / reserve-management flow,
DISTINCT from `me` (1011/1019, fires every month-end conditional on a >=1 ATR move; this is quarter-end-
SPECIFIC + unconditional) and from the dead IMM futures roll (2057, instrument-neutral in spot).

Ex-ante (no-lookahead): bar i is the last trading day of a quarter when bar i+1 is in a new month AND
bar i's month is in {3,6,9,12} (calendar knowledge, arc-1005/me convention). ATR = Wilder(14) on MID,
shift(1). Signal fires at bar i close declaring `Direction.SHORT`; the pool/engine enters at the next
bar's open (first trading day of the new quarter), SL above entry, `final_r` short-signed. Intended
TF = D1, XXXUSD majors (pass only EUR/GBP/AUD/NZD-USD to fire the clean short leg).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from core.arc.signal_protocol import PerPairSignalState, SignalEvaluation, SignalModule
from core.sim.account import Direction
from core.sim.panel import Panel
from discovery.tools.trend_entry_signals import _atr_shift1_mid


@dataclass(frozen=True)
class QuarterEndUsdReversionShortSignal:
    """SHORT a XXXUSD major on the last trading day of Mar/Jun/Sep/Dec (engine enters first day of new
    quarter), betting on the post-turn USD-strength reversion (the major falls). Unconditional on
    magnitude (the arc-2058 effect is unconditional); the exit menu governs the reversion horizon.

    ``quarters``: the calendar months whose last trading day fires (default the 4 quarter-ends).
    Set ``year_end_only=True`` to restrict to December (the strongest leg, but ~1 fire/pair/yr = thin).
    """

    quarters: tuple[int, ...] = (3, 6, 9, 12)
    year_end_only: bool = False
    atr_period: int = 14
    signal_name: str = "quarter_end_usd_rev_short_v0.1"
    primary_tf: str = "D1"
    auxiliary_tfs: tuple[str, ...] = ()
    causal_lineage: str = "clean"

    def required_aux_data(self) -> list[str]:
        return list(self.auxiliary_tfs)

    def _name(self) -> str:
        tag = "ye" if self.year_end_only else "qe"
        return f"quarter_end_usd_rev_short_{tag}_v0.1"

    def evaluate(self, panels: Mapping[str, Panel]) -> SignalEvaluation:
        primary = panels[self.primary_tf]
        months = (12,) if self.year_end_only else self.quarters
        per_pair: dict[str, PerPairSignalState] = {}
        for pair in sorted(primary.pairs):
            df = primary.pair_dfs[pair]
            idx = df.index
            n = len(df)
            atr = _atr_shift1_mid(df, self.atr_period)
            ym = pd.PeriodIndex(idx, freq="M")
            is_last = np.zeros(n, dtype=bool)
            if n >= 2:
                is_last[:-1] = ym[1:] != ym[:-1]
            month_ok = np.isin(idx.month.to_numpy(), np.array(months))
            fire = is_last & month_ok & np.isfinite(atr) & (atr > 0)
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


assert isinstance(QuarterEndUsdReversionShortSignal(), SignalModule)

__all__ = ("QuarterEndUsdReversionShortSignal",)
