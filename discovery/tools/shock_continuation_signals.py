"""Forward-confirmed EXTREME-SHOCK CONTINUATION SignalModule (EXPERIMENT tool).

Built by arc 3019 (chat 3000s). EXPERIMENT tool per `discovery/TOOL_REGISTRY.md`: defines an entry
MASK + ATR (price geometry) + a per-pair Direction ONLY; never realizes P&L. Scoring routes through the
canonical apparatus (`build_arc_pool`, `ArcFoldRunner` -> `MultiPairBacktester`, SL-honest take-the-loss).
Conforms to `core.arc.signal_protocol.SignalModule`.

Mechanism (arc 3019 observation + magnitude/confirmation controls): an EXTREME single-bar shock
(|close-to-close body| >= `shock_atr` * ATR, a ~tail event = FORCED flow — margin liquidations, stop
cascades, central-bank action) creates PERSISTENT one-way pressure for days as leveraged players unwind,
UNLIKE an ordinary 1-1.5 ATR vol spike (which reverts — arc 3012, and the 1.0-ATR control here captures
a coin-flip 0.498). The continuation is harvested only when FORWARD-CONFIRMED: the bar AFTER the shock
breaks the shock bar's extreme in the shock direction (down-shock: a later low; up-shock: a later high) —
entering on the RESUMPTION, the fix to the i+1-bounce death (arcs 1016/2009/3012 entered at the local
extreme that bounces) the same way arc 1013's reclaim fixed gap-fill's capturability wall.

arc-3019 controls (gross +1R-before-SL capture, IS 2010-2020, 7 USD majors H4):
  - MAGNITUDE MONOTONE: capture 0.498 (1.0 ATR, coin-flip) -> 0.528 (2.0) -> 0.549 (2.5) -> 0.589 (3.0).
    The edge is in the EXTREME TAIL, not generic momentum (so NOT closed-ground shallow breakout).
  - FORWARD-CONFIRM load-bearing: confirmed 0.589 > unconfirmed 0.543 > base 0.488 (3.0 ATR).
  - Per-pair robust: all 7 pairs capture >0.50 at 3.0 ATR; 2016 (Brexit/Trump/oil) positive in all 7.

VERDICT (arc 3019): KILL — EPOCH-DEPENDENT. IS (2010-2020) was the strongest continuation result in the
corpus (honest engine tp_3r 9/10, mean +0.034, beats fair null +0.048; +2015 AND +2016, the exact 4-way
book blockers; survives mega-event-window removal). BUT the one-shot frozen-exit OOS (2021-2026) FAILS:
tp_3r combined 2/6 folds, mean -0.0022, and LOSES to the null (-0.005pp). The shock->continuation
directionality did NOT persist post-2020 (2022 Fed-hiking/LDI shocks continued; 2023-2025 post-shock
MEAN-REVERTED). Also un-scalable: extreme shocks cluster across pairs on macro-event days, so the
FundedNext 5%-daily-DD cap binds catastrophically at deployment risk (breaches 0->7-84/yr from risk
0.005->0.5, signs flip). Tool kept (valid reusable signal); the mechanism is real in-sample but
epoch-specific + cap-clustered. Intended TF=H4, USD majors.

Direction is per-PAIR-fixed (the protocol carries one Direction per pair, not per bar): instantiate
TWICE — `direction="short"` fires DOWN-shocks (continuation down), `direction="long"` fires UP-shocks
(continuation up) — and combine the two components' per-fold ROIs (disjoint event timing).

Ex-ante (no-lookahead): the shock body move[i-1] & ATR(i-1, shift1) are known at bar i-1 close; the
confirmation (bar i's low/high breaking bar i-1's extreme) is known at bar i close; the signal fires at
bar i close, the pool/engine enters at bar i+1 open. No future bar is read. Intended TF = H4, USD majors.
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
class ShockContinuationSignal:
    """Forward-confirmed extreme-shock continuation, ONE direction per instance.

    ``shock_atr``: minimum |close-to-close body move| / ATR for the shock bar (arc 3019: 2.5-3.0;
        3.0 is the capture peak, 2.5 keeps 2018 ~neutral with ~2x the trades).
    ``direction``: "short" fires confirmed DOWN-shocks (sell the resumption); "long" fires confirmed
        UP-shocks (buy the resumption).
    ``confirm``: require the bar after the shock to break the shock bar's extreme in the shock
        direction (True = the load-bearing forward-confirmation; False = enter i+1 raw, the control).
    """

    shock_atr: float = 3.0
    direction: str = "short"
    confirm: bool = True
    atr_period: int = 14
    signal_name: str = "shock_continuation_v0.1"
    primary_tf: str = "H4"
    auxiliary_tfs: tuple[str, ...] = ()
    causal_lineage: str = "clean"

    def required_aux_data(self) -> list[str]:
        return list(self.auxiliary_tfs)

    def _dir(self) -> Direction:
        return Direction.SHORT if str(self.direction).lower() == "short" else Direction.LONG

    def _name(self) -> str:
        c = "conf" if self.confirm else "raw"
        return f"shock_cont_{self.direction}_a{self.shock_atr:.1f}_{c}_v0.1"

    def evaluate(self, panels: Mapping[str, Panel]) -> SignalEvaluation:
        primary = panels[self.primary_tf]
        is_short = str(self.direction).lower() == "short"
        per_pair: dict[str, PerPairSignalState] = {}
        for pair in sorted(primary.pairs):
            df = primary.pair_dfs[pair]
            n = len(df)
            idx = df.index
            close_mid = (df["close_bid"].to_numpy(float) + df["close_ask"].to_numpy(float)) / 2.0
            low_bid = df["low_bid"].to_numpy(float)
            high_ask = df["high_ask"].to_numpy(float)
            atr = _atr_shift1_mid(df, self.atr_period)

            move = np.full(n, np.nan)
            move[1:] = close_mid[1:] - close_mid[:-1]
            with np.errstate(invalid="ignore"):
                mr = move / atr
            finite = np.isfinite(atr) & (atr > 0)
            if is_short:
                shock = finite & (mr <= -self.shock_atr)        # big down body
            else:
                shock = finite & (mr >= self.shock_atr)          # big up body

            mask = np.zeros(n, dtype=bool)
            # fire at bar t where bar t-1 was a shock (and, if confirm, bar t broke its extreme)
            prev_shock = np.zeros(n, dtype=bool)
            prev_shock[1:] = shock[:-1]
            if self.confirm:
                if is_short:
                    broke = np.zeros(n, dtype=bool)
                    broke[1:] = low_bid[1:] < low_bid[:-1]
                else:
                    broke = np.zeros(n, dtype=bool)
                    broke[1:] = high_ask[1:] > high_ask[:-1]
                fire = prev_shock & broke & finite
            else:
                # control: enter the bar AFTER the shock itself (no confirmation gate)
                fire = shock & finite
            mask[fire] = True

            per_pair[pair] = PerPairSignalState(
                signal_mask=pd.Series(mask, index=idx, name="signal_mask"),
                atr=pd.Series(atr, index=idx, name="atr_14_shift1"),
                additional_gates={},
                exit_predicate=None,
                path_feature_anchor=None,
                direction=self._dir(),
            )
        return SignalEvaluation(
            primary_tf=self.primary_tf,
            per_pair=per_pair,
            signal_name=self._name(),
            causal_lineage=self.causal_lineage,
            direction=self._dir(),
        )


assert isinstance(ShockContinuationSignal(), SignalModule)

__all__ = ("ShockContinuationSignal",)
