"""KH24SignalModule — adapts the existing KH-24 signal layer to the
:class:`core.arc.signal_protocol.SignalModule` Protocol.

Per chat decision §6.2 (option (a) — full generalisation): KH-24 is a
config-only instantiation of architecture A1. The signal-class-inherent
mechanics (C1-C9 evaluation, H1 CIR gate, kijun_d1 exit) travel together
as part of the SignalModule; SL multiplier / trail / exposure / risk
live in the A1Config layer.

The adapter does not duplicate signal logic — it wraps the existing
``evaluate_kh24_signal``, ``evaluate_h1_cir``, ``make_kijun_d1_exit_predicate``
functions and surfaces them through the SignalModule contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd

from core.arc.signal_protocol import (
    PerPairSignalState,
    SignalEvaluation,
    SignalModule,
    validate_panels,
)
from core.sim.panel import Panel
from core.strategies.kh24.exits.kijun_d1 import make_kijun_d1_exit_predicate
from core.strategies.kh24.filters.h1_cir import H1CIRParams, evaluate_h1_cir
from core.strategies.kh24.signal import KH24SignalParams, evaluate_kh24_signal


@dataclass(frozen=True)
class KH24SignalModule(SignalModule):
    """SignalModule implementation for the deployed KH-24 signal.

    Inherits the locked KH-24 mechanics:
      - C1-C9 entry conditions (C7 disabled) per
        ``core.strategies.kh24.signal``
      - H1 CIR T=0.28 gate per ``core.strategies.kh24.filters.h1_cir``
      - kijun_d1 exit predicate per
        ``core.strategies.kh24.exits.kijun_d1``

    Tuning these breaks the deployed-EA contract; arcs that want a
    different signal write their own SignalModule.
    """

    signal_name: str = "kh24_kb_exhaustion_bar"
    primary_tf: str = "H4"
    auxiliary_tfs: tuple[str, ...] = ("D1", "H1")
    causal_lineage: str = "clean"
    signal_params: KH24SignalParams = field(default_factory=KH24SignalParams)
    h1_cir_params: H1CIRParams = field(default_factory=H1CIRParams)

    def evaluate(self, panels: Mapping[str, Panel]) -> SignalEvaluation:
        validate_panels(self, panels)
        h4 = panels[self.primary_tf]
        d1 = panels["D1"]
        h1 = panels["H1"]
        per_pair: dict[str, PerPairSignalState] = {}
        for pair in sorted(h4.pairs):
            df_h4 = h4.pair_dfs[pair]
            df_d1 = d1.pair_dfs[pair]
            df_h1 = h1.pair_dfs[pair]
            sig = evaluate_kh24_signal(df_h4, df_d1, params=self.signal_params)
            cir = evaluate_h1_cir(df_h4, df_h1, params=self.h1_cir_params)
            exit_pred = make_kijun_d1_exit_predicate(
                pair, df_h4, df_d1, kijun_period=self.signal_params.d1_kijun_period
            )
            per_pair[pair] = PerPairSignalState(
                signal_mask=pd.Series(sig.signal_mask, index=df_h4.index, name="signal_mask"),
                atr=pd.Series(sig.atr_h4, index=df_h4.index, name="atr_h4"),
                additional_gates={"h1_cir": pd.Series(cir, index=df_h4.index, name="h1_cir")},
                exit_predicate=exit_pred,
                path_feature_anchor=df_h4.index,
            )
        return SignalEvaluation(
            primary_tf=self.primary_tf,
            per_pair=per_pair,
            signal_name=self.signal_name,
            causal_lineage=self.causal_lineage,
        )


__all__ = ("KH24SignalModule",)
