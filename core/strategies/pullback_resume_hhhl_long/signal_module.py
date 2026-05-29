"""SignalModule Protocol wrapper for the PR-HHHL long signal.

The raw signal logic lives in :mod:`core.signals.pullback_resume_hhhl`
(``evaluate_pullback_resume_hhhl_signal`` + ``PullbackResumeParams``).
This module wraps it in the canonical
:class:`core.arc.signal_protocol.SignalModule` Protocol so
:class:`core.arc.arc_orchestrator.ArcOrchestrator` can consume the signal
through the uniform v3 interface.

Conforms to :class:`core.arc.signal_protocol.SignalModule`.

PR-HHHL is H4-only with no HTF lookup — auxiliary_tfs is empty;
no additional_gates; no inherent exit_predicate. Causal lineage `clean`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from core.arc.signal_protocol import PerPairSignalState, SignalEvaluation, SignalModule
from core.signals.pullback_resume_hhhl import (
    PullbackResumeParams,
    evaluate_pullback_resume_hhhl_signal,
)
from core.sim.panel import Panel


@dataclass(frozen=True)
class PullbackResumeHHHLLongSignal:
    """L_PROTOCOL v3.0 SignalModule for Arc 8 — pullback-resume HH/HL long.

    Implements the :class:`core.arc.signal_protocol.SignalModule` Protocol.
    """

    signal_name: str = "pullback_resume_hhhl_long_v0.1"
    primary_tf: str = "H4"
    auxiliary_tfs: tuple[str, ...] = ()
    causal_lineage: str = "clean"
    params: PullbackResumeParams = PullbackResumeParams()

    def required_aux_data(self) -> list[str]:
        """Backward-compat with legacy SignalAdapter callers."""
        return list(self.auxiliary_tfs)

    def evaluate(self, panels: Mapping[str, Panel]) -> SignalEvaluation:
        """Apply the signal to every pair in panels[primary_tf].

        Calls :func:`evaluate_pullback_resume_hhhl_signal` per pair to
        produce the signal_mask + ATR series; PR-HHHL has no inherent
        exit predicate and no additional gates.
        """
        primary = panels[self.primary_tf]
        per_pair: dict[str, PerPairSignalState] = {}
        for pair in sorted(primary.pairs):
            df = primary.pair_dfs[pair]
            res = evaluate_pullback_resume_hhhl_signal(df, self.params)
            per_pair[pair] = PerPairSignalState(
                signal_mask=res.signal_mask.astype(bool),
                atr=res.atr_h4.astype(float),
                additional_gates={},
                exit_predicate=None,
                path_feature_anchor=None,
            )
        return SignalEvaluation(
            primary_tf=self.primary_tf,
            per_pair=per_pair,
            signal_name=self.signal_name,
            causal_lineage=self.causal_lineage,
        )


# Protocol conformance check (runtime).
assert isinstance(PullbackResumeHHHLLongSignal(), SignalModule)


__all__ = ("PullbackResumeHHHLLongSignal",)
