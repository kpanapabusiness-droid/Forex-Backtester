"""Arc 8 — pullback-resume HH/HL long (PR-HHHL).

Wraps :func:`core.signals.pullback_resume_hhhl.evaluate_pullback_resume_hhhl_signal`
in the canonical :class:`core.arc.signal_protocol.SignalModule` Protocol so
arc orchestrators (Step 5 dispatch, A1/A4 architectures) can consume it
through the uniform v3 interface.

Causal lineage: clean — H4-only single-TF signal, no HTF lookup.
State classification: A (single-TF) under both UTC and 5ers_eet boundary
conventions per ``docs/audits/signal_module_eet_audit_2026_05.md``.
"""

from core.strategies.pullback_resume_hhhl_long.signal_module import (
    PullbackResumeHHHLLongSignal,
)

__all__ = ("PullbackResumeHHHLLongSignal",)
