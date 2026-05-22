"""SignalModule Protocol — the contract every arc signal must satisfy.

L_PROTOCOL v3.0 + Amendment 2 require Step 5's six architectures to be
signal-agnostic. The runtime achieves that by routing every signal
through a uniform :class:`SignalModule` Protocol.

A SignalModule owns:

  - Signal evaluation (the bare entry rule)
  - Signal-class-inherent filters (e.g. KH-24's H1 CIR)
  - Signal-class-inherent exit predicates (e.g. KH-24's kijun_d1)
  - The list of timeframes the signal needs (e.g. KH-24: H4, D1, H1)

It does NOT own:

  - SL multiplier (arc-config)
  - Trail / time-exit policy (arc-config, searched at Step 5)
  - Exposure caps (arc-config, searched at Step 5)
  - External rule-based filters from Step 4 (architecture-level overlay)
  - Risk size (arc-config)

The split keeps the signal's identity stable across arcs while letting
Step 5's architecture search vary the system-level wrapper.

Concrete signal modules live next to their signal logic; e.g.
``core/strategies/kh24/signal_module.py``. They register themselves by
construction — no global registry is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from core.sim.exit_hooks import ExitPredicate
from core.sim.panel import Panel


@dataclass(frozen=True)
class PerPairSignalState:
    """Precomputed per-pair signal output for one timeframe iteration axis.

    The arc's primary TF panel is the iteration axis. Strategy callables
    look up ``signal_mask.loc[t]`` and ``atr.loc[t]`` at every bar; the
    auxiliary panels (D1, H1) are consumed during precompute only.

    ``additional_gates`` is a dict of per-bar boolean Series the strategy
    must AND with the signal mask before emitting an order. KH-24's H1 CIR
    lives here.

    ``exit_predicate`` is the signal-class-inherent exit predicate; the
    architecture passes it to ``MultiPairBacktester(exit_predicates=...)``.
    None means the signal has no inherent exit hook (most arcs).

    ``path_feature_anchor`` records the per-pair index alignment between
    the primary TF and per-trade path emission — Step 1's path builder
    uses it to slice forward windows correctly. For most signal modules
    this is the primary TF's index unchanged.
    """

    signal_mask: pd.Series  # bool, indexed by primary-TF timestamp
    atr: pd.Series  # float, primary-TF
    additional_gates: Mapping[str, pd.Series] = field(default_factory=dict)
    exit_predicate: ExitPredicate | None = None
    path_feature_anchor: pd.Index | None = None


@dataclass(frozen=True)
class SignalEvaluation:
    """Output of :meth:`SignalModule.evaluate` — what arc-pool-builder consumes."""

    primary_tf: str
    per_pair: Mapping[str, PerPairSignalState]
    signal_name: str
    causal_lineage: str  # "clean" | "suspect" | "unverified"


@runtime_checkable
class SignalModule(Protocol):
    """The contract for any arc signal.

    Implementations expose four class attributes that arcs read at
    configure time:

      - ``signal_name``: unique identifier (e.g. "kh24_kb_exhaustion_bar")
      - ``primary_tf``: the iteration axis (e.g. "H4")
      - ``auxiliary_tfs``: tuple of TFs needed for precompute (e.g. ("D1", "H1"))
      - ``causal_lineage``: "clean" | "suspect" | "unverified" — feeds Step 6

    And one method:

      - ``evaluate(panels) -> SignalEvaluation``
    """

    signal_name: str
    primary_tf: str
    auxiliary_tfs: tuple[str, ...]
    causal_lineage: str

    def evaluate(self, panels: Mapping[str, Panel]) -> SignalEvaluation:
        """Run signal evaluation across every pair in ``panels[primary_tf]``.

        ``panels`` must contain ``primary_tf`` plus every entry in
        ``auxiliary_tfs``. All TF panels must share the same pair set.

        Implementations are responsible for ex-ante (no-lookahead)
        evaluation — every value in the returned signal_mask / atr /
        additional_gates must be derivable from data strictly prior to
        the bar's open. The runtime spot-checks this at Step 1
        integrity.
        """
        ...


def validate_panels(
    module: SignalModule, panels: Mapping[str, Panel]
) -> None:
    """Raise ValueError if ``panels`` is missing TFs the module needs or
    has TF panels with mismatched pair sets.

    Called by the arc-pool builder before invoking ``module.evaluate``.
    """
    needed = (module.primary_tf,) + tuple(module.auxiliary_tfs)
    missing = [tf for tf in needed if tf not in panels]
    if missing:
        raise ValueError(
            f"signal module {module.signal_name!r} requires panels "
            f"{needed}; missing {missing}"
        )
    primary_pairs = set(panels[module.primary_tf].pairs)
    for tf in module.auxiliary_tfs:
        if set(panels[tf].pairs) != primary_pairs:
            raise ValueError(
                f"panel pair sets differ: {module.primary_tf}={sorted(primary_pairs)}, "
                f"{tf}={sorted(panels[tf].pairs)}"
            )


__all__ = (
    "PerPairSignalState",
    "SignalEvaluation",
    "SignalModule",
    "validate_panels",
)
