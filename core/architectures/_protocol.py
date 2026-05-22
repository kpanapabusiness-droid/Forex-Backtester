"""The Architecture Protocol and StrategyResult dataclass.

Every L_PROTOCOL §2 Step 5 architecture (A1..A6) implements this
interface. The fold runner (:mod:`core.runners.arc_fold_runner`) invokes
``architecture.run(...)`` once per (config, fold) pair and collects
:class:`StrategyResult` objects to roll up into a WFO search result.

The interface is deliberately wide enough for every architecture:

  - A1 / A2 / A6 read precomputed Step 4 outputs (classifier + threshold)
    plus the signal_pool + per-pair signal evaluation
  - A3 / A4 retrain a classifier per fold from path-so-far features
  - A5 composes ≥ 2 architectures' results into a portfolio

All six can be expressed as a single ``run(...)`` returning a
StrategyResult. Architecture-specific config goes inside the
``arch_config`` payload (architecture's choice of dataclass).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import pandas as pd

from core.arc.signal_protocol import SignalEvaluation
from core.sim.multipair_backtester import RunResult
from core.sim.panel import Panel
from core.wfo.folds import Fold
from core.wfo.gates import FoldStats


@dataclass(frozen=True)
class StrategyResult:
    """Per-architecture, per-fold output.

    ``run_result`` carries the raw RunResult from MultiPairBacktester.
    ``fold_stats`` is the §3 gate's view of it. ``metadata`` is
    architecture-specific (e.g. A6 records per-trade size_multiplier
    distribution; A2 records admit/reject counts; A3 records per-fold
    classifier AUC).

    ``equity_curve`` is pulled out for convenience — A5 needs to
    combine equity curves before computing portfolio DD.
    """

    architecture: str  # "A1" .. "A6"
    config_id: str
    fold: Fold
    run_result: RunResult
    fold_stats: FoldStats
    equity_curve: pd.Series
    closed_trades: tuple
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class Architecture(Protocol):
    """Architectures conform to a uniform ``run(...)`` interface.

    Implementations may take additional architecture-specific arguments
    via ``arch_config``. The protocol's positional arguments cover the
    common Step 5 inputs: signal evaluation, panels, fold, top-level
    architecture-search config.
    """

    architecture_name: str  # "A1" .. "A6"

    def run(
        self,
        *,
        signal_evaluation: SignalEvaluation,
        panels: Mapping[str, Panel],
        fold: Fold,
        arch_config: Any,
        config_id: str,
    ) -> StrategyResult:
        ...


__all__ = ("StrategyResult", "Architecture")
