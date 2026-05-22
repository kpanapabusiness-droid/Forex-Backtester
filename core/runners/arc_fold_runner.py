"""Generic per-architecture fold runner.

Replaces KH-24-specific :class:`core.wfo.fold_runner.KH24FoldRunner` for
new arcs. The runner takes:

  - Architecture instance (A1..A6) — defines the runtime mechanics
  - SignalEvaluation — produced once by the arc-pool builder
  - Panels — full TF panels (the architecture slices per fold)
  - Architecture config — passed through to ``architecture.run(...)``

And returns ``FoldStats`` per (fold, config). Plugs into the existing
``core.wfo.orchestrator.run_search`` / ``run_holdout`` API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import pandas as pd

from core.arc.signal_protocol import SignalEvaluation
from core.architectures._protocol import Architecture, StrategyResult
from core.sim.panel import Panel
from core.wfo.folds import Fold
from core.wfo.gates import FoldStats


@dataclass
class ArcFoldRunner:
    """Architecture-agnostic fold runner.

    Construct with an architecture instance + signal evaluation +
    panels; call with (fold, arch_config) -> FoldStats. The instance
    can be reused across folds with the same panels (cheap, no
    re-evaluation of the signal).
    """

    architecture: Architecture
    signal_evaluation: SignalEvaluation
    panels: Mapping[str, Panel]
    run_context: Any = None  # arc-specific context (per-trade features, etc.)
    last_result: StrategyResult | None = None  # populated after each call

    def __call__(self, fold: Fold, arch_config: Any) -> FoldStats:
        config_id = getattr(arch_config, "config_id", repr(arch_config))
        result = self.architecture.run(
            signal_evaluation=self.signal_evaluation,
            panels=self.panels,
            fold=fold,
            arch_config=arch_config,
            config_id=config_id,
            run_context=self.run_context,
        )
        self.last_result = result
        return result.fold_stats


def make_arc_fold_runner(
    architecture: Architecture,
    signal_evaluation: SignalEvaluation,
    panels: Mapping[str, Panel],
    *,
    run_context: Any = None,
) -> ArcFoldRunner:
    """Convenience constructor — returns an ArcFoldRunner."""
    return ArcFoldRunner(
        architecture=architecture,
        signal_evaluation=signal_evaluation,
        panels=panels,
        run_context=run_context,
    )


__all__ = ("ArcFoldRunner", "make_arc_fold_runner")
