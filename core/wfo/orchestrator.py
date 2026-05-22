"""WFO orchestrator: per-fold backtest + top-K selection + holdout one-shot.

The orchestrator knows nothing about specific signals or architectures.
It accepts:

  - A ``WfoStructure`` (from ``core.wfo.folds``)
  - A ``fold_runner(fold, config) -> FoldStats`` callable that does the
    actual backtest for one (fold, config) pair
  - A list of candidate configs
  - Optional minimum-IS-days threshold (folds with shorter IS are skipped)

Returns a ``WfoSearchResult`` with per-fold stats per candidate, top-K
selection by worst-fold ratio, holdout one-shot stats for each top-K
candidate, and a §3 verdict per candidate.

L_PROTOCOL §2 Step 5 invariant: the holdout is NEVER touched during the
search loop. ``run_search`` operates only on ``structure.folds``;
``run_holdout`` is a separate call invoked only after search completes,
and it accepts only the top-K candidates.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from core.wfo.folds import Fold, WfoStructure
from core.wfo.gates import FoldStats, GateResult, classify_fold_stats

ConfigT = TypeVar("ConfigT")


# Caller-supplied: run one fold for one config → FoldStats. The callable
# may consult panel data, train classifiers, simulate trades, etc. — none
# of that is the orchestrator's concern.
FoldRunner = Callable[[Fold, ConfigT], FoldStats]


@dataclass(frozen=True)
class CandidateSearchResult(Generic[ConfigT]):
    config: ConfigT
    config_id: str  # human-readable identifier
    fold_stats: tuple[FoldStats, ...]
    gate: GateResult


@dataclass(frozen=True)
class CandidateHoldoutResult(Generic[ConfigT]):
    config: ConfigT
    config_id: str
    search_gate: GateResult  # search-window verdict
    holdout_stats: FoldStats
    holdout_gate: GateResult  # holdout-window verdict (one fold)
    deployable: bool  # both gates PASS_DEPLOYABLE


@dataclass(frozen=True)
class WfoSearchResult(Generic[ConfigT]):
    structure_name: str
    n_candidates_evaluated: int
    candidates: tuple[CandidateSearchResult[ConfigT], ...]
    top_k: tuple[CandidateSearchResult[ConfigT], ...]


def run_search(
    structure: WfoStructure,
    candidates: list[tuple[str, ConfigT]],
    fold_runner: FoldRunner[ConfigT],
    min_is_days: int = 365,
    top_k: int = 3,
) -> WfoSearchResult[ConfigT]:
    """Run the search loop over ``structure.folds`` for each candidate.

    ``candidates`` is a list of ``(config_id, config)`` tuples. The
    orchestrator calls ``fold_runner(fold, config)`` for each (fold,
    config) pair, gathers stats, runs §3 gate per candidate, and selects
    the top-K by worst-fold ratio.

    Folds with ``is_days < min_is_days`` are skipped — ``min_is_days``
    defaults to 365 (one year). Set to 0 to evaluate every fold.

    The holdout is NOT touched here.
    """
    eligible_folds = tuple(f for f in structure.folds if f.is_days >= min_is_days)

    results: list[CandidateSearchResult[ConfigT]] = []
    for config_id, config in candidates:
        stats = tuple(fold_runner(f, config) for f in eligible_folds)
        gate = classify_fold_stats(stats)
        results.append(
            CandidateSearchResult(
                config=config,
                config_id=config_id,
                fold_stats=stats,
                gate=gate,
            )
        )

    # Rank by worst-fold ratio (descending). Tie-break by mean ratio.
    ranked = sorted(
        results,
        key=lambda r: (r.gate.worst_fold_ratio, r.gate.mean_fold_ratio),
        reverse=True,
    )
    return WfoSearchResult(
        structure_name=structure.name,
        n_candidates_evaluated=len(results),
        candidates=tuple(results),
        top_k=tuple(ranked[:top_k]),
    )


def run_holdout(
    structure: WfoStructure,
    top_k: tuple[CandidateSearchResult[ConfigT], ...],
    fold_runner: FoldRunner[ConfigT],
) -> tuple[CandidateHoldoutResult[ConfigT], ...]:
    """Evaluate top-K candidates ONCE on the locked holdout window.

    Raises ``ValueError`` if ``structure.holdout`` is None (KH-24 anchor
    mode has no holdout).
    """
    if structure.holdout is None:
        raise ValueError(
            f"WFO structure {structure.name!r} has no holdout window; use search results directly"
        )
    out: list[CandidateHoldoutResult[ConfigT]] = []
    for candidate in top_k:
        stats = fold_runner(structure.holdout, candidate.config)
        # Single-fold gate — same logic on a one-element sequence
        holdout_gate = classify_fold_stats((stats,))
        out.append(
            CandidateHoldoutResult(
                config=candidate.config,
                config_id=candidate.config_id,
                search_gate=candidate.gate,
                holdout_stats=stats,
                holdout_gate=holdout_gate,
                deployable=(
                    candidate.gate.verdict.value == "pass_deployable"
                    and holdout_gate.verdict.value == "pass_deployable"
                ),
            )
        )
    return tuple(out)
