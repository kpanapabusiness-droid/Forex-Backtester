"""Discovery-arc measurement glue — the measurement primitives the autonomous
discovery protocol needs that the canonical WFO core does not already expose,
plus a thin per-fold run helper.

This module is CANONICAL/LOCKED measurement apparatus (see
``discovery/TOOL_REGISTRY.md``): discovery arcs CALL it, they do not re-roll it
in scratch. It is *additive* — it composes the trusted core
(``core.wfo.folds``, ``core.wfo.gates``, the per-fold runners) and reimplements
none of it. The engine, cost application and per-fold ``FoldStats``
construction all stay exactly where they are; this file only supplies:

  1. ``build_oos_year_folds`` — per-year OOS folds (default 2021-present). The
     canonical ``build_v3_folds`` treats 2021-present as ONE locked holdout
     window; the discovery judge instead measures EACH holdout year as its own
     fold (all-folds-positive on OOS). Arc 0 hand-built this in scratch
     (``_arc0_work/wfo_validate.oos_year_folds``); it is promoted here so arcs
     stop re-rolling it.

  2. ``judge_all_folds_positive`` — the DISCOVERY judge: all-folds-positive on
     IS AND OOS (``DISCOVERY_PROTOCOL.md`` §5g). This is DELIBERATELY SEPARATE
     from ``core.wfo.gates.classify_fold_stats`` (the L_PROTOCOL dual-tier DD
     disposition) — discovery uses the simpler "every fold ROI > 0" sole judge.
     It only reads ``FoldStats.roi_pct`` / ``n_trades``; FundedNext costs are
     already netted upstream when the per-fold runner builds each ``FoldStats``.

  3. ``run_config_over_folds`` — a thin loop that calls the canonical per-fold
     runner (``ArcFoldRunner`` / ``OracleFoldRunner``) once per fold and
     collects the resulting ``FoldStats``. No scoring logic of its own.

Costs (FundedNext: 1.5x spread, 0.5 pip/fill slippage, $5/lot RT, no swaps),
the SL-first take-the-loss invariant, EET daily-DD bucketing and every ROI/DD
number come from the per-fold runner + ``build_fold_stats_from_run`` —
unchanged. Nothing here scores a trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Iterable, Sequence

from core.wfo.folds import (
    V3_TRAIN_END,
    V3_TRAIN_START,
    Fold,
    _last_complete_month_end,
)
from core.wfo.gates import FoldStats

# A per-fold runner: ArcFoldRunner / OracleFoldRunner — (fold, config) -> FoldStats.
FoldRunner = Callable[[Fold, Any], FoldStats]


def build_oos_year_folds(
    start_year: int = 2021,
    end: date | None = None,
    *,
    is_start: date = V3_TRAIN_START,
    is_end: date = V3_TRAIN_END,
) -> tuple[Fold, ...]:
    """Per-year OOS folds for the discovery holdout (default 2021-present).

    Each fold holds the IS window fixed at the v3 development window
    (``is_start``..``is_end`` = 2010-2020) and measures one calendar year of
    OOS. Architectures that train (A2/A4/A6) fit on the full dev window and are
    MEASURED on the (strictly later) OOS year — lookahead-safe; A1 has no
    training so the IS bounds are inert for it.

    ``end`` defaults to the last complete calendar month (reusing
    ``build_v3_folds``'s month-end convention via ``_last_complete_month_end``).
    The final year's OOS window is clamped to ``end`` so the current, incomplete
    year is measured only through available data; full prior years run
    Jan-1..Dec-31. ``fold_id`` is the OOS year itself (e.g. 2021), which never
    collides with ``build_v3_folds`` ids (1..12).

    NEVER tune to these folds — protocol §4: OOS may be MEASURED many times,
    never optimized against.
    """
    if is_end < is_start:
        raise ValueError(f"is_end {is_end} must be ≥ is_start {is_start}")
    end = end or _last_complete_month_end()
    end_year = end.year
    if start_year > end_year:
        raise ValueError(
            f"start_year {start_year} is after the last OOS year {end_year} (end={end})"
        )
    folds: list[Fold] = []
    for year in range(start_year, end_year + 1):
        oos_start = date(year, 1, 1)
        oos_end = date(year, 12, 31) if year < end_year else end
        folds.append(
            Fold(
                fold_id=year,
                is_start=is_start,
                is_end=is_end,
                oos_start=oos_start,
                oos_end=oos_end,
            )
        )
    return tuple(folds)


@dataclass(frozen=True)
class DiscoveryVerdict:
    """Result of the discovery all-folds-positive judge over one fold set.

    ``all_folds_positive`` is the protocol's SOLE judge for the set (applied
    independently to the IS set and the OOS set). ROI is decimal percent
    (0.0192 = 1.92%), matching ``FoldStats``.
    """

    all_folds_positive: bool
    worst_fold_roi: float
    n_negative_folds: int  # folds with roi_pct < 0
    n_folds: int
    min_trades_per_fold: int


def judge_all_folds_positive(fold_stats: Sequence[FoldStats]) -> DiscoveryVerdict:
    """Apply the discovery sole judge: is EVERY fold ROI strictly positive?

    Discovery's deployment judge is all-folds-positive on IS AND OOS
    (``DISCOVERY_PROTOCOL.md`` §5g) — distinct from the L_PROTOCOL dual-tier
    gate (``core.wfo.gates.classify_fold_stats``), which is NOT used here. Costs
    are already netted in each ``FoldStats`` upstream; this only reduces over
    ``roi_pct``. A fold of exactly 0.0 is NOT positive (fails the judge) but is
    not counted as negative — matching the Arc-0 trial convention.

    Raises ``ValueError`` on an empty sequence (a thin/empty fold set is a
    caller error, not a vacuous pass).
    """
    stats = tuple(fold_stats)
    if not stats:
        raise ValueError("judge_all_folds_positive requires ≥1 fold; got an empty sequence")
    rois = [fs.roi_pct for fs in stats]
    return DiscoveryVerdict(
        all_folds_positive=all(r > 0.0 for r in rois),
        worst_fold_roi=min(rois),
        n_negative_folds=sum(1 for r in rois if r < 0.0),
        n_folds=len(stats),
        min_trades_per_fold=min(fs.n_trades for fs in stats),
    )


def run_config_over_folds(
    fold_runner: FoldRunner,
    folds: Iterable[Fold],
    config: Any,
) -> tuple[FoldStats, ...]:
    """Run one architecture config across ``folds`` via the canonical per-fold
    runner, returning the per-fold ``FoldStats`` (the input to
    ``judge_all_folds_positive``).

    ``fold_runner`` is the trusted ``ArcFoldRunner`` (or ``OracleFoldRunner``) —
    this helper only loops and collects; it scores nothing itself. Each call
    routes through the architecture → ``MultiPairBacktester`` →
    ``build_fold_stats_from_run`` (FundedNext costs netted).
    """
    return tuple(fold_runner(fold, config) for fold in folds)
