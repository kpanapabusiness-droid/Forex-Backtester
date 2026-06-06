"""Nested walk-forward EXIT/SL hyperparameter selection (§5f) — EXPERIMENT tool.

Pure selection ARITHMETIC over ALREADY-SCORED per-config per-fold ``FoldStats``
(each produced by the canonical ``ArcFoldRunner`` / ``run_config_over_folds``).
This module NEVER scores a trade, never realizes P&L, never touches the gate
engine — it only chooses, per fold, which already-scored config's number to
report, using a no-lookahead rule.

WHY THIS EXISTS (the §5f anti-exit-fishing discipline)
------------------------------------------------------
``DISCOVERY_PROTOCOL.md`` §5f mandates that for any entry whose base is NOT a
coin-flip, the exit/SL is a NESTED WFO hyperparameter:

  "selected on the IS portion of each 2010–2020 walk-forward fold, scored on that
   same fold's OOS. The chosen exit/SL is then FROZEN onto the 2021+ holdout —
   never re-selected per holdout year. NEVER pick the single best exit across the
   full sample and report its number — that is exit-fishing, an Arc-10-class gate
   inflation."

The v3 IS folds (``build_v3_folds``) are EXPANDING-window with a 1-year OOS
slice each (fold k → OOS year = 2010+k-1). So "the IS portion of fold k" is
exactly the set of folds whose OOS year is STRICTLY EARLIER than fold k's. This
tool encodes that: for fold k, pick the config that was best over the strictly
earlier folds (pure in-sample to k), then report fold k scored with THAT config.
The number reported for year k therefore uses an exit chosen only from data
before year k — a genuine walk-forward, no full-sample best-pick.

The FROZEN holdout choice is the config best over ALL IS folds; the caller scores
it once on each OOS year (it is NOT re-selected per holdout year, §4).

Built by arc 2040 (nested-WFO exit selection on the load-bearing fbr component).
EXPERIMENT tool: a bug fails loudly; it reimplements no measurement apparatus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from core.wfo.gates import FoldStats

# A selection metric ranks a config from its FoldStats over the PRIOR (in-sample)
# folds. Higher = better. Returns a sortable key (tuple ok).
SelectionMetric = Callable[[Sequence[FoldStats]], object]


def metric_mean_roi(prior: Sequence[FoldStats]) -> float:
    """Mean ROI over the prior folds (the conventional WFO objective)."""
    return sum(fs.roi_pct for fs in prior) / len(prior)


def metric_afp_then_mean(prior: Sequence[FoldStats]) -> tuple[int, float]:
    """(# positive folds, mean ROI) — rewards all-folds-positive-ness first.

    Aligned with the discovery sole judge (all-folds-positive), so the in-sample
    selection optimizes the SAME objective the gate scores, tie-broken by mean.
    """
    n_pos = sum(1 for fs in prior if fs.roi_pct > 0.0)
    return (n_pos, metric_mean_roi(prior))


def metric_worst_then_mean(prior: Sequence[FoldStats]) -> tuple[float, float]:
    """(worst-fold ROI, mean ROI) — rewards the least-bad fold first."""
    return (min(fs.roi_pct for fs in prior), metric_mean_roi(prior))


@dataclass(frozen=True)
class NestedFoldChoice:
    """One walk-forward fold's honest result under nested selection."""

    fold_id: int
    selected_label: str          # config chosen using ONLY strictly-earlier folds
    roi_pct: float               # fold scored with the selected config
    n_trades: int
    n_prior_folds: int           # how many in-sample folds the choice rested on
    is_warmup: bool              # True if < min_prior_folds priors (excluded from verdict)


@dataclass(frozen=True)
class NestedSelectionResult:
    """Outcome of a nested walk-forward exit/SL selection over one fold set."""

    per_fold: tuple[NestedFoldChoice, ...]
    frozen_label: str            # best config over ALL folds (the holdout freeze)
    # Verdict computed over the EVALUABLE (non-warmup) folds only:
    all_folds_positive: bool
    worst_fold_roi: float
    n_negative_folds: int
    n_evaluable_folds: int
    selection_name: str

    @property
    def evaluable(self) -> tuple[NestedFoldChoice, ...]:
        return tuple(c for c in self.per_fold if not c.is_warmup)


def _order_folds(labels: Sequence[str], scored: Mapping[str, Sequence[FoldStats]]) -> list[int]:
    """Return the common fold order (by fold_id) shared by every config.

    Every config must have been scored over the SAME folds in the same id set;
    we order by fold_id (ascending = chronological for build_v3_folds, whose
    fold_id k → OOS year 2010+k-1, and build_oos_year_folds, whose id IS the
    year).
    """
    ref = [fs.fold_id for fs in scored[labels[0]]]
    ref_sorted = sorted(ref)
    for lab in labels:
        ids = sorted(fs.fold_id for fs in scored[lab])
        if ids != ref_sorted:
            raise ValueError(
                f"config {lab!r} fold ids {ids} != reference {ref_sorted} — "
                "every config must be scored over the identical fold set"
            )
    return ref_sorted


def nested_walk_forward_select(
    scored: Mapping[str, Sequence[FoldStats]],
    *,
    selection: SelectionMetric = metric_mean_roi,
    selection_name: str = "mean_roi",
    min_prior_folds: int = 2,
) -> NestedSelectionResult:
    """Run §5f nested walk-forward exit/SL selection over already-scored configs.

    ``scored`` maps each config label -> its per-fold ``FoldStats`` (one per
    fold, same fold set for every config; produced upstream by the canonical
    runner). For each fold k (chronological by fold_id), select the config that
    maximizes ``selection`` over the STRICTLY EARLIER folds, then report fold k
    scored with that config. Folds with fewer than ``min_prior_folds`` priors are
    flagged ``is_warmup`` and excluded from the all-folds-positive verdict (no
    honest in-sample choice exists yet). ``frozen_label`` is the config best over
    ALL folds by the same metric — the choice to FREEZE onto the holdout.

    Returns a ``NestedSelectionResult``. This is pure arithmetic over canonical
    numbers; it scores nothing.
    """
    labels = sorted(scored)
    if not labels:
        raise ValueError("scored must contain ≥1 config")
    order = _order_folds(labels, scored)
    # index each config's stats by fold_id for O(1) lookup
    by_id: dict[str, dict[int, FoldStats]] = {
        lab: {fs.fold_id: fs for fs in scored[lab]} for lab in labels
    }

    choices: list[NestedFoldChoice] = []
    for pos, fid in enumerate(order):
        prior_ids = order[:pos]
        is_warmup = len(prior_ids) < min_prior_folds
        if is_warmup:
            # No honest in-sample choice yet: fall back to the global frozen pick
            # for a reported value, but mark warmup so it never enters the verdict.
            sel_label = _freeze_best(labels, by_id, order, selection)
        else:
            sel_label = max(
                labels,
                key=lambda lab: selection([by_id[lab][i] for i in prior_ids]),
            )
        fs_k = by_id[sel_label][fid]
        choices.append(
            NestedFoldChoice(
                fold_id=fid,
                selected_label=sel_label,
                roi_pct=fs_k.roi_pct,
                n_trades=fs_k.n_trades,
                n_prior_folds=len(prior_ids),
                is_warmup=is_warmup,
            )
        )

    frozen = _freeze_best(labels, by_id, order, selection)
    evaluable = [c for c in choices if not c.is_warmup]
    if not evaluable:
        raise ValueError(
            f"no evaluable folds (all {len(choices)} are warmup at "
            f"min_prior_folds={min_prior_folds})"
        )
    rois = [c.roi_pct for c in evaluable]
    return NestedSelectionResult(
        per_fold=tuple(choices),
        frozen_label=frozen,
        all_folds_positive=all(r > 0.0 for r in rois),
        worst_fold_roi=min(rois),
        n_negative_folds=sum(1 for r in rois if r < 0.0),
        n_evaluable_folds=len(evaluable),
        selection_name=selection_name,
    )


def _freeze_best(
    labels: Sequence[str],
    by_id: Mapping[str, Mapping[int, FoldStats]],
    order: Sequence[int],
    selection: SelectionMetric,
) -> str:
    """The config best over ALL folds by ``selection`` — the holdout freeze."""
    return max(labels, key=lambda lab: selection([by_id[lab][i] for i in order]))


def freeze_best_over_folds(
    scored: Mapping[str, Sequence[FoldStats]],
    *,
    selection: SelectionMetric = metric_mean_roi,
) -> str:
    """Public helper: the config best over ALL folds (the all-IS frozen pick)."""
    labels = sorted(scored)
    by_id = {lab: {fs.fold_id: fs for fs in scored[lab]} for lab in labels}
    order = _order_folds(labels, scored)
    return _freeze_best(labels, by_id, order, selection)


__all__ = (
    "NestedFoldChoice",
    "NestedSelectionResult",
    "nested_walk_forward_select",
    "freeze_best_over_folds",
    "metric_mean_roi",
    "metric_afp_then_mean",
    "metric_worst_then_mean",
)
