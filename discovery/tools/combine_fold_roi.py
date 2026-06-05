"""Portfolio fold-ROI combiner — discovery EXPERIMENT tool (BUILT).

Built by arc 2006 (chat 2000s). Combines the per-fold ROI series of two-or-more
ALREADY-SCORED components (each a tuple of canonical ``FoldStats`` from
``ArcFoldRunner`` / ``run_config_over_folds``) into one portfolio per-fold ROI
series, so the discovery all-folds-positive judge can be applied to the COMBINED
book (``DISCOVERY_PROTOCOL.md`` §6/§11: the combined book is its own
all-folds-positive WFO gate).

EXPERIMENT tool: it does a LINEAR combination of canonical per-fold ROI numbers —
it never realizes a trade's P&L (the engine did that upstream, SL-first). The two
weighting modes are:

  - ``"equal"``      — equal capital weight (the naive book): mean of the per-fold ROIs.
  - ``"risk_parity"``— inverse-volatility weight, ``w_k ∝ 1/σ_k`` where ``σ_k`` is
    the std of component k's per-fold ROI computed on the WEIGHTING fold set
    (IS), then FROZEN. This down-weights the high-variance component so each
    contributes equal risk — the lever arc 1011 flagged ("the gap-fill's ±8%
    swings dominate my ±1%").

CRITICAL no-lookahead discipline (§4): the risk-parity weights are a FREE
PARAMETER. Compute them ONCE on the IS per-fold ROIs (``fit_weights``), then pass
the SAME frozen weights to the OOS combination — NEVER recompute weights on OOS.

LIMITATION (documented, not hidden): this is a per-fold linear combination of two
independently-simulated books, NOT a single co-simulated equity curve. It is a
faithful first-order risk-parity portfolio only when the components trade
DISJOINT universes / rarely hold simultaneous positions (so shared-margin &
shared-daily-DD interaction is negligible). For arc 2006's pair (weekend gap-fill
on JPY crosses H4 vs month-end reversion on USD majors D1) that holds — disjoint
universes, disjoint event timing. A fully-honest single-engine co-simulation
(one SignalEvaluation, unioned universe) is the stronger test; if the canonical
runner ever supports multi-signal co-simulation, prefer it. Until then this tool
gives the honest first-order combined book and is explicit about the assumption.

Folds MUST be aligned: component k's fold j must be the SAME calendar window as
component m's fold j (both built from ``build_v3_folds`` / ``build_oos_year_folds``,
which are date-range folds — TF/universe-agnostic — so fold j is the same period
for an H4 and a D1 signal). The caller passes the per-fold ROI in fold order; this
tool asserts equal length and combines index-by-index.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from core.wfo.gates import FoldStats


def _std(xs: Sequence[float]) -> float:
    """Population-ish std (ddof=1 when n>1) without importing numpy semantics."""
    n = len(xs)
    if n == 0:
        raise ValueError("std of empty sequence")
    if n == 1:
        return 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return var ** 0.5


def rois_from_fold_stats(fold_stats: Sequence[FoldStats]) -> list[float]:
    """Extract per-fold ROI (decimal fraction, e.g. 0.0069 = 0.69%) in fold order."""
    return [fs.roi_pct for fs in fold_stats]


def fit_weights(
    component_is_rois: Sequence[Sequence[float]],
    mode: str = "risk_parity",
) -> list[float]:
    """Compute portfolio weights from the IS per-fold ROI series of each component.

    ``component_is_rois``: one per-fold ROI sequence per component (IS fold set).
    Returns weights that sum to 1.0, in component order. FREEZE these for OOS.
    """
    k = len(component_is_rois)
    if k == 0:
        raise ValueError("need ≥1 component")
    if mode == "equal":
        return [1.0 / k] * k
    if mode == "risk_parity":
        inv = []
        for rois in component_is_rois:
            s = _std(rois)
            inv.append(0.0 if s == 0.0 else 1.0 / s)
        total = sum(inv)
        if total == 0.0:  # all-zero-variance degenerate -> fall back to equal
            return [1.0 / k] * k
        return [w / total for w in inv]
    raise ValueError(f"unknown mode {mode!r}; use 'equal' or 'risk_parity'")


@dataclass(frozen=True)
class CombinedBook:
    """Per-fold combined ROI series + the weights used (frozen across IS/OOS)."""

    weights: tuple[float, ...]
    combined_roi: tuple[float, ...]  # decimal fraction per fold, fold order
    component_roi: tuple[tuple[float, ...], ...]  # per-component per-fold (for the record)


def combine_fold_rois(
    component_rois: Sequence[Sequence[float]],
    weights: Sequence[float],
) -> CombinedBook:
    """Linear-combine aligned per-fold ROI series with given (frozen) weights.

    All components must have the SAME number of folds (aligned by fold order).
    """
    if len(component_rois) != len(weights):
        raise ValueError(
            f"{len(component_rois)} components vs {len(weights)} weights"
        )
    n_folds = {len(r) for r in component_rois}
    if len(n_folds) != 1:
        raise ValueError(f"components have mismatched fold counts: {n_folds}")
    nf = n_folds.pop()
    combined = []
    for j in range(nf):
        combined.append(sum(weights[k] * component_rois[k][j] for k in range(len(weights))))
    return CombinedBook(
        weights=tuple(weights),
        combined_roi=tuple(combined),
        component_roi=tuple(tuple(r) for r in component_rois),
    )


__all__ = (
    "rois_from_fold_stats",
    "fit_weights",
    "combine_fold_rois",
    "CombinedBook",
)
