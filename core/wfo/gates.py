"""L_PROTOCOL §3 gate logic for WFO results.

A WFO candidate is classified into one of:

    Verdict.PASS_DEPLOYABLE   ships
    Verdict.PASS_VIABLE       portfolio candidate (does NOT ship alone)
    Verdict.FAIL              everything else

``classify_fold_stats`` consumes a tuple of per-fold ``FoldStats`` and
emits a ``GateResult`` with the verdict plus an explanation of which gate
failed (if any). Step 6 causal-audit cleanliness is a separate input that
the orchestrator passes through.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class Verdict(Enum):
    PASS_DEPLOYABLE = "pass_deployable"
    PASS_VIABLE = "pass_viable"
    FAIL = "fail"


@dataclass(frozen=True)
class FoldStats:
    """Per-fold metrics consumed by the gates.

    All ROI / DD values are decimal percent (e.g. 0.0192 = 1.92%).
    """

    fold_id: int
    n_trades: int
    roi_pct: float
    max_dd_pct: float
    days_breaching_daily_5pct: int  # 5ers daily-DD breach counter
    roi_dd_ratio: float  # roi_pct / max_dd_pct (caller's convention)


@dataclass(frozen=True)
class GateResult:
    verdict: Verdict
    reason: str
    worst_fold_roi: float
    worst_fold_dd: float
    worst_fold_ratio: float
    mean_fold_ratio: float
    n_negative_folds: int
    min_trades_per_fold: int
    max_dd_across_folds: float


# §3 gate constants — locked at L_PROTOCOL v3.0
WORST_FOLD_RATIO_MIN_DEPLOYABLE = 2.0
MEAN_FOLD_RATIO_MIN_VIABLE = 2.5
DD_MAX_DEPLOYABLE_PCT = 0.08  # 8% — both PASS tiers
DD_MAX_VIABLE_PCT = 0.10  # 10% — 5ers hard limit
MIN_TRADES_PER_FOLD = 25
MAX_DAYS_BREACHING_DAILY_5PCT = 0  # 5ers daily-DD invariant


def classify_fold_stats(
    folds: Sequence[FoldStats],
    causal_audit_clean: bool = True,
) -> GateResult:
    """Apply §3 PASS-DEPLOYABLE / VIABLE / FAIL gates.

    Returns a ``GateResult``; the verdict line is human-readable
    explanation suitable for landing in closure docs.

    ``causal_audit_clean`` defaults to True; orchestrator passes the real
    value when Step 6 audit has been invoked (lazy — only runs for
    PASS-tier candidates).
    """
    if not folds:
        return GateResult(
            verdict=Verdict.FAIL,
            reason="no folds evaluated",
            worst_fold_roi=0.0,
            worst_fold_dd=0.0,
            worst_fold_ratio=0.0,
            mean_fold_ratio=0.0,
            n_negative_folds=0,
            min_trades_per_fold=0,
            max_dd_across_folds=0.0,
        )

    rois = [f.roi_pct for f in folds]
    dds = [f.max_dd_pct for f in folds]
    ratios = [f.roi_dd_ratio for f in folds]
    trades = [f.n_trades for f in folds]
    daily_breaches = [f.days_breaching_daily_5pct for f in folds]

    worst_fold_roi = min(rois)
    worst_fold_dd = max(dds)
    worst_fold_ratio = min(ratios)
    mean_fold_ratio = sum(ratios) / len(ratios)
    n_negative = sum(1 for r in rois if r < 0)
    min_trades = min(trades)
    max_dd_all = max(dds)
    total_daily_breaches = sum(daily_breaches)

    def _fail(reason: str) -> GateResult:
        return GateResult(
            verdict=Verdict.FAIL,
            reason=reason,
            worst_fold_roi=worst_fold_roi,
            worst_fold_dd=worst_fold_dd,
            worst_fold_ratio=worst_fold_ratio,
            mean_fold_ratio=mean_fold_ratio,
            n_negative_folds=n_negative,
            min_trades_per_fold=min_trades,
            max_dd_across_folds=max_dd_all,
        )

    # Common minimum: trades + daily-DD invariant
    if min_trades < MIN_TRADES_PER_FOLD:
        return _fail(f"fold with only {min_trades} trades; need ≥ {MIN_TRADES_PER_FOLD}")
    if total_daily_breaches > MAX_DAYS_BREACHING_DAILY_5PCT:
        return _fail(f"{total_daily_breaches} days breaching the 5% daily-DD invariant")
    if max_dd_all > DD_MAX_VIABLE_PCT:
        return _fail(f"max DD {max_dd_all:.4%} exceeds 5ers 10% hard limit")

    # PASS-DEPLOYABLE gate
    deploy_ok = (
        worst_fold_ratio >= WORST_FOLD_RATIO_MIN_DEPLOYABLE
        and n_negative == 0
        and max_dd_all <= DD_MAX_DEPLOYABLE_PCT
        and causal_audit_clean
    )
    if deploy_ok:
        return GateResult(
            verdict=Verdict.PASS_DEPLOYABLE,
            reason=(
                f"worst-fold ratio {worst_fold_ratio:.2f} ≥ {WORST_FOLD_RATIO_MIN_DEPLOYABLE}, "
                f"0/0 negative folds, max DD {max_dd_all:.4%} ≤ {DD_MAX_DEPLOYABLE_PCT:.0%}, "
                f"causal audit clean"
            ),
            worst_fold_roi=worst_fold_roi,
            worst_fold_dd=worst_fold_dd,
            worst_fold_ratio=worst_fold_ratio,
            mean_fold_ratio=mean_fold_ratio,
            n_negative_folds=n_negative,
            min_trades_per_fold=min_trades,
            max_dd_across_folds=max_dd_all,
        )

    # PASS-VIABLE gate (one negative fold permitted).
    # L_PROTOCOL §3 reading: when a single negative fold is allowed, the
    # worst-fold-ratio constraint is checked against the worst non-negative
    # fold's ratio (the negative fold is the "permitted" exception). The
    # mean-fold-ratio constraint still uses every fold.
    positive_ratios = [r.roi_dd_ratio for r in folds if r.roi_pct >= 0]
    worst_positive_ratio = min(positive_ratios) if positive_ratios else float("-inf")
    viable_ok = (
        n_negative <= 1
        and worst_positive_ratio >= WORST_FOLD_RATIO_MIN_DEPLOYABLE
        and mean_fold_ratio >= MEAN_FOLD_RATIO_MIN_VIABLE
        and max_dd_all <= DD_MAX_VIABLE_PCT
        and causal_audit_clean
    )
    if viable_ok:
        return GateResult(
            verdict=Verdict.PASS_VIABLE,
            reason=(
                f"worst-fold ratio {worst_fold_ratio:.2f} ≥ {WORST_FOLD_RATIO_MIN_DEPLOYABLE}, "
                f"mean ratio {mean_fold_ratio:.2f} ≥ {MEAN_FOLD_RATIO_MIN_VIABLE}, "
                f"{n_negative} negative fold(s) permitted"
            ),
            worst_fold_roi=worst_fold_roi,
            worst_fold_dd=worst_fold_dd,
            worst_fold_ratio=worst_fold_ratio,
            mean_fold_ratio=mean_fold_ratio,
            n_negative_folds=n_negative,
            min_trades_per_fold=min_trades,
            max_dd_across_folds=max_dd_all,
        )

    # FAIL with most-relevant reason
    if worst_fold_ratio < WORST_FOLD_RATIO_MIN_DEPLOYABLE:
        return _fail(
            f"worst-fold ratio {worst_fold_ratio:.2f} < required {WORST_FOLD_RATIO_MIN_DEPLOYABLE}"
        )
    if not causal_audit_clean:
        return _fail("Step 6 causal audit failed for load-bearing feature")
    return _fail(
        f"worst-fold ratio {worst_fold_ratio:.2f}, "
        f"mean ratio {mean_fold_ratio:.2f}, "
        f"{n_negative} negative folds — neither PASS tier met"
    )
