"""Verify the new amended-gate logic against Arcs 8/10/11 manual re-evaluation.

Per dispatch §"Group 9" + chat directive Q9: run the new
`classify_amended_fold_stats` against the metrics each closure's §10
"Amendment 3 re-evaluation" block uses; assert the engine reproduces
the closure's outcomes within reasonable tolerance.

Sources:
  - results/l_arc_8/ARC_CLOSURE.md  §10 Amendment 3 re-evaluation
  - results/l_arc_10/ARC_CLOSURE.md §10 Amendment 3 re-evaluation
  - results/l_arc_11/ARC_CLOSURE.md §10 Amendment 3 re-evaluation

Engine-wins-on-tiebreak: any discrepancy between engine output and the
closure's hand-derived numbers is a housekeeping issue for the
closure, not a bug in the engine. Each discrepancy is documented
inline below; the test asserts what the engine produces.

Key numbers extracted (from §1 tracker_payload of each closure):

Arc 8 (A6 meta_labeling, FAIL):
  worst_fold_roi_base_pct = 1.7486 (decimal: 0.017486)
  worst_fold_dd_base_pct  = 1.9826 (decimal: 0.019826)
  closure §10 says: step5_ratio_below_gate_after_scaling

Arc 10 (A1 system_level_filter, PASS-DEPLOYABLE-PROVISIONAL → PASS-DEPLOYABLE):
  worst_fold_roi_base_pct = 26.49 (decimal: 0.2649)
  worst_fold_dd_base_pct  =  9.22 (decimal: 0.0922)
  closure §10 says: PASS-VIABLE → PASS-DEPLOYABLE-PROVISIONAL (Amendment 3)

Arc 11 (A2 classifier_filter, FAIL):
  worst_fold_roi_base_pct = -24.0273 (decimal: -0.240273)
  worst_fold_dd_base_pct  =  38.36 (decimal: 0.3836)
  closure §10 says: step5_not_scalable
"""

from __future__ import annotations

import pytest

from core.wfo.amended_gates import (
    R_MAX,
    R_MIN,
    AmendedVerdict,
    PrimaryFailureMode,
    classify_amended_fold_stats,
    compute_scaling_factors,
)
from core.wfo.gates import FoldStats


def _synthesise_folds(
    *,
    worst_roi_base: float,
    worst_dd_base: float,
    n_folds: int = 11,
    fill_roi: float = 0.03,
    fill_dd: float = 0.03,
    fill_ratio: float = 3.0,
    n_trades_per_fold: int = 50,
    n_negative: int = 0,
) -> tuple[FoldStats, ...]:
    """Synthesize a FoldStats sequence whose roll-up matches the arc.

    Places ``worst_roi_base`` / ``worst_dd_base`` in fold 1; remaining
    folds filled with `fill_*` values. Sufficient for gate-logic
    reproduction (the gate consumes only min/max/mean of per-fold
    metrics, not the equity curves).
    """
    worst_ratio = (
        worst_roi_base / worst_dd_base if worst_dd_base > 0 else 0.0
    )
    folds = [
        FoldStats(
            fold_id=1, n_trades=n_trades_per_fold,
            roi_pct=worst_roi_base, max_dd_pct=worst_dd_base,
            days_breaching_daily_5pct=0, roi_dd_ratio=worst_ratio,
        )
    ]
    # Fill the rest with healthy folds; if n_negative > 0, make one of
    # the remaining folds negative
    for i in range(2, n_folds + 1):
        if i == 2 and n_negative > 0:
            folds.append(FoldStats(
                fold_id=i, n_trades=n_trades_per_fold,
                roi_pct=-0.01, max_dd_pct=0.02,
                days_breaching_daily_5pct=0, roi_dd_ratio=-0.5,
            ))
        else:
            folds.append(FoldStats(
                fold_id=i, n_trades=n_trades_per_fold,
                roi_pct=fill_roi, max_dd_pct=fill_dd,
                days_breaching_daily_5pct=0, roi_dd_ratio=fill_ratio,
            ))
    return tuple(folds)


# ── Arc 11 — step5_not_scalable ────────────────────────────────────


def test_arc_11_reproduces_step5_not_scalable() -> None:
    """Arc 11: worst_fold_dd_base = 38.36% → both r_safe and r_hard
    push BELOW R_MIN = 0.15% → step5_not_scalable.

    Reproduces closure §10: "Re-evaluated primary_failure_mode:
    step5_not_scalable (per Amendment 3 §3 priority order)".
    """
    # k_safe = 8/38.36 = 0.2086, r_safe = 0.005 * 0.2086 = 0.001043 < R_MIN
    # k_hard = 10/38.36 = 0.2608, r_hard = 0.005 * 0.2608 = 0.001304 < R_MIN
    sf = compute_scaling_factors(0.3836, r_base=0.005)
    assert sf.scalable_to_safe is False
    assert sf.scalable_to_hard is False
    assert sf.r_safe_pct < R_MIN
    assert sf.r_hard_pct < R_MIN

    folds = _synthesise_folds(
        worst_roi_base=-0.240273, worst_dd_base=0.3836,
        n_negative=1,
    )
    res = classify_amended_fold_stats(
        folds=folds,
        chained_max_dd_base_pct=0.3836,   # at least worst_fold_dd
        per_day_max_dd_df=None,
        holdout_stats_at_r_safe=None,
        holdout_stats_at_r_hard=None,
        sizing_convention="reset_floor",
        accept_equity_pct=False,
    )
    assert res.verdict == AmendedVerdict.FAIL
    assert res.primary_failure_mode == PrimaryFailureMode.STEP5_NOT_SCALABLE


# ── Arc 8 — engine reproduces partial closure but flags scalability ──


def test_arc_8_reproduces_fail_verdict() -> None:
    """Arc 8: worst_fold_dd_base = 1.9826% → very low DD pushes r_safe
    JUST above R_MAX = 2.00%. Engine reports step5_not_scalable.

    Closure §10 reports step5_ratio_below_gate_after_scaling. The
    discrepancy is the closure's hand-derivation didn't apply the
    R_MAX = 2.00% upper bound check; it concluded the arc was
    scalable and failed on the ratio gate. Per dispatch §"Group 9":
    engine wins on tiebreak. Verdict (FAIL) is identical in both
    readings.

    Documented as a housekeeping follow-up for the closure: re-issue
    Arc 8's §10 with the corrected primary_failure_mode after this PR
    merges.
    """
    # k_safe = 8/1.9826 = 4.0351; r_safe = 0.005 * 4.0351 = 0.020175 > R_MAX
    # k_hard = 10/1.9826 = 5.0439; r_hard = 0.005 * 5.0439 = 0.025220 > R_MAX
    sf = compute_scaling_factors(0.019826, r_base=0.005)
    assert sf.r_safe_pct > R_MAX, (
        f"r_safe = {sf.r_safe_pct:.6%} should exceed R_MAX = {R_MAX:.2%}"
    )
    assert sf.scalable_to_safe is False
    assert sf.scalable_to_hard is False

    folds = _synthesise_folds(
        worst_roi_base=0.017486, worst_dd_base=0.019826,
        # All folds at 1.5 ratio (below 2.0) to mirror Arc 8's ratio
        # profile — the closure's §3 math reads 0.882
        fill_roi=0.025, fill_dd=0.018, fill_ratio=1.39,
    )
    res = classify_amended_fold_stats(
        folds=folds,
        chained_max_dd_base_pct=0.019826,
        per_day_max_dd_df=None,
        holdout_stats_at_r_safe=None,
        holdout_stats_at_r_hard=None,
        sizing_convention="reset_floor",
        accept_equity_pct=False,
    )
    # Verdict identical to closure: FAIL
    assert res.verdict == AmendedVerdict.FAIL
    # Engine primary_failure_mode is step5_not_scalable (closure says
    # step5_ratio_below_gate_after_scaling — discrepancy documented above)
    assert res.primary_failure_mode == PrimaryFailureMode.STEP5_NOT_SCALABLE


# ── Arc 10 — PASS-DEPLOYABLE under Amendment 3 ──────────────────────


def test_arc_10_scaling_factors_match_closure() -> None:
    """Arc 10: worst_fold_dd_base = 9.22% → k_safe ≈ 0.8677, r_safe = 0.4339%.

    Closure §10 explicitly quotes: "k_safe = 0.8677". Reproduce.
    """
    sf = compute_scaling_factors(0.0922, r_base=0.005)
    assert sf.k_safe == pytest.approx(0.8677, abs=1e-3)
    assert sf.r_safe_pct == pytest.approx(0.005 * 0.8677, abs=1e-5)
    assert sf.scalable_to_safe is True
    # r_hard = 0.005 * 10/9.22 = 0.005 * 1.0846 = 0.005423 — within
    # [R_MIN=0.15%, R_MAX=2.0%]
    assert sf.scalable_to_hard is True


def test_arc_10_reproduces_pass_deployable_provisional() -> None:
    """Arc 10: PASS-DEPLOYABLE-PROVISIONAL.

    With clean Step 6 + holdout matching the IS profile, the engine
    should arrive at PASS-DEPLOYABLE. The "provisional" framing in
    closure §10 reflects unmeasured chained DD / daily DD; this test
    feeds explicit values so the engine sees the same state.

    Per the closure's §10 declaration: "chained_max_dd_base_pct ≤
    11.52% (scales to ≤ 10% at k_safe = 0.87)". Use a value within
    that bound.
    """
    folds = _synthesise_folds(
        worst_roi_base=0.2649, worst_dd_base=0.0922,
        # Mean ratio strongly above 2.5 to support PASS-DEPLOYABLE
        fill_roi=0.20, fill_dd=0.06, fill_ratio=3.5,
        n_negative=0,
    )
    # Provide a holdout that matches Arc 10's profile
    holdout_safe = FoldStats(
        fold_id=12, n_trades=50, roi_pct=0.15, max_dd_pct=0.05,
        days_breaching_daily_5pct=0, roi_dd_ratio=3.0,
    )
    res = classify_amended_fold_stats(
        folds=folds,
        chained_max_dd_base_pct=0.10,   # within the closure's ≤ 11.52% bound
        per_day_max_dd_df=None,
        holdout_stats_at_r_safe=holdout_safe,
        holdout_stats_at_r_hard=None,
        sizing_convention="reset_floor",
        accept_equity_pct=False,
        causal_audit_clean=True,
    )
    # Closure §10 finalised PASS-DEPLOYABLE post Step 6 (causal audit
    # clean). The engine reaches the same verdict.
    assert res.verdict == AmendedVerdict.PASS_DEPLOYABLE
    assert res.primary_failure_mode == PrimaryFailureMode.NONE
    assert res.scalable_to_safe is True
    # k_safe should match the closure's quoted 0.8677 within rounding
    assert res.k_safe == pytest.approx(0.8677, abs=1e-3)


# ── Summary tabulation (informational; test never fails) ────────────


def test_reconciliation_summary_printable() -> None:
    """Informational: print the reconciliation table for human review.

    Not a hard assertion. Surfaces the engine-vs-closure deltas the
    dispatch §"Group 9" called for.
    """
    rows: list[dict] = []
    for arc, dd, closure_mode in [
        ("Arc 8",  0.019826, "step5_ratio_below_gate_after_scaling"),
        ("Arc 10", 0.0922,   "PASS-DEPLOYABLE (post Step 6)"),
        ("Arc 11", 0.3836,   "step5_not_scalable"),
    ]:
        sf = compute_scaling_factors(dd, r_base=0.005)
        rows.append({
            "arc": arc,
            "dd_base_pct": dd,
            "k_safe": round(sf.k_safe, 4),
            "r_safe_pct": round(sf.r_safe_pct, 6),
            "scalable_to_safe": sf.scalable_to_safe,
            "closure_says": closure_mode,
        })
    # Pretty-printed via repr is enough for human review when the test
    # runs with -v / -s. CI suppresses stdout so this is for dev use.
    for r in rows:
        print(r)
    assert len(rows) == 3
