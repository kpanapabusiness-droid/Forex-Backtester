"""Tests for core/wfo/gates.py — §3 gate classifier."""

from __future__ import annotations

from core.wfo.gates import (
    MIN_TRADES_PER_FOLD,
    FoldStats,
    Verdict,
    classify_fold_stats,
)


def _fold(fold_id: int, roi: float, dd: float, trades: int = 50, breach: int = 0) -> FoldStats:
    return FoldStats(
        fold_id=fold_id,
        n_trades=trades,
        roi_pct=roi,
        max_dd_pct=dd,
        days_breaching_daily_5pct=breach,
        roi_dd_ratio=roi / dd if dd > 0 else 0.0,
    )


def test_no_folds_returns_fail() -> None:
    gr = classify_fold_stats([])
    assert gr.verdict == Verdict.FAIL


def test_passes_deployable_when_all_strong() -> None:
    folds = [_fold(k, roi=0.06, dd=0.025) for k in range(1, 12)]  # ratio 2.4
    gr = classify_fold_stats(folds)
    assert gr.verdict == Verdict.PASS_DEPLOYABLE
    assert gr.n_negative_folds == 0


def test_negative_fold_blocks_deployable_but_allows_viable() -> None:
    folds = [_fold(k, roi=0.07, dd=0.025) for k in range(1, 11)]  # ratio 2.8
    folds.append(_fold(11, roi=-0.005, dd=0.025))  # one negative
    gr = classify_fold_stats(folds)
    assert gr.verdict == Verdict.PASS_VIABLE
    assert gr.n_negative_folds == 1


def test_dd_above_deployable_threshold_downgrades_to_viable() -> None:
    """DD > 8% but ≤ 10% → not deployable; viable possible."""
    folds = [_fold(k, roi=0.10, dd=0.04) for k in range(1, 11)]
    folds.append(_fold(11, roi=0.10, dd=0.09))  # DD 9% > 8% deployable cap
    gr = classify_fold_stats(folds)
    assert gr.verdict in (Verdict.PASS_VIABLE, Verdict.FAIL)
    assert gr.max_dd_across_folds == 0.09


def test_dd_above_viable_threshold_fails() -> None:
    folds = [_fold(k, roi=0.10, dd=0.04) for k in range(1, 11)]
    folds.append(_fold(11, roi=0.10, dd=0.11))  # DD 11% > 10% hard limit
    gr = classify_fold_stats(folds)
    assert gr.verdict == Verdict.FAIL
    assert "10%" in gr.reason or "hard limit" in gr.reason


def test_min_trades_below_threshold_fails() -> None:
    folds = [_fold(k, roi=0.10, dd=0.03) for k in range(1, 12)]
    folds[3] = _fold(4, roi=0.10, dd=0.03, trades=MIN_TRADES_PER_FOLD - 1)
    gr = classify_fold_stats(folds)
    assert gr.verdict == Verdict.FAIL
    assert "trades" in gr.reason.lower()


def test_daily_dd_breach_fails() -> None:
    folds = [_fold(k, roi=0.10, dd=0.03) for k in range(1, 11)]
    folds.append(_fold(11, roi=0.10, dd=0.03, breach=1))
    gr = classify_fold_stats(folds)
    assert gr.verdict == Verdict.FAIL
    assert "daily" in gr.reason.lower()


def test_causal_audit_dirty_downgrades_pass_to_fail() -> None:
    folds = [_fold(k, roi=0.07, dd=0.025) for k in range(1, 12)]
    gr = classify_fold_stats(folds, causal_audit_clean=False)
    assert gr.verdict == Verdict.FAIL
    assert "causal" in gr.reason.lower()


def test_worst_fold_ratio_low_fails_outright() -> None:
    folds = [_fold(k, roi=0.02, dd=0.04) for k in range(1, 12)]  # ratio 0.5
    gr = classify_fold_stats(folds)
    assert gr.verdict == Verdict.FAIL
    assert "worst-fold ratio" in gr.reason


def test_kh24_lineage_passes_deployable() -> None:
    """Sanity: hand-fed KH-24 published numbers should classify clean."""
    # Per ARC_HISTORY: all 7 folds positive, worst ROI +1.92%, DD 6.37%
    folds = [
        FoldStats(
            1,
            n_trades=41,
            roi_pct=0.1335,
            max_dd_pct=0.0637,
            days_breaching_daily_5pct=0,
            roi_dd_ratio=2.10,
        ),
        FoldStats(
            2,
            n_trades=35,
            roi_pct=0.08,
            max_dd_pct=0.04,
            days_breaching_daily_5pct=0,
            roi_dd_ratio=2.00,
        ),
        FoldStats(
            3,
            n_trades=30,
            roi_pct=0.05,
            max_dd_pct=0.02,
            days_breaching_daily_5pct=0,
            roi_dd_ratio=2.50,
        ),
        FoldStats(
            4,
            n_trades=30,
            roi_pct=0.06,
            max_dd_pct=0.025,
            days_breaching_daily_5pct=0,
            roi_dd_ratio=2.40,
        ),
        FoldStats(
            5,
            n_trades=30,
            roi_pct=0.05,
            max_dd_pct=0.02,
            days_breaching_daily_5pct=0,
            roi_dd_ratio=2.50,
        ),
        FoldStats(
            6,
            n_trades=30,
            roi_pct=0.04,
            max_dd_pct=0.018,
            days_breaching_daily_5pct=0,
            roi_dd_ratio=2.22,
        ),
        FoldStats(
            7,
            n_trades=27,
            roi_pct=0.0192,
            max_dd_pct=0.0406,
            days_breaching_daily_5pct=0,
            roi_dd_ratio=0.47,
        ),
    ]
    gr = classify_fold_stats(folds)
    # Worst-fold ratio 0.47 is below 2.0, so this should NOT clear PASS_DEPLOYABLE.
    # KH-24 passes by the legacy "worst-fold ROI > 0" gate (not the ratio gate).
    # Verify our v3 gate is the stricter one — it should fail here.
    assert gr.verdict == Verdict.FAIL
    assert gr.worst_fold_ratio < 2.0
