"""Phase 6.3 — Final report helper logic tests (DD improvement, thresholds, churn)."""

from analytics.phase6_exit_final_report import (
    _churn_breach,
    _dd_improvement_rel,
    _mode_qualifies,
)


def test_dd_improvement_sign_logic():
    """DD improvement is positive when candidate DD is less negative (e.g. -17 vs -18)."""
    baseline_dd = -18.0
    better_dd = -17.0
    worse_dd = -19.0
    assert _dd_improvement_rel(better_dd, baseline_dd) > 0.0
    assert _dd_improvement_rel(worse_dd, baseline_dd) < 0.0


def test_dd_improvement_10pct_threshold():
    """10% relative DD improvement threshold is applied."""
    baseline_dd = -18.0
    # ~9% improvement: |baseline|=18, candidate=16.38 -> (18-16.38)/18 ≈ 0.09
    almost_dd = -16.38
    qualifies, qualifies_by, dd_impr, _ = _mode_qualifies(
        dd=almost_dd,
        exp=0.0,
        ref_dd=baseline_dd,
        ref_exp=0.0,
    )
    assert not qualifies
    assert qualifies_by == "none" or dd_impr < 0.10

    # ~11% improvement: candidate=-16.0 -> (18-16)/18 ≈ 0.11
    better_dd = -16.0
    qualifies, qualifies_by, dd_impr, _ = _mode_qualifies(
        dd=better_dd,
        exp=0.0,
        ref_dd=baseline_dd,
        ref_exp=0.0,
    )
    assert qualifies
    assert qualifies_by in ("dd", "both")
    assert dd_impr >= 0.10


def test_churn_hard_reject_limits():
    """Churn hard limits: trade_ratio>1.25 or hold_ratio<0.70 -> breach."""
    assert _churn_breach(1.30, 0.80) is True
    assert _churn_breach(1.00, 0.60) is True
    assert _churn_breach(1.20, 0.80) is False
    assert _churn_breach(float("nan"), float("nan")) is False

