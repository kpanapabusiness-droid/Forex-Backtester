"""Phase 6.2 — Leaderboard decision logic (no WFO/disk in tests)."""

from analytics.phase6_c1_as_exit_leaderboard import _compute_decision


def test_compute_decision_insufficient_trades_total():
    """trade_count_total=21 -> REJECT insufficient_trades_total."""
    decision, reason = _compute_decision(21, None, False, False)
    assert decision == "REJECT"
    assert reason == "insufficient_trades_total"


def test_compute_decision_insufficient_trades_fold():
    """min_fold < 50 -> REJECT insufficient_trades_fold."""
    decision, reason = _compute_decision(400, 30, False, False)
    assert decision == "REJECT"
    assert reason == "insufficient_trades_fold"


def test_compute_decision_pass():
    """Enough trades and no churn -> PASS."""
    decision, reason = _compute_decision(400, 60, False, False)
    assert decision == "PASS"
    assert reason == ""


def test_compute_decision_churn():
    """Churn flags -> REJECT churn."""
    decision, reason = _compute_decision(400, 60, True, False)
    assert decision == "REJECT"
    assert reason == "churn"
