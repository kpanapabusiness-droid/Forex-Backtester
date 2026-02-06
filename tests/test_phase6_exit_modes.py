"""Phase 6 — Unit tests for c1_exit_mode (disagree vs flip_only) and exit_combine_mode (or)."""
import pandas as pd

from core.signal_logic import apply_signal_logic, c1_exit_now


def test_c1_exit_now_disagree_exits_on_opposite_nonzero():
    """Disagree: exit when current C1 is non-zero and opposite to position."""
    # Long (1), C1 goes +1 -> -1: exit
    assert c1_exit_now(1, -1, 1, "disagree") is True
    # Short (-1), C1 goes -1 -> +1: exit
    assert c1_exit_now(-1, 1, -1, "disagree") is True


def test_c1_exit_now_disagree_exits_when_curr_opposite_even_if_prev_zero():
    """Disagree: 0 -> -1 with long position exits (current is opposite)."""
    assert c1_exit_now(0, -1, 1, "disagree") is True
    assert c1_exit_now(0, 1, -1, "disagree") is True


def test_c1_exit_now_disagree_does_not_exit_on_neutral_current():
    """Disagree: current C1 == 0 does not trigger exit."""
    assert c1_exit_now(1, 0, 1, "disagree") is False
    assert c1_exit_now(-1, 0, -1, "disagree") is False


def test_c1_exit_now_flip_only_exits_on_full_flip():
    """Flip_only: exit only on full flip +1 <-> -1."""
    assert c1_exit_now(1, -1, 1, "flip_only") is True
    assert c1_exit_now(-1, 1, -1, "flip_only") is True


def test_c1_exit_now_flip_only_does_not_exit_on_neutral():
    """Flip_only: 0 -> -1 or 0 -> +1 does NOT trigger (no full flip)."""
    assert c1_exit_now(0, -1, 1, "flip_only") is False
    assert c1_exit_now(0, 1, -1, "flip_only") is False
    assert c1_exit_now(1, 0, 1, "flip_only") is False
    assert c1_exit_now(-1, 0, -1, "flip_only") is False


def test_c1_exit_now_flip_only_does_not_exit_same_direction():
    """Flip_only: +1 -> +1 or -1 -> -1 does not trigger."""
    assert c1_exit_now(1, 1, 1, "flip_only") is False
    assert c1_exit_now(-1, -1, -1, "flip_only") is False


def test_disagree_vs_flip_only_integration_neutral_then_opposite():
    """apply_signal_logic: bar with 0 -> -1 (long): disagree exits, flip_only does not."""
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    close = pd.Series([100.0] * 5)
    high = close + 1
    low = close - 1
    # C1: +1, +1, 0, -1 -> at bar 3 we have 0->-1 (long still open from bar 0)
    c1 = pd.Series([1, 1, 0, -1, -1])
    df = pd.DataFrame(
        {"date": dates, "open": close, "high": high, "low": low, "close": close, "volume": 1000}
    )
    df["c1_signal"] = c1
    df["atr"] = 1.0

    out_disagree = apply_signal_logic(
        df.copy(),
        {"indicators": {}, "rules": {}, "exit": {"exit_on_c1_reversal": True, "c1_exit_mode": "disagree"}},
    )
    out_flip = apply_signal_logic(
        df.copy(),
        {"indicators": {}, "rules": {}, "exit": {"exit_on_c1_reversal": True, "c1_exit_mode": "flip_only"}},
    )
    # Disagree: at bar 3 curr_c1=-1 != position 1 -> exit
    assert out_disagree.loc[3, "exit_signal_final"] == 1
    # Flip_only: at bar 3 prev=0, curr=-1 -> no full flip -> no exit at bar 3; exit at bar 4 (prev=-1, curr=-1 no) - actually at bar 4 we have prev=-1 curr=-1 so no flip. So with flip_only we never exit in this series (entry bar 0, then 1,1,0,-1,-1; at bar 4 we have prev_c1=-1 curr_c1=-1, no flip). So flip_only should not exit in this 5-bar path.
    assert out_flip.loc[3, "exit_signal_final"] == 0


def test_exit_combine_mode_or_exits_on_indicator_only():
    """D1-style OR: exit when indicator says exit even if no C1 flip."""
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    close = pd.Series([100.0] * 5)
    high = close + 1
    low = close - 1
    c1 = pd.Series([1, 1, 1, 1, 1])
    df = pd.DataFrame(
        {"date": dates, "open": close, "high": high, "low": low, "close": close, "volume": 1000}
    )
    df["c1_signal"] = c1
    df["atr"] = 1.0
    df["exit_signal"] = pd.Series([0, 0, 1, 0, 0])

    cfg = {
        "indicators": {"use_exit": True},
        "rules": {},
        "exit": {
            "exit_on_c1_reversal": False,
            "exit_on_exit_signal": True,
            "c1_exit_mode": "flip_only",
            "exit_combine_mode": "or",
        },
    }
    out = apply_signal_logic(df, cfg)
    assert out.loc[2, "exit_signal_final"] == 1
    assert out.loc[2, "exit_reason"] == "exit_indicator"


def test_exit_combine_mode_or_exits_on_flip_even_if_indicator_zero():
    """D1-style OR: exit when C1 full flip even if indicator does not fire."""
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    close = pd.Series([100.0] * 5)
    high = close + 1
    low = close - 1
    c1 = pd.Series([1, 1, -1, -1, -1])
    df = pd.DataFrame(
        {"date": dates, "open": close, "high": high, "low": low, "close": close, "volume": 1000}
    )
    df["c1_signal"] = c1
    df["atr"] = 1.0
    df["exit_signal"] = pd.Series([0, 0, 0, 0, 0])

    cfg = {
        "indicators": {"use_exit": True},
        "rules": {},
        "exit": {
            "exit_on_c1_reversal": False,
            "exit_on_exit_signal": True,
            "c1_exit_mode": "flip_only",
            "exit_combine_mode": "or",
        },
    }
    out = apply_signal_logic(df, cfg)
    assert out.loc[2, "exit_signal_final"] == 1
    assert out.loc[2, "exit_reason"] == "c1_reversal"


def test_exit_c1_name_uses_exit_c1_signal_mode_y_flip_only():
    """When exit_c1_name is set, exit uses exit_c1_signal column with Mode Y (flip_only), not entry C1."""
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    close = pd.Series([100.0] * 5)
    high = close + 1
    low = close - 1
    c1 = pd.Series([1, 1, 1, 1, 1])
    exit_c1 = pd.Series([1, 1, -1, -1, -1])
    df = pd.DataFrame(
        {"date": dates, "open": close, "high": high, "low": low, "close": close, "volume": 1000}
    )
    df["c1_signal"] = c1
    df["exit_c1_signal"] = exit_c1
    df["atr"] = 1.0
    df["exit_signal"] = 0

    cfg = {
        "indicators": {"use_exit": True},
        "rules": {},
        "exit": {
            "exit_on_c1_reversal": True,
            "exit_on_exit_signal": False,
            "c1_exit_mode": "disagree",
            "exit_combine_mode": "single",
            "exit_c1_name": "c1_coral",
        },
    }
    out = apply_signal_logic(df, cfg)
    assert out.loc[2, "exit_signal_final"] == 1
    assert out.loc[2, "exit_reason"] == "c1_reversal"
