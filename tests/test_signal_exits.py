import pandas as pd

from core.signal_logic import apply_signal_logic


def _df_with_signals():
    # craft a simple c1 flip around mid
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    close = pd.Series([100, 100, 100, 100, 100, 100])
    high = close + 1
    low = close - 1
    # c1: 0, +1, +1, -1, -1, ...
    c1 = pd.Series([0, 1, 1, -1, -1, -1])
    df = pd.DataFrame(
        {"date": dates, "open": close, "high": high, "low": low, "close": close, "volume": 1000}
    )
    df["c1_signal"] = c1
    df["atr"] = 1.0
    return df


def test_exit_on_c1_reversal_enabled():
    df = _df_with_signals()
    cfg = {"indicators": {}, "rules": {}, "exit": {"exit_on_c1_reversal": True}}
    out = apply_signal_logic(df, cfg)
    # exit should appear at bar where yesterday flipped vs prev2 (i=3)
    assert out.loc[3, "exit_signal"] == 1


def test_exit_on_c1_reversal_disabled():
    df = _df_with_signals()
    cfg = {"indicators": {}, "rules": {}, "exit": {"exit_on_c1_reversal": False}}
    out = apply_signal_logic(df, cfg)
    assert out.loc[3, "exit_signal"] == 0  # Should be 0 (int) when disabled, not NaN


def test_exit_indicator_signal():
    """Test that exit indicator signal triggers exit when exit_on_exit_signal=True"""
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    close = pd.Series([100, 100, 100, 100, 100, 100])
    high = close + 1
    low = close - 1
    # c1: +1 throughout (no reversal)
    c1 = pd.Series([1, 1, 1, 1, 1, 1])
    df = pd.DataFrame(
        {"date": dates, "open": close, "high": high, "low": low, "close": close, "volume": 1000}
    )
    df["c1_signal"] = c1
    df["atr"] = 1.0
    # Exit indicator signals exit at bar 3
    df["exit_signal"] = pd.Series([0, 0, 0, 1, 0, 0])
    
    cfg = {
        "indicators": {},
        "rules": {},
        "exit": {"exit_on_c1_reversal": False, "exit_on_exit_signal": True},
    }
    out = apply_signal_logic(df, cfg)
    
    # Entry should occur at bar 0 (C1 signal present)
    assert out.loc[0, "entry_signal"] == 1
    # Exit should be triggered by exit indicator at bar 3
    assert out.loc[3, "exit_signal_final"] == 1
    assert out.loc[3, "exit_reason"] == "exit_indicator"


def test_exit_indicator_priority_after_c1_reversal():
    """Test that C1 reversal takes priority over exit indicator"""
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    close = pd.Series([100, 100, 100, 100, 100, 100])
    high = close + 1
    low = close - 1
    # c1: +1, +1, -1 (reversal at bar 2)
    c1 = pd.Series([1, 1, -1, -1, -1, -1])
    df = pd.DataFrame(
        {"date": dates, "open": close, "high": high, "low": low, "close": close, "volume": 1000}
    )
    df["c1_signal"] = c1
    df["atr"] = 1.0
    # Exit indicator also signals at bar 2
    df["exit_signal"] = pd.Series([0, 0, 1, 0, 0, 0])
    
    cfg = {
        "indicators": {},
        "rules": {},
        "exit": {"exit_on_c1_reversal": True, "exit_on_exit_signal": True},
    }
    out = apply_signal_logic(df, cfg)
    
    # Entry at bar 0
    assert out.loc[0, "entry_signal"] == 1
    # Exit at bar 2 should be due to C1 reversal (higher priority)
    assert out.loc[2, "exit_signal_final"] == 1
    assert out.loc[2, "exit_reason"] == "c1_reversal"


def test_exit_indicator_does_not_affect_entries():
    """Test that exit indicator signals do not prevent entries, but can exit same bar"""
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    close = pd.Series([100, 100, 100, 100, 100, 100])
    high = close + 1
    low = close - 1
    # c1: +1 throughout
    c1 = pd.Series([1, 1, 1, 1, 1, 1])
    df = pd.DataFrame(
        {"date": dates, "open": close, "high": high, "low": low, "close": close, "volume": 1000}
    )
    df["c1_signal"] = c1
    df["atr"] = 1.0
    # Exit indicator signals at bar 1 (after entry at bar 0)
    df["exit_signal"] = pd.Series([0, 1, 0, 0, 0, 0])
    
    cfg = {
        "indicators": {},
        "rules": {},
        "exit": {"exit_on_c1_reversal": False, "exit_on_exit_signal": True},
    }
    out = apply_signal_logic(df, cfg)
    
    # Entry should occur at bar 0 (exit indicator doesn't prevent entries)
    assert out.loc[0, "entry_signal"] == 1
    # Exit should trigger at bar 1 when exit indicator signals (position is open)
    assert out.loc[1, "exit_signal_final"] == 1
    assert out.loc[1, "exit_reason"] == "exit_indicator"