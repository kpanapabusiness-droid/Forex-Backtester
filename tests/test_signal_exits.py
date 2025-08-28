import pandas as pd

from signal_logic import apply_signal_logic


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
    assert pd.isna(out.loc[3, "exit_signal"])
