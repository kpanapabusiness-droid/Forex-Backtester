import importlib, pandas as pd
from indicators import baseline_funcs as B

def _assert_contract(df):
    assert "baseline" in df.columns
    assert "baseline_signal" in df.columns
    assert set(df["baseline_signal"].dropna().unique()).issubset({-1,0,1})

def test_baseline_hma(dummy_ohlcv):
    df = dummy_ohlcv.copy()
    df = B.baseline_hma(df, length=21)
    _assert_contract(df)

def test_baseline_vidya(dummy_ohlcv):
    df = dummy_ohlcv.copy()
    df = B.baseline_vidya(df, length=10, cmo_length=5)
    _assert_contract(df)
