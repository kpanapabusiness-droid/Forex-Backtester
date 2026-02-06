import numpy as np
import pandas as pd

from indicators.volume_funcs import volume_waddah_attar_explosion


def _make_ohlc(n: int = 900, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.006, size=n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.002, size=n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.002, size=n))
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_wae_returns_df_and_signal_col_int8():
    df = _make_ohlc()
    out = volume_waddah_attar_explosion(df, signal_col="volume_signal")
    assert isinstance(out, pd.DataFrame)
    assert out.index.equals(df.index)
    assert "volume_signal" in out.columns
    assert str(out["volume_signal"].dtype) == "int8"


def test_wae_signal_domain_is_binary():
    df = _make_ohlc()
    out = volume_waddah_attar_explosion(df)
    s = out["volume_signal"].dropna()
    assert set(s.unique()).issubset({0, 1})


def test_wae_not_stub_produces_some_passes():
    df = _make_ohlc()
    out = volume_waddah_attar_explosion(df)
    s = out["volume_signal"]
    assert int(s.sum()) > 0


def test_wae_is_deterministic():
    df = _make_ohlc()
    out1 = volume_waddah_attar_explosion(df)
    out2 = volume_waddah_attar_explosion(df)
    pd.testing.assert_series_equal(out1["volume_signal"], out2["volume_signal"])


def test_wae_no_nans_after_warmup():
    df = _make_ohlc()
    out = volume_waddah_attar_explosion(df)
    s = out["volume_signal"]
    warmup = 250
    assert not s.iloc[warmup:].isna().any()

