 # ruff: noqa: I001
import numpy as np
import pandas as pd
import pytest

from indicators.volume_funcs import (
    volume_normalized,
    volume_silence,
    volume_stiffness,
    volume_volatility_ratio_mt4,
    volume_william_vix_fix,
    volume_waddah_attar_explosion,
)


pytestmark = pytest.mark.research


def _make_ohlcv(n: int = 60) -> pd.DataFrame:
    rng = pd.date_range("2023-01-01", periods=n, freq="D")
    base = 1.10 + np.linspace(0, 0.01, n)
    noise = np.sin(np.linspace(0, 4 * np.pi, n)) * 0.002
    close = base + noise
    open_ = close + np.random.default_rng(0).normal(0, 0.0005, size=n)
    high = np.maximum(open_, close) + 0.0005
    low = np.minimum(open_, close) - 0.0005
    volume = 1000 + (np.arange(n) % 10) * 100

    df = pd.DataFrame(
        {
            "date": rng,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    df["atr"] = 0.002
    return df


def _assert_gate(df: pd.DataFrame, col: str = "volume_signal", warmup: int = 5) -> None:
    assert col in df.columns
    vals = pd.to_numeric(df[col], errors="raise")
    assert set(vals.unique()).issubset({0, 1})
    if len(vals) > warmup:
        assert (vals.iloc[:warmup] == 0).all()


def test_volume_normalized_basic():
    df = _make_ohlcv()
    out = volume_normalized(df.copy(), signal_col="volume_signal")
    assert len(out) == len(df)
    _assert_gate(out, warmup=14)


def test_volume_volatility_ratio_mt4_basic():
    df = _make_ohlcv()
    out = volume_volatility_ratio_mt4(df.copy(), length=25, signal_col="volume_signal")
    assert len(out) == len(df)
    _assert_gate(out, warmup=25 * 2)


def test_volume_silence_basic():
    df = _make_ohlcv()
    out = volume_silence(
        df.copy(), length=12, buffer_size=24, signal_col="volume_signal"
    )
    assert len(out) == len(df)
    _assert_gate(out, warmup=30)


def test_volume_stiffness_basic():
    df = _make_ohlcv(120)
    out = volume_stiffness(
        df.copy(),
        length_trend=50,
        sum_period=30,
        signal_period=3,
        signal_col="volume_signal",
    )
    assert len(out) == len(df)
    _assert_gate(out, warmup=50 + 30)


def test_volume_william_vix_fix_basic():
    df = _make_ohlcv()
    out = volume_william_vix_fix(
        df.copy(), length=22, threshold=20.0, signal_col="volume_signal"
    )
    assert len(out) == len(df)
    _assert_gate(out, warmup=22)


def test_volume_waddah_attar_explosion_basic():
    df = _make_ohlcv(160)
    out = volume_waddah_attar_explosion(
        df.copy(),
        sensitive=150.0,
        dead_zone_mult=1.0,
        explosion_power=10.0,
        trend_power=5.0,
        length=20,
        signal_col="volume_signal",
    )
    assert len(out) == len(df)
    _assert_gate(out, warmup=60)


