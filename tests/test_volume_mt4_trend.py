import numpy as np
import pandas as pd

from indicators.volume_funcs import volume_trend_direction_force


def _make_ohlcv_trend(n: int = 80) -> pd.DataFrame:
    rng = pd.date_range("2023-01-01", periods=n, freq="D")
    base = 1.10 + np.linspace(0, 0.02, n)
    noise = np.sin(np.linspace(0, 6 * np.pi, n)) * 0.001
    close = base + noise
    open_ = close + np.random.default_rng(1).normal(0, 0.0004, size=n)
    high = np.maximum(open_, close) + 0.0005
    low = np.minimum(open_, close) - 0.0005
    atr = np.full(n, 0.002)
    volume = 1000 + (np.arange(n) % 5) * 150

    return pd.DataFrame(
        {
            "date": rng,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "atr": atr,
        }
    )


def test_volume_trend_direction_force_basic():
    df = _make_ohlcv_trend()
    out = volume_trend_direction_force(
        df.copy(),
        trend_period=20,
        trigger_up=0.05,
        trigger_down=-0.05,
        smooth_length=5,
        signal_col="volume_signal",
    )
    assert len(out) == len(df)
    assert "volume_signal" in out.columns
    vals = pd.to_numeric(out["volume_signal"], errors="raise")
    assert set(vals.unique()).issubset({0, 1})
    assert (vals.iloc[:25] == 0).all()


