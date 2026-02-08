# tests/test_phaseb1_c1_archetypes.py — Phase B.1 C1 archetype contract + behavioral tests.

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests import _indicator_contracts as ic

PHASEB1_C1_NAMES = [
    "c1_regime_sm__binary",
    "c1_regime_sm__neutral_gate",
    "c1_vol_dir__binary",
    "c1_vol_dir__neutral_gate",
    "c1_persist_momo__binary",
    "c1_persist_momo__neutral_gate",
]
SIGNAL_COL = "c1_signal"
WARMUP = 200


def _get_phaseb1_functions():
    full_list = ic.discover_c1_functions()
    return [(n, f) for n, f in full_list if n in PHASEB1_C1_NAMES]


_C1_LIST = _get_phaseb1_functions()
_C1_IDS = [n for n, _ in _C1_LIST]


def _call_c1(func, df: pd.DataFrame, **kwargs):
    return func(df.copy(), signal_col=SIGNAL_COL, **kwargs)


# ----- Synthetic datasets (no market data dependency) -----


def _make_trend_up(size: int = 500, noise: float = 0.001) -> pd.DataFrame:
    np.random.seed(42)
    t = np.arange(size, dtype=float)
    close = 1.0 + 0.002 * t + noise * np.random.randn(size)
    close = np.maximum(close, 0.5)
    return _ohlcv_from_close(close)


def _make_chop(size: int = 500, mean: float = 1.0, noise: float = 0.01) -> pd.DataFrame:
    np.random.seed(43)
    close = mean + noise * np.random.randn(size)
    close = np.maximum(close, 0.5)
    return _ohlcv_from_close(close)


def _ohlcv_from_close(close: np.ndarray) -> pd.DataFrame:
    n = len(close)
    high = close + 0.0005
    low = close - 0.0005
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    vol = np.ones(n) * 1000
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


# ----- A) Contract tests -----


@pytest.mark.parametrize("name,func", _C1_LIST, ids=_C1_IDS)
def test_phaseb1_returns_dataframe(name, func):
    df = _make_trend_up(300)
    out = _call_c1(func, df)
    assert isinstance(out, pd.DataFrame), f"{name}: must return DataFrame"
    assert out.index.equals(df.index), f"{name}: index must equal input"
    assert SIGNAL_COL in out.columns, f"{name}: must write {SIGNAL_COL}"


@pytest.mark.parametrize("name,func", _C1_LIST, ids=_C1_IDS)
def test_phaseb1_signal_domain(name, func):
    df = _make_trend_up(400)
    out = _call_c1(func, df)
    s = out[SIGNAL_COL].iloc[WARMUP:]
    valid = s.dropna()
    uniq = set(int(x) for x in valid.unique())
    if "__neutral_gate" in name:
        assert uniq <= {-1, 0, 1}, f"{name}: neutral_gate must be in {{-1,0,+1}}; got {uniq}"
    else:
        assert uniq <= {-1, 1}, f"{name}: binary must be in {{-1,+1}} after warmup; got {uniq}"


@pytest.mark.parametrize("name,func", _C1_LIST, ids=_C1_IDS)
def test_phaseb1_determinism(name, func):
    df = _make_chop(350)
    out1 = _call_c1(func, df)
    out2 = _call_c1(func, df)
    pd.testing.assert_series_equal(
        out1[SIGNAL_COL], out2[SIGNAL_COL], check_names=True, check_exact=False
    )


# ----- B) Behavioral intent (lenient thresholds) -----


@pytest.mark.parametrize("name,func", _C1_LIST, ids=_C1_IDS)
def test_phaseb1_trend_up_mostly_long(name, func):
    df = _make_trend_up(500)
    out = _call_c1(func, df)
    s = out[SIGNAL_COL].iloc[WARMUP:].dropna()
    if len(s) == 0:
        pytest.skip("no valid signal after warmup")
    plus = (s == 1).sum()
    minus = (s == -1).sum()
    total = len(s)
    frac_plus = plus / total
    frac_minus = minus / total
    assert frac_minus <= 0.55, (
        f"{name}: on TREND_UP expect not mostly -1; got frac -1={frac_minus:.2f}"
    )
    if "vol_dir" not in name:
        assert frac_plus >= 0.4, (
            f"{name}: on TREND_UP expect mostly +1 or mixed; got frac +1={frac_plus:.2f}"
        )


def _count_flips(series: pd.Series) -> int:
    s = series.dropna()
    if len(s) < 2:
        return 0
    return int((s.diff().fillna(0) != 0).sum())


@pytest.mark.parametrize("name,func", _C1_LIST, ids=_C1_IDS)
def test_phaseb1_chop_neutral_gate_has_zeros(name, func):
    if "__binary" in name:
        pytest.skip("binary does not output 0 by design")
    df = _make_chop(500)
    out = _call_c1(func, df)
    s = out[SIGNAL_COL].iloc[WARMUP:].dropna()
    if len(s) == 0:
        pytest.skip("no valid signal after warmup")
    zeros = (s == 0).sum()
    frac_zero = zeros / len(s)
    assert frac_zero >= 0.05, (
        f"{name}: on CHOP neutral_gate should have meaningful 0 fraction; got {frac_zero:.2f}"
    )


@pytest.mark.parametrize("name,func", _C1_LIST, ids=_C1_IDS)
def test_phaseb1_chop_binary_fewer_flips_than_naive(name, func):
    if "__neutral_gate" in name:
        pytest.skip("flip comparison is for binary vs naive")
    df = _make_chop(500)
    out = _call_c1(func, df)
    s = out[SIGNAL_COL].iloc[WARMUP:]

    close = df["close"].astype(float)
    ema_fast = close.ewm(span=20, adjust=False).mean()
    ema_slow = close.ewm(span=50, adjust=False).mean()
    naive = np.sign(ema_fast - ema_slow)
    naive_series = pd.Series(naive, index=df.index).iloc[WARMUP:]

    flips_binary = _count_flips(s)
    flips_naive = _count_flips(naive_series)
    assert flips_binary <= flips_naive + 20, (
        f"{name}: binary should have similar or lower flip count than naive; "
        f"binary={flips_binary} naive={flips_naive}"
    )
