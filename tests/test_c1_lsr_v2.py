# tests/test_c1_lsr_v2.py — C1 LSR v2 indicator unit tests.

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.confirmation_funcs import c1_lsr_v2

SIGNAL_COL = "c1_signal"
VALID_SIGNALS = {-1, 0, 1}


def _make_ohlcv(n: int, base: float = 1.0) -> pd.DataFrame:
    """Build deterministic OHLCV with DateTimeIndex; flat baseline bars."""
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    open_ = pd.Series([base] * n, index=dates)
    high = pd.Series([base + 0.005] * n, index=dates)
    low = pd.Series([base - 0.005] * n, index=dates)
    close = pd.Series([base] * n, index=dates)
    volume = pd.Series([100] * n, index=dates)
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_contract():
    """Returns DataFrame, c1_signal exists, values in {-1,0,+1}, length/index unchanged."""
    df = _make_ohlcv(50)
    params = dict(signal_col=SIGNAL_COL, lookback_n=5, atr_period=5)
    out = c1_lsr_v2(df.copy(), **params)
    assert isinstance(out, pd.DataFrame)
    assert SIGNAL_COL in out.columns
    assert len(out) == len(df)
    assert out.index.equals(df.index)
    s = out[SIGNAL_COL]
    uniq = set(pd.to_numeric(s, errors="coerce").dropna().astype(int).unique())
    assert uniq <= VALID_SIGNALS, f"Signal values must be in {VALID_SIGNALS}; got {uniq}"


def test_determinism():
    """Same input and params -> identical signal series."""
    df = _make_ohlcv(60)
    params = dict(signal_col=SIGNAL_COL, lookback_n=8, variant="A_pin")
    out1 = c1_lsr_v2(df.copy(), **params)
    out2 = c1_lsr_v2(df.copy(), **params)
    pd.testing.assert_series_equal(
        out1[SIGNAL_COL], out2[SIGNAL_COL], check_names=True, check_exact=False
    )


def test_causality_no_lookahead():
    """Only last 5 bars differ; signals before last 5 must be identical."""
    df_base = _make_ohlcv(60)
    df_future = df_base.copy()
    for col in ("open", "high", "low", "close"):
        df_future.loc[df_future.index[-5:], col] += 0.01
    params = dict(signal_col=SIGNAL_COL, lookback_n=5, atr_period=5)
    out_base = c1_lsr_v2(df_base, **params)
    out_future = c1_lsr_v2(df_future, **params)
    before = slice(None, -5)
    pd.testing.assert_series_equal(
        out_base[SIGNAL_COL].iloc[before].reset_index(drop=True),
        out_future[SIGNAL_COL].iloc[before].reset_index(drop=True),
        check_names=True,
        check_exact=False,
    )


def test_variant_a_pin_bullish_trigger():
    """Bullish: prior_low exists, one bar sweeps below, big lower wick, closes above -> +1."""
    n = 50
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    base = 1.02
    open_ = np.full(n, base)
    high = np.full(n, base + 0.005)
    low = np.full(n, base - 0.005)
    close = np.full(n, base)

    lookback_n = 5
    warmup = lookback_n + 14 + 5
    trig = warmup

    for i in range(trig):
        low[i] = base - 0.005
        high[i] = base + 0.005

    prior_swing_low = base - 0.005
    sweep_low = prior_swing_low - 0.020
    rng = 0.10
    open_[trig] = base + 0.065
    low[trig] = sweep_low
    high[trig] = sweep_low + rng
    close[trig] = prior_swing_low + 0.065

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 100,
    }, index=dates)
    params = dict(
        signal_col=SIGNAL_COL,
        variant="A_pin",
        lookback_n=lookback_n,
        sweep_atr=0.05,
        wick_min_frac=0.60,
        body_max_frac=0.45,
        close_pos_min=0.65,
        reclaim_frac=0.05,
        cooldown_bars=2,
        range_cap_atr=0,
        min_range_atr=0,
    )
    out = c1_lsr_v2(df.copy(), **params)
    sig = out[SIGNAL_COL]
    nonzero = np.where(sig.values != 0)[0]
    assert len(nonzero) == 1, f"Expected exactly one signal; got {len(nonzero)} at {nonzero}"
    assert nonzero[0] == trig, f"Expected signal at index {trig}; got {nonzero[0]}"
    assert sig.iloc[trig] == 1, f"Expected +1 at trigger index {trig}; got {sig.iloc[trig]}"


def test_variant_a_pin_bearish_trigger():
    """Bearish: prior_high exists, one bar sweeps above, big upper wick, closes below -> -1."""
    n = 50
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    base = 1.02
    open_ = np.full(n, base)
    high = np.full(n, base + 0.005)
    low = np.full(n, base - 0.005)
    close = np.full(n, base)

    lookback_n = 5
    warmup = lookback_n + 14 + 5
    trig = warmup

    for i in range(trig):
        high[i] = base + 0.005
        low[i] = base - 0.005

    prior_swing_high = base + 0.005
    sweep_high = prior_swing_high + 0.020
    rng = 0.10
    open_[trig] = base - 0.065
    high[trig] = sweep_high
    low[trig] = sweep_high - rng
    close[trig] = prior_swing_high - 0.065

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 100,
    }, index=dates)
    params = dict(
        signal_col=SIGNAL_COL,
        variant="A_pin",
        lookback_n=lookback_n,
        sweep_atr=0.05,
        wick_min_frac=0.60,
        body_max_frac=0.45,
        close_pos_min=0.65,
        reclaim_frac=0.05,
        cooldown_bars=2,
        range_cap_atr=0,
        min_range_atr=0,
    )
    out = c1_lsr_v2(df.copy(), **params)
    sig = out[SIGNAL_COL]
    nonzero = np.where(sig.values != 0)[0]
    assert len(nonzero) == 1, f"Expected exactly one signal; got {len(nonzero)} at {nonzero}"
    assert nonzero[0] == trig, f"Expected signal at index {trig}; got {nonzero[0]}"
    assert sig.iloc[trig] == -1, f"Expected -1 at trigger index {trig}; got {sig.iloc[trig]}"


def test_variant_b_confirm_trigger():
    """Sweep/rejection bar at i, confirm bar i+1 closes above high[i] -> signal at i+1, not i."""
    n = 55
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    base = 1.02
    lookback_n = 5
    warmup = lookback_n + 14 + 5
    rej_bar = warmup
    conf_bar = warmup + 1

    open_ = np.full(n, base)
    high = np.full(n, base + 0.005)
    low = np.full(n, base - 0.005)
    close = np.full(n, base)

    prior_swing_low = base - 0.005
    sweep_low = prior_swing_low - 0.025
    rng = 0.10

    open_[rej_bar] = base + 0.065
    low[rej_bar] = sweep_low
    high[rej_bar] = sweep_low + rng
    close[rej_bar] = prior_swing_low + 0.065

    rej_high = high[rej_bar]
    open_[conf_bar] = rej_high - 0.002
    high[conf_bar] = rej_high + 0.005
    low[conf_bar] = rej_high - 0.008
    close[conf_bar] = rej_high + 0.003

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 100,
    }, index=dates)
    params = dict(
        signal_col=SIGNAL_COL,
        variant="B_confirm",
        lookback_n=lookback_n,
        sweep_atr=0.05,
        wick_min_frac=0.55,
        body_max_frac=0.45,
        close_pos_min=0.60,
        reclaim_frac=0.05,
        confirm_mode="break_high",
        no_resweep=1,
        cooldown_bars=2,
        range_cap_atr=0,
        min_range_atr=0,
    )
    out = c1_lsr_v2(df.copy(), **params)
    sig = out[SIGNAL_COL]
    assert sig.iloc[rej_bar] == 0, f"Expected 0 at rejection bar {rej_bar}; got {sig.iloc[rej_bar]}"
    assert sig.iloc[conf_bar] == 1, f"Expected +1 at confirm bar {conf_bar}; got {sig.iloc[conf_bar]}"


def test_cooldown_suppresses_second_signal():
    """Two valid setups within cooldown window; only first produces signal."""
    n = 80
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    base = 1.02
    lookback_n = 5
    warmup = lookback_n + 14 + 5
    cooldown = 8
    trig1 = warmup
    trig2 = warmup + 4

    open_ = np.full(n, base)
    high = np.full(n, base + 0.005)
    low = np.full(n, base - 0.005)
    close = np.full(n, base)

    def _set_bullish_bar(i, plow):
        sweep_low = plow - 0.025
        rng = 0.10
        open_[i] = base + 0.065
        low[i] = sweep_low
        high[i] = sweep_low + rng
        close[i] = plow + 0.065

    _set_bullish_bar(trig1, base - 0.005)
    for j in range(trig1 + 1, trig2):
        low[j] = base - 0.005
        high[j] = base + 0.005
    _set_bullish_bar(trig2, base - 0.005)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 100,
    }, index=dates)
    params = dict(
        signal_col=SIGNAL_COL,
        variant="A_pin",
        lookback_n=lookback_n,
        sweep_atr=0.05,
        wick_min_frac=0.55,
        body_max_frac=0.50,
        close_pos_min=0.60,
        reclaim_frac=0.05,
        cooldown_bars=cooldown,
        range_cap_atr=0,
        min_range_atr=0,
    )
    out = c1_lsr_v2(df.copy(), **params)
    sig = out[SIGNAL_COL]
    nonzero = np.where(sig.values != 0)[0]
    assert len(nonzero) >= 1, "Expected at least one signal"
    assert sig.iloc[trig1] == 1, "First setup at trig1 should produce +1"
    assert sig.iloc[trig2] == 0, "Second setup within cooldown should be suppressed"
