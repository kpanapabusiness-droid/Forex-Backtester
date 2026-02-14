# tests/test_c1_compression_escape_state_machine.py — C1 CEB v2 state machine tests.

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indicators.confirmation_funcs import c1_compression_escape_state_machine
from tests import _indicator_contracts as ic

SIGNAL_COL = "c1_signal"
VALID_SIGNALS = {-1, 0, 1}


@pytest.fixture(scope="module")
def test_df():
    """Load deterministic OHLCV for CEB v2 tests; skip if data/daily missing."""
    try:
        return ic.load_test_df()
    except FileNotFoundError:
        pytest.skip("data/daily missing or empty; need OHLCV CSV for CEB v2 tests")


def test_returns_dataframe_aligned(test_df):
    """CEB v2 must return a DataFrame with same index and length as input."""
    out = c1_compression_escape_state_machine(test_df.copy(), signal_col=SIGNAL_COL)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(test_df)
    assert out.index.equals(test_df.index)


def test_signal_values_in_valid_set(test_df):
    """CEB v2 must write c1_signal with values only in {-1, 0, 1}."""
    out = c1_compression_escape_state_machine(test_df.copy(), signal_col=SIGNAL_COL)
    assert SIGNAL_COL in out.columns
    warmup = 300
    if len(out) > warmup:
        s = out[SIGNAL_COL].iloc[warmup:]
        uniq = set(pd.to_numeric(s, errors="coerce").dropna().astype(int).unique())
        assert uniq <= VALID_SIGNALS, f"Signal values must be in {VALID_SIGNALS}; got {uniq}"


def test_no_lookahead(test_df):
    """CEB v2 must not use future bars (causality). Modify last row, earlier signals unchanged."""
    if len(test_df) <= 20:
        pytest.skip("test_df too short for causality probe")
    ic.run_causality_probe(
        c1_compression_escape_state_machine,
        test_df,
        call_kwargs={"signal_col": SIGNAL_COL},
        signal_col=SIGNAL_COL,
        future_window=20,
    )


def test_determinism(test_df):
    """Same input -> identical output."""
    df1 = test_df.copy()
    df2 = test_df.copy()
    out1 = c1_compression_escape_state_machine(df1, signal_col=SIGNAL_COL)
    out2 = c1_compression_escape_state_machine(df2, signal_col=SIGNAL_COL)
    pd.testing.assert_series_equal(
        out1[SIGNAL_COL], out2[SIGNAL_COL], check_names=True, check_exact=False
    )


def test_frequency_sanity(test_df):
    """With relaxed params, signal rate between 0.1% and 25% on fixture."""
    relaxed = dict(
        signal_col=SIGNAL_COL,
        L_p=120,
        L_box=12,
        q_atr=0.55,
        q_rng=0.55,
        q_box=0.50,
        W_enter=6,
        N_enter=4,
        W_exit=6,
        N_exit=4,
        K_max=40,
        alpha=0.85,
        beta=1.05,
        gamma=0.50,
        cooldown_bars=5,
    )
    out = c1_compression_escape_state_machine(test_df.copy(), **relaxed)
    s = pd.to_numeric(out[SIGNAL_COL], errors="coerce")
    non_zero = (s != 0) & s.notna()
    rate = non_zero.sum() / len(out)
    assert rate >= 0.001, f"Signal rate {rate:.4f} too low (not all zeros)"
    assert rate <= 0.25, f"Signal rate {rate:.4f} too high (not always-on)"


def test_cooldown_behavior():
    """Non-zero signals separated by at least cooldown_bars+1 (cooldown enforces zeros)."""
    np.random.seed(123)
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = pd.Series(1.0 + np.cumsum(np.random.randn(n) * 0.005))
    high = close + 0.01
    low = close - 0.01
    open_ = close.shift(1).fillna(close.iloc[0])
    df = pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": 100,
    })

    cooldown = 8
    out = c1_compression_escape_state_machine(
        df.copy(),
        signal_col=SIGNAL_COL,
        L_p=60,
        L_box=10,
        q_atr=0.6,
        q_rng=0.6,
        q_box=0.6,
        W_enter=5,
        N_enter=3,
        W_exit=5,
        N_exit=3,
        K_max=20,
        alpha=0.5,
        beta=1.0,
        gamma=0.4,
        cooldown_bars=cooldown,
    )
    sig = out[SIGNAL_COL]
    nonzero_positions = np.where(sig.values != 0)[0]
    if len(nonzero_positions) >= 2:
        gaps = np.diff(nonzero_positions)
        min_gap = int(np.min(gaps))
        assert min_gap >= cooldown, (
            f"Cooldown gating: non-zero signals must be >= {cooldown} bars apart; got min_gap={min_gap}"
        )
